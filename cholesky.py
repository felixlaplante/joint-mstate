import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict


class _MH:
    """A simple Metropolis-Hastings class."""

    def __init__(
        self,
        log_prob_fn,
        curr_z,
        curr_log_prob,
        step_size,
        accept_step_size,
        accept_target,
    ):
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size
        self.accept_step_size = accept_step_size
        self.accept_target = accept_target
        self.curr_z = curr_z
        self.curr_log_prob = curr_log_prob

    def __call__(self):
        self.curr_z = self.curr_z.detach()
        self.curr_log_prob = self.curr_log_prob.detach()
        prop_z = self.curr_z + torch.randn_like(self.curr_z) * self.step_size
        prop_log_prob = self.log_prob_fn(prop_z)
        log_prob_diff = prop_log_prob - self.curr_log_prob
        target = torch.log(torch.clip(torch.rand_like(log_prob_diff), 1e-8))
        accept = target < log_prob_diff
        self.curr_z[accept] = prop_z[accept]
        self.curr_log_prob[accept] = prop_log_prob[accept]
        self.step_size *= torch.exp(
            (accept.to(torch.float32).mean() - self.accept_target)
            * self.accept_step_size
        )
        return self.curr_z, self.curr_log_prob


class Fun:
    """A simple callable wrapper for a function with specified input and output dimensions."""

    def __init__(self, fun, input_dim, output_dim):
        self.fun = fun
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, *args):
        return self.fun(*args)


class JointModel:
    """A joint model combining longitudinal and hazard components with MCMC and optimization routines."""

    def __init__(
        self,
        h,
        f,
        surv,
        n_quad=16,
        n_bissect=16,
    ):
        self.h = h
        self.f = f
        self.surv = surv
        self.params = {}

        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        self.std_nodes = torch.tensor(nodes, dtype=torch.float32)
        self.std_weights = torch.tensor(weights, dtype=torch.float32)

        self.n_bissect = n_bissect

        self.fit_ = False

    def _cholesky(self, flat, n):
        L = torch.zeros(n, n, dtype=flat.dtype)
        iu = torch.tril_indices(n, n)
        L[iu[0], iu[1]] = flat
        return L

    def _log_hazard(self, t0, t1, x, psi, alpha, beta, log_lambda0, g):
        base = log_lambda0(t1, t0)
        mod = g(t1, x, psi)

        return base + torch.einsum("ijk,k->ij", mod, alpha) + x @ beta.unsqueeze(1)

    def _cum_hazard(self, t0, t1, x, psi, alpha, beta, log_lambda0, g):
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)
        mid = 0.5 * (t0 + t1)
        half = 0.5 * (t1 - t0)
        ts = mid + half * self.std_nodes

        vals = torch.exp(self._log_hazard(t0, ts, x, psi, alpha, beta, log_lambda0, g))
        return half.flatten() * (vals * self.std_weights).sum(dim=1)

    def _log_and_cum_hazard(self, t0, t1, x, psi, alpha, beta, log_lambda0, g):
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)
        mid = 0.5 * (t0 + t1)
        half = 0.5 * (t1 - t0)
        ts = torch.cat([t1, mid + half * self.std_nodes], axis=1)

        temp = self._log_hazard(t0, ts, x, psi, alpha, beta, log_lambda0, g)
        log_hazard, vals = temp[:, :1], torch.exp(temp[:, 1:])

        return log_hazard.flatten(), half.flatten() * (vals * self.std_weights).sum(
            dim=1
        )

    def _hazard_ll(self, psi):
        ll = torch.zeros(self.n, dtype=torch.float32)

        for d, info in self._buckets.items():
            alpha, beta = self.params["alpha"][d], self.params["beta"][d]
            idx, t0, t1, obs = info["idx"], info["t0"], info["t1"], info["obs"]

            obs_ll, alts_ll = self._log_and_cum_hazard(
                t0, t1, self.x[idx], psi[idx], alpha, beta, **self.surv[d]
            )

            vals = obs * obs_ll - alts_ll
            ll.scatter_add_(0, idx, vals)

        return ll

    def _long_ll(self, psi):
        diff = self.y - self.h(self.t, self.x, psi) * self._valid
        R_inv = self._cholesky(self.params["R_inv"], self.h.output_dim)
        log_det_R = -torch.diag(R_inv).sum() * 2
        R_inv.view(-1)[:: self.h.output_dim + 1] = torch.exp(
            R_inv.view(-1)[:: self.h.output_dim + 1]
        )
        R_inv = R_inv @ R_inv.T
        quad_form = torch.einsum("ijk,kl,ijl->i", diff, R_inv, diff)
        return -0.5 * (log_det_R * self._n_valid + quad_form)

    def _pr_ll(self, b):
        Q_inv = self._cholesky(self.params["Q_inv"], self.f.input_dim[1])
        log_det_Q = -torch.diag(Q_inv).sum() * 2
        Q_inv.view(-1)[:: self.f.input_dim[1] + 1] = torch.exp(
            Q_inv.view(-1)[:: self.f.input_dim[1] + 1]
        )
        Q_inv = Q_inv @ Q_inv.T
        quad_form = torch.einsum("ik,kl,il->i", b, Q_inv, b)
        return -0.5 * (log_det_Q + quad_form)

    def _ll(self, b):
        psi = self.f(self.params["gamma"], b)
        return self._long_ll(psi) + self._hazard_ll(psi) + self._pr_ll(b)

    def _mcmc(self, mh, warmup, batch_size):
        with torch.no_grad():
            for _ in range(warmup):
                mh()
        ll = 0
        for _ in range(batch_size):
            curr_b, curr_ll = mh()
            ll += curr_ll.sum()
        return ll / batch_size, curr_b, curr_ll

    def _build_buckets(self, T, C):
        surv = set(self.surv)
        alt_map = defaultdict(list)
        for a, b in self.surv:
            alt_map[a].append(b)
        buf = defaultdict(lambda: [[], [], [], []])
        for i, tr in enumerate(T):
            for (t0, s0), (t1, s1) in zip(tr, tr[1:] + [(float(C[i]), torch.nan)]):
                if t0 >= t1:
                    continue
                for virt_s1 in alt_map.get(s0, ()):
                    key = (s0, virt_s1)
                    if key in surv:
                        buf[key][0].append(i)
                        buf[key][1].append(t0)
                        buf[key][2].append(t1)
                        buf[key][3].append(virt_s1 == s1)
        return {
            k: {
                "idx": torch.tensor(v[0], dtype=torch.int64),
                "t0": torch.tensor(v[1], dtype=torch.float32).view(-1, 1),
                "t1": torch.tensor(v[2], dtype=torch.float32).view(-1, 1),
                "obs": torch.tensor(v[3], dtype=bool),
            }
            for k, v in buf.items()
        }

    def fit(
        self,
        x,
        t,
        y,
        T,
        C,
        optimizer,
        optimizer_params,
        n_iter,
        batch_size,
        callback=None,
        n_iter_fim=1000,
        K=10.0,
        step_size=1.0,
        accept_step_size=0.1,
        accept_target=0.234,
    ):

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.t = torch.as_tensor(t, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self._valid = ~torch.isnan(self.y)
        self._n_valid = self._valid.any(dim=2).sum(dim=1)
        self.y = torch.nan_to_num(self.y)
        self.T = T
        self.C = torch.as_tensor(C, dtype=torch.float32)
        self.n, self.p = self.x.shape

        self.params["gamma"] = torch.zeros(
            self.f.input_dim[0], dtype=torch.float32, requires_grad=True
        )
        self.params["Q_inv"] = torch.zeros(
            self.f.input_dim[1] * (self.f.input_dim[1] + 1) // 2,
            dtype=torch.float32,
            requires_grad=True,
        )
        self.params["R_inv"] = torch.zeros(
            self.h.output_dim * (self.h.output_dim + 1) // 2,
            dtype=torch.float32,
            requires_grad=True,
        )
        self.params["alpha"] = {
            key: torch.zeros(
                self.surv[key]["g"].output_dim, dtype=torch.float32, requires_grad=True
            )
            for key in self.surv.keys()
        }
        self.params["beta"] = {
            key: torch.zeros(self.p, dtype=torch.float32, requires_grad=True)
            for key in self.surv.keys()
        }

        params = [v for v in self.params.values() if not isinstance(v, dict)]
        params += [
            *self.params["alpha"].values(),
            *self.params["beta"].values(),
        ]
        optimizer = optimizer(params=params, **optimizer_params)

        warmup = int(K)
        curr_b = torch.zeros((self.n, self.f.input_dim[1]), dtype=torch.float32)
        curr_ll = torch.full((self.n,), -torch.inf, dtype=torch.float32)
        mh = _MH(self._ll, curr_b, curr_ll, step_size, accept_step_size, accept_target)

        self._buckets = self._build_buckets(self.T, self.C)

        for _ in tqdm(range(n_iter), "Fitting..."):
            ll, curr_b, curr_ll = self._mcmc(mh, warmup, batch_size)
            nll = -ll

            params_before = [p.detach().clone() for p in params]

            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

            step_norm = torch.sqrt(
                sum(((p - pb) ** 2).sum() for pb, p in zip(params_before, params))
            ).item()

            warmup = int(step_norm * K)

            if callback is not None:
                callback()

        d = sum(p.numel() for p in params)
        self.fim = torch.zeros(d, d)

        for _ in tqdm(range(n_iter_fim), desc="Getting FIM..."):
            ll, curr_b, curr_ll = self._mcmc(mh, 1, 1)

            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

            ll.backward()
            grad = torch.cat([p.grad.view(-1) for p in params])
            self.fim += torch.outer(grad, grad) / n_iter_fim

        self.fit_ = True

    def get_ci(self, alpha=0.05):
        assert self.fit_

        params = [v for v in self.params.values()]
        params = torch.concat([p.detach().flatten() for p in params])

        se = torch.sqrt(torch.linalg.pinv(self.fim).diag())
        q = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2))

        lower = params - q * se
        upper = params + q * se

        ci = {}
        i = 0
        for key, val in self.params.items():
            if isinstance(val, dict):
                ci[key] = {}
                for subkey, subval in val.items():
                    n = subval.numel()
                    shape = subval.shape
                    ci[key][subkey] = {
                        "lower": lower[i : i + n].view(shape),
                        "upper": upper[i : i + n].view(shape),
                    }
                    i += n
            else:
                n = val.numel()
                shape = val.shape
                ci[key] = {
                    "lower": lower[i : i + n].view(shape),
                    "upper": upper[i : i + n].view(shape),
                }
                i += n

        return ci

    def _sample(
        self,
        t_left,
        t_right,
        x,
        psi,
        alpha,
        beta,
        log_lambda0,
        g,
        t_surv,
    ):
        n = x.shape[0]

        t0 = t_left.clone().view(-1, 1)
        t_left, t_right = t_left.view(-1, 1), t_right.view(-1, 1)
        target = -torch.log(torch.clip(torch.rand(n), 1e-8))
        if t_surv is not None:
            t_surv = t_surv.view(-1, 1)
            res = self._cum_hazard(
                t0,
                t_surv,
                x,
                psi,
                alpha,
                beta,
                log_lambda0,
                g,
            )
            target += res
        for _ in range(self.n_bissect):
            t_mid = 0.5 * (t_left + t_right)
            res = self._cum_hazard(
                t0,
                t_mid,
                x,
                psi,
                alpha,
                beta,
                log_lambda0,
                g,
            )

            accept = res < target
            t_left[accept] = t_mid[accept]
            t_right[~accept] = t_mid[~accept]
        return t_right.flatten()

    def sample(self, T, C, x, psi, t_surv=None, max_iter=100):
        x = torch.as_tensor(x, dtype=torch.float32)
        psi = torch.as_tensor(psi, dtype=torch.float32)

        n = x.shape[0]

        last_alts = self._build_buckets([trajectory[-1:] for trajectory in T], C)
        T = list(map(list, T))
        for i in range(max_iter):
            if last_alts == {}:
                break
            t_cand = torch.full(
                (
                    n,
                    len(last_alts),
                ),
                torch.inf,
                dtype=torch.float32,
            )

            for j, (d, info) in enumerate(last_alts.items()):
                alpha, beta = self.params["alpha"][d], self.params["beta"][d]

                idx, t0, t1 = info["idx"], info["t0"], info["t1"]
                t_sample = self._sample(
                    t0,
                    t1 + 1,
                    x[idx],
                    psi[idx],
                    alpha,
                    beta,
                    **self.surv[d],
                    t_surv=t_surv[idx] if not i and t_surv is not None else None,
                )

                t_cand[idx, j] = t_sample
            min_t, argmin_t = torch.min(t_cand, dim=1)
            valid = torch.nonzero(torch.isfinite(min_t)).flatten()
            for i in valid:
                n1 = int(argmin_t[i])
                t1 = min_t[i].item()
                s1 = list(last_alts.keys())[n1][1]
                T[i].append((t1, s1))
            last_alts = self._build_buckets([trajectory[-1:] for trajectory in T], C)
        return [
            trajectory[:-1] if trajectory[-1][0] > C[i] else trajectory
            for i, trajectory in enumerate(T)
        ]

    def predict_surv(
        self,
        C_max,
        x,
        t,
        y,
        T,
        C,
        n_iter_b,
        n_iter_T,
        warmup,
        max_iter=100,
    ):
        assert self.fit_

        dummy_jm = JointModel(self.h, self.f, self.surv)
        dummy_jm.params = {
            k: v.detach().clone()
            for k, v in self.params.items()
            if not isinstance(v, dict)
        }
        dummy_jm.params["alpha"] = {
            key: self.params["alpha"][key].detach().clone()
            for key in self.params["alpha"].keys()
        }
        dummy_jm.params["beta"] = {
            key: self.params["beta"][key].detach().clone()
            for key in self.params["beta"].keys()
        }

        dummy_jm.x = torch.as_tensor(x, dtype=torch.float32)
        dummy_jm.t = torch.as_tensor(t, dtype=torch.float32)
        dummy_jm.y = torch.as_tensor(y, dtype=torch.float32)
        dummy_jm._valid = ~torch.isnan(dummy_jm.y)
        dummy_jm._n_valid = dummy_jm._valid.any(dim=2).sum(dim=1)
        dummy_jm.y = torch.nan_to_num(dummy_jm.y)
        dummy_jm.T = T
        dummy_jm.C = torch.as_tensor(C, dtype=torch.float32)
        dummy_jm.n, dummy_jm.p = dummy_jm.x.shape

        dummy_jm._buckets = self._build_buckets(dummy_jm.T, dummy_jm.C)

        curr_b = torch.zeros((dummy_jm.n, dummy_jm.f.input_dim[1]), dtype=torch.float32)
        curr_ll = torch.full((dummy_jm.n,), -torch.inf, dtype=torch.float32)

        x_rep = dummy_jm.x.repeat(n_iter_T, 1)
        T_rep = dummy_jm.T * n_iter_T
        C_rep = dummy_jm.C.repeat(n_iter_T)
        C_max = torch.as_tensor(C_max, dtype=torch.float32)
        C_max_rep = C_max.repeat(n_iter_T)

        T_pred = []
        for _ in tqdm(range(n_iter_b), "Predicting..."):
            for _ in range(warmup):
                curr_b, curr_ll = dummy_jm._mh(
                    curr_b,
                    curr_ll,
                )

            psi_rep = dummy_jm.f(dummy_jm.params["gamma"], curr_b).repeat(n_iter_T, 1)

            res = dummy_jm.sample(
                T_rep,
                C_max_rep,
                x_rep,
                psi_rep,
                C_rep,
                max_iter,
            )
            chunks = [
                res[i * dummy_jm.n : (i + 1) * dummy_jm.n] for i in range(n_iter_T)
            ]

            T_pred.append(chunks)

        del dummy_jm
        return T_pred
