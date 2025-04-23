import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import minimize_scalar


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
        h: Fun,
        f: Fun,
        surv: dict,
        K: float = 10.0,
        step_size: float = 1.0,
        accept_step_size: float = 0.1,
        accept_target: float = None,
        n_quad: int = 16,
        n_dichotomy: int = 16,
    ):
        self.h = h
        self.f = f
        self.surv = surv
        self.K = torch.tensor(K)
        self.step_size = torch.tensor(step_size)
        self.accept_step_size = torch.tensor(accept_step_size)
        self.accept_target = (
            torch.tensor(accept_target)
            if accept_target is not None
            else self._default_accept()
        )
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        self.std_nodes = torch.tensor(nodes, dtype=torch.float32)
        self.std_weights = torch.tensor(weights, dtype=torch.float32)
        self.n_dichotomy = n_dichotomy
        self.fit_ = False

    def _default_accept(self):
        l = minimize_scalar(
            lambda l: -(l**2) * norm.cdf(-0.5 * l), bounds=(2, 3), method="bounded"
        ).x
        return torch.tensor(2.0 * norm.cdf(-0.5 * l))

    def _log_hazard(self, t0, t1, x, psi, alpha, beta, log_lambda0, g, reset):
        base = log_lambda0(t1 - t0) if reset else log_lambda0(t1)
        mod = g(t1, x, psi)
        return (
            base + torch.einsum("ijk,k->ij", mod, alpha) + x.matmul(beta).unsqueeze(1)
        )

    def _cum_hazard(self, t0, t1, x, psi, alpha, beta, log_lambda0, g, reset):
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)
        mid = (t0 + t1) / 2
        half = (t1 - t0) / 2
        ts = mid + half * self.std_nodes
        vals = torch.exp(
            self._log_hazard(t0, ts, x, psi, alpha, beta, log_lambda0, g, reset)
        )
        return half.flatten() * (vals * self.std_weights).sum(1)

    def _long_ll(self, psi):
        diff = self.y - self.h(self.t, self.x, psi)
        R_inv = torch.exp(-self.log_R)
        log_det_R = self.log_R.sum()
        quad_form = torch.einsum("ijk,k,ijk->ij", diff, R_inv, diff)
        return (-0.5 * log_det_R - 0.5 * quad_form).sum(1)

    def _hazard_ll(self, psi):
        ll = torch.zeros(self.n)

        for d, info in self.trans.items():
            alpha, beta = self.alpha[d], self.beta[d]
            idx, t0, t1 = info["idx"], info["t0"], info["t1"]
            trans_ll = self._log_hazard(
                t0,
                t1,
                self.x[idx],
                psi[idx],
                alpha,
                beta,
                **self.surv[d],
            ).flatten()
            ll.scatter_add_(0, idx, trans_ll)
        for d, info in self.alts.items():
            alpha, beta = self.alpha[d], self.beta[d]
            idx, t0, t1 = info["idx"], info["t0"], info["t1"]
            alts_ll = -self._cum_hazard(
                t0, t1, self.x[idx], psi[idx], alpha, beta, **self.surv[d]
            )
            ll.scatter_add_(0, idx, alts_ll)
        return ll

    def _pr_ll(self, b):
        diff = b - self.mu
        Q_inv = torch.exp(-self.log_Q)
        log_det_Q = torch.sum(self.log_Q)
        quad_form = torch.einsum("ij,j,ij->i", diff, Q_inv, diff)
        return -0.5 * log_det_Q - 0.5 * quad_form

    def _ll(self, b):
        psi = self.f(self.gamma, b)
        return self._long_ll(psi) + self._hazard_ll(psi) + self._pr_ll(b)

    def _mh(self, curr_b, curr_ll):
        prop_b = curr_b + torch.randn_like(curr_b) * self.step_size
        prop_logLik = self._ll(prop_b)
        logLik_diff = prop_logLik - curr_ll
        accept = torch.rand(curr_b.shape[0]) < torch.exp(logLik_diff)
        curr_b[accept] = prop_b[accept]
        curr_ll[accept] = prop_logLik[accept]
        self.step_size *= torch.exp(
            (accept.type(torch.float32).mean() - self.accept_target)
            * self.accept_step_size
        )
        return curr_b, curr_ll

    def _mcmc(self, curr_b, curr_ll, burn_in, batch_size):
        with torch.no_grad():
            for _ in range(burn_in):
                curr_b, curr_ll = self._mh(
                    curr_b.detach(),
                    curr_ll.detach(),
                )
        ll = 0
        for _ in range(batch_size):
            curr_b, curr_ll = self._mh(
                curr_b.detach(),
                curr_ll.detach(),
            )
            ll += curr_ll.sum()
        return curr_b, curr_ll, ll / batch_size

    def _build(self, T, C, mode):
        assert mode in ("trans", "alts")
        D = {k: {"i": [], "s": [], "e": []} for k in self.surv}
        for i, tr in enumerate(T):
            if mode == "trans":
                for (t0, s0), (t1, s1) in zip(tr, tr[1:]):
                    d = D[(s0, s1)]
                    d["i"] += [i]
                    d["s"] += [t0]
                    d["e"] += [t1]
            else:
                for (t0, s0), (t1, _) in zip(tr, tr[1:] + [(C[i], -1)]):
                    if t0 >= t1:
                        continue
                    for k, d in D.items():
                        if k[0] == s0:
                            d["i"] += [i]
                            d["s"] += [t0]
                            d["e"] += [t1]
        return {
            k: {
                "idx": torch.tensor(v["i"], dtype=int),
                "t0": torch.tensor(v["s"]).view(-1, 1),
                "t1": torch.tensor(v["e"]).view(-1, 1),
            }
            for k, v in D.items()
            if v["i"]
        }

    def fit(
        self,
        x,
        t,
        y,
        T,
        C,
        optimizer,
        lr,
        n_iter,
        batch_size,
        callback=None,
    ):
        self.n, self.p = x.shape

        self.x = x
        self.t = t
        self.y = y
        self.T = T
        self.C = C

        self.gamma = torch.zeros(self.f.input_dim[0], requires_grad=True)
        self.mu = torch.zeros(self.f.input_dim[1], requires_grad=True)
        self.log_Q = torch.zeros(self.f.input_dim[1], requires_grad=True)
        self.log_R = torch.zeros(self.h.output_dim, requires_grad=True)
        self.alpha = {
            key: torch.zeros(self.surv[key]["g"].output_dim, requires_grad=True)
            for key in self.surv.keys()
        }
        self.beta = {
            key: torch.zeros(p, requires_grad=True) for key in self.surv.keys()
        }

        params = (
            [self.gamma, self.mu, self.log_Q, self.log_R]
            + list(self.alpha.values())
            + list(self.beta.values())
        )
        optimizer = optimizer(params=params, lr=lr)

        burn_in = int(torch.ceil(self.K).item())
        curr_b = self.mu.detach().repeat(self.n, 1)
        curr_ll = torch.full((n,), -torch.inf)

        trans, alts = self._build(T, C, mode="trans"), self._build(T, C, mode="alts")

        self.trans = trans
        self.alts = alts

        for _ in tqdm(range(n_iter), "Fitting..."):
            curr_b, curr_ll, ll = self._mcmc(curr_b, curr_ll, burn_in, batch_size)
            nll = -ll

            params_before = [p.detach().clone() for p in params]

            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

            step_norm = torch.sqrt(
                sum(((p - pb) ** 2).sum() for pb, p in zip(params_before, params))
            ).item()

            burn_in = int(torch.ceil(step_norm * self.K).item())

            if callback is not None:
                callback()

        self.fit_ = True

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
        reset,
    ):
        n = x.shape[0]

        t0 = t_left.clone().view(-1, 1)
        t_left, t_right = t_left.clone().view(-1, 1), t_right.clone().view(-1, 1)
        target = -torch.log(torch.clip(torch.rand(n), 1e-8))

        for _ in range(self.n_dichotomy):
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
                reset,
            )

            accept = res < target
            t_left[accept] = t_mid[accept]
            t_right[~accept] = t_mid[~accept]

        return t_right.flatten()

    def sample(self, T_init, C, x, psi):
        n = x.shape[0]

        T = copy.deepcopy(T_init)
        last_alts = self._build([trajectory[-1:] for trajectory in T], C, mode="alts")
        while last_alts != {}:
            t_cand = torch.full(
                (
                    n,
                    len(last_alts),
                ),
                torch.inf,
            )
            for j, (d, info) in enumerate(last_alts.items()):
                alpha, beta = self.alpha[d], self.beta[d]

                idx, t0, t1 = info["idx"], info["t0"], info["t1"]
                t_sample = self._sample(
                    t0,
                    t1 + 1,
                    x[idx],
                    psi[idx],
                    alpha,
                    beta,
                    **self.surv[d],
                )
                t_cand[:, j].scatter_(0, idx, t_sample)

            min_t, argmin_t = torch.min(t_cand, dim=1)
            for i, (t1, n_d) in enumerate(zip(min_t, argmin_t)):
                if t1 == torch.inf:
                    continue
                s1 = list(last_alts)[n_d][1]
                T[i].append((t1, s1))

            last_alts = self._build(
                [trajectory[-1:] for trajectory in T], C, mode="alts"
            )

        T = [
            trajectory[:-1] if trajectory[-1][0] > C[i] else trajectory
            for i, trajectory in enumerate(T)
        ]

        return T

    def predict(self, u, x, t, y, T, C, n_iter, n_samples, burn_in):
        assert self.fit_

        dummy_jm = JointModel(self.h, self.f, self.surv)
        dummy_jm.gamma = self.gamma.detach().clone()
        dummy_jm.mu = self.mu.detach().clone()
        dummy_jm.log_Q = self.log_Q.detach().clone()
        dummy_jm.log_R = self.log_R.detach().clone()
        dummy_jm.alpha = {
            key: self.alpha[key].detach().clone() for key in self.alpha.keys()
        }
        dummy_jm.beta = {
            key: self.beta[key].detach().clone() for key in self.beta.keys()
        }

        dummy_jm.n, dummy_jm.p = x.shape

        dummy_jm.x = x
        dummy_jm.t = t
        dummy_jm.y = y
        dummy_jm.T = T
        dummy_jm.C = C

        dummy_jm.trans = self.trans
        dummy_jm.alts = self.alts

        curr_b = dummy_jm.mu.repeat(dummy_jm.n, 1)
        curr_ll = torch.full((dummy_jm.n,), -torch.inf)

        T_pred = []
        for _ in tqdm(range(burn_in), "Burning-in..."):
            curr_b, curr_ll = dummy_jm._mh(
                curr_b,
                curr_ll,
            )
        for _ in tqdm(range(n_iter), "Predicting..."):
            curr_b, curr_ll = dummy_jm._mh(curr_b, curr_ll)
            T_pred += [
                dummy_jm.sample(T, u, x, dummy_jm.f(dummy_jm.gamma, curr_b))
                for _ in range(n_samples)
            ]

        del dummy_jm
        return T_pred
