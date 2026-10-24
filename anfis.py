"""
anfis.py  --  Module 4: first-order Sugeno ANFIS with hybrid learning (Jang, 1993)
===================================================================================
Pure NumPy, no ML framework, so every line can be explained in the viva.

THE 5 LAYERS (R rules, d inputs)
  L1  Fuzzify    mu_ij = exp(-(x_j - c_ij)^2 / (2 s_ij^2))      how well input j fits rule i
  L2  Firing     w_i   = prod_j mu_ij                           "AND" of all conditions
  L3  Normalise  wbar_i = w_i / sum_k w_k                       rule i's share of the decision
  L4  Consequent f_i   = p_i . [x, 1]                           a linear model per rule (Sugeno)
  L5  Output     y     = sum_i wbar_i * f_i                     weighted average

  A rule reads:  IF x1 is ~c_i1 AND x2 is ~c_i2 ... THEN y = p_i . [x, 1].
  Target is 0 (healthy) / 1 (mastitis); y is a RISK SCORE (threshold 0.5).

HYBRID LEARNING (two parameter groups, two methods, every epoch)
  Forward pass : premise params (c, s) fixed  ->  y is LINEAR in the consequent
                 params p  ->  solve them EXACTLY by (ridge) least squares.
  Backward pass: consequent params fixed       ->  gradient descent on (c, s)
                 using the analytic gradient of E = 0.5 * mean((y - t)^2).
  Why hybrid: LSE jumps straight to the optimal p each epoch, so gradient
  descent only searches the much smaller (c, s) space -> faster, fewer local minima.

NUMERICAL NOTE
  w_i can underflow (product of many small Gaussians). We work in the LOG domain:
  log w_i = -0.5 * sum_j ((x_j - c_ij)/s_ij)^2 and get wbar via a softmax.
  The gradient is then clean:  dE/dlogw_i = e * wbar_i * (f_i - y),  e = (y - t)/n.
"""
from __future__ import annotations

import numpy as np


class ANFIS:
    def __init__(self, centers: np.ndarray, sigmas: np.ndarray,
                 ridge: float = 1e-3, sigma_min: float = 1e-2):
        self.c = np.array(centers, dtype=float)       # (R, d)
        self.s = np.array(sigmas, dtype=float)        # (R, d)
        self.R, self.d = self.c.shape
        self.theta = np.zeros((self.R, self.d + 1))   # consequent params (R, d+1)
        self.ridge, self.sigma_min = ridge, sigma_min
        self.history: list[dict] = []
        self.best_epoch_ = 0

    # ---------------- forward pass (layers 1-5) ---------------- #
    def _logw(self, X):
        diff = X[:, None, :] - self.c[None, :, :]                 # (n, R, d)
        return -0.5 * ((diff / self.s[None]) ** 2).sum(axis=2)    # (n, R)

    def _wbar(self, X):
        lw = self._logw(X)
        lw = lw - lw.max(axis=1, keepdims=True)                   # softmax stabilisation
        w = np.exp(lw)
        return w / w.sum(axis=1, keepdims=True)                   # layer 3

    @staticmethod
    def _aug(X):
        return np.column_stack([X, np.ones(len(X))])              # [x, 1]

    def _rule_outputs(self, X):
        return self._aug(X) @ self.theta.T                        # (n, R)   layer 4

    def predict_score(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (self._wbar(X) * self._rule_outputs(X)).sum(axis=1)  # layer 5

    def predict(self, X, thr: float = 0.5) -> np.ndarray:
        return (self.predict_score(X) >= thr).astype(int)

    def loss(self, X, y) -> float:
        return 0.5 * float(np.mean((self.predict_score(X) - y) ** 2))

    # ---------------- hybrid learning ---------------- #
    def _lse(self, X, y):
        """Forward pass of hybrid learning: exact ridge least squares for theta."""
        n = len(X)
        A = (self._wbar(X)[:, :, None] * self._aug(X)[:, None, :]).reshape(n, -1)  # (n, R*(d+1))
        G = A.T @ A + self.ridge * np.eye(A.shape[1])
        self.theta = np.linalg.solve(G, A.T @ y).reshape(self.R, self.d + 1)

    def _grad_premise(self, X, y):
        """Backward pass: analytic dE/dc, dE/ds with consequents fixed."""
        n = len(X)
        diff = X[:, None, :] - self.c[None, :, :]                 # (n, R, d)
        wbar, f = self._wbar(X), self._rule_outputs(X)
        yhat = (wbar * f).sum(axis=1)
        e = (yhat - y) / n                                        # dE/dyhat
        g = e[:, None] * wbar * (f - yhat[:, None])               # dE/dlogw  (n, R)
        dc = np.einsum("nr,nrd->rd", g, diff / self.s[None] ** 2)
        ds = np.einsum("nr,nrd->rd", g, diff ** 2 / self.s[None] ** 3)
        return dc, ds

    def _snapshot(self):
        return self.c.copy(), self.s.copy(), self.theta.copy()

    def _restore(self, snap):
        self.c, self.s, self.theta = (a.copy() for a in snap)

    def fit(self, X, y, X_val=None, y_val=None, epochs: int = 100, lr: float = 0.01,
            optimizer: str = "adam", patience: int = 15, verbose: bool = False):
        X = np.asarray(X, float); y = np.asarray(y, float)
        has_val = X_val is not None
        mC, vC = np.zeros_like(self.c), np.zeros_like(self.c)     # Adam state
        mS, vS = np.zeros_like(self.s), np.zeros_like(self.s)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best_val, best_snap, since_best = np.inf, None, 0
        self.history = []

        for ep in range(1, epochs + 1):
            self._lse(X, y)                                       # FORWARD: LSE
            tr = self.loss(X, y)
            rec = {"epoch": ep, "train_loss": tr}
            if has_val:
                rec["val_loss"] = self.loss(np.asarray(X_val, float), np.asarray(y_val, float))
                if rec["val_loss"] < best_val - 1e-6:
                    best_val, best_snap, since_best = rec["val_loss"], self._snapshot(), 0
                    self.best_epoch_ = ep
                else:
                    since_best += 1
            else:
                self.best_epoch_ = ep
            self.history.append(rec)
            if verbose and ep % 10 == 0:
                print(rec)
            if has_val and since_best >= patience:
                break

            dc, ds = self._grad_premise(X, y)                     # BACKWARD: gradient descent
            if optimizer == "adam":
                for p, g, m, v in ((self.c, dc, mC, vC), (self.s, ds, mS, vS)):
                    m *= b1; m += (1 - b1) * g
                    v *= b2; v += (1 - b2) * g ** 2
                    p -= lr * (m / (1 - b1 ** ep)) / (np.sqrt(v / (1 - b2 ** ep)) + eps)
            else:                                                 # plain gradient descent
                self.c -= lr * dc
                self.s -= lr * ds
            self.s = np.maximum(self.s, self.sigma_min)

        if has_val and best_snap is not None:
            self._restore(best_snap)
        else:
            self._lse(X, y)                                       # re-fit theta for final (c, s)
        return self

    # ---------------- interpretability ---------------- #
    def save(self, path):
        np.savez(path, c=self.c, s=self.s, theta=self.theta)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(z["c"], z["s"]); m.theta = z["theta"]
        return m
