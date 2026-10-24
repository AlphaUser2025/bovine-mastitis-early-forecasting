"""
subtractive_clustering.py  --  Module 3: automatic rule generation (Chiu, 1994)
================================================================================
PROBLEM IT SOLVES
  A grid-partition ANFIS with m membership functions per input and N inputs needs
  m^N rules (3 MFs x 7 inputs = 2187 rules). Subtractive clustering instead
  finds a SMALL number of "typical points" in the data; each becomes ONE rule.

THE IDEA IN 4 LINES (first principles)
  1. Give every data point a "potential" = how crowded its neighbourhood is:
         P_i = sum_j exp( -alpha * ||x_i - x_j||^2 ),   alpha = 4 / ra^2
     (ra = neighbourhood radius. Close neighbours add ~1, far ones add ~0.)
  2. The point with the highest potential is the first cluster centre.
  3. Subtract that centre's influence from everybody nearby:
         P_i <- P_i - P_k* exp( -beta * ||x_i - x_k*||^2 ),   beta = 4 / rb^2,  rb = 1.5*ra
     (rb > ra so points right next to a centre cannot become the next centre.)
  4. Repeat: accept the next-best point if its potential is high enough
     (> 0.5 * P_1), stop if it is too low (< 0.15 * P_1); in between, accept only
     if it is also far enough from existing centres (Chiu's gray-zone test).

FROM CENTRES TO RULES
  Each centre c_i  ->  one rule with a Gaussian membership function per input:
         mu_ij(x_j) = exp( -(x_j - c_ij)^2 / (2 sigma^2) ),   sigma = ra / sqrt(8)
  (sqrt(8) comes from alpha = 4/ra^2 = 1/(2 sigma^2).)

JOINT SPACE (default)
  Like MATLAB genfis2 / Chiu's paper, we cluster on [inputs, label]. The label is
  0/1, so a cluster is mostly one class -> rules become class-specific. The label
  is used ONLY to find centres from TRAINING data; prediction never uses it.
"""
from __future__ import annotations

import numpy as np


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape (len(A), len(B))."""
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


def subtractive_clustering(X: np.ndarray, ra: float = 0.5, rb_factor: float = 1.5,
                           accept: float = 0.5, reject: float = 0.15,
                           max_clusters: int = 15, chunk: int = 1000) -> np.ndarray:
    """Return cluster centres, shape (R, D). X must already be scaled to ~[0, 1]."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    alpha = 4.0 / ra ** 2
    beta = 4.0 / (rb_factor * ra) ** 2

    # Step 1: initial potentials (chunked so memory stays small for ~5000 rows)
    P = np.empty(n)
    for s in range(0, n, chunk):
        P[s:s + chunk] = np.exp(-alpha * _sqdist(X[s:s + chunk], X)).sum(axis=1)

    p1 = P.max()
    centers: list[np.ndarray] = []
    while len(centers) < max_clusters:
        k = int(P.argmax())
        pk = P[k]
        if pk <= 0:
            break
        if not centers or pk > accept * p1:
            ok = True
        elif pk < reject * p1:
            break
        else:  # gray zone: accept only if far enough from existing centres
            dmin = np.sqrt(_sqdist(X[k:k + 1], np.array(centers)).min())
            ok = (dmin / ra + pk / p1) >= 1.0
        if ok:
            centers.append(X[k].copy())
            P = np.maximum(P - pk * np.exp(-beta * _sqdist(X, X[k:k + 1]).ravel()), 0.0)
        else:
            P[k] = 0.0  # reject this candidate, look at the next-best point
    return np.array(centers)


def initial_rule_params(X: np.ndarray, y: np.ndarray | None, ra: float = 0.5,
                        use_target: bool = True, max_rules: int = 12,
                        **kw) -> tuple[np.ndarray, np.ndarray]:
    """
    Centres + sigmas for the ANFIS premise (IF-part).
    Returns (centers[R, d], sigmas[R, d]) in INPUT space only.
    """
    X = np.asarray(X, dtype=float)
    Z = np.column_stack([X, np.asarray(y, dtype=float)]) if (use_target and y is not None) else X
    C = subtractive_clustering(Z, ra=ra, max_clusters=max_rules, **kw)[:, :X.shape[1]]
    sigma = ra / np.sqrt(8.0)
    return C, np.full_like(C, sigma)
