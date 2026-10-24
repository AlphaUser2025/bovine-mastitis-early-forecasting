"""
test_modules.py  --  quick correctness tests for Modules 3-4 (run: python test_modules.py)
Each test proves one claim you can make in the viva.
"""
import numpy as np

from anfis import ANFIS
from subtractive_clustering import subtractive_clustering


def test_clustering_finds_blobs():
    rng = np.random.default_rng(0)
    blobs = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.85)]
    X = np.vstack([rng.normal(b, 0.03, (80, 2)) for b in blobs])
    C = subtractive_clustering(X, ra=0.3)
    assert len(C) == 3, f"expected 3 centres, got {len(C)}"
    for b in blobs:  # every true blob has a centre close to it
        assert np.min(np.linalg.norm(C - np.array(b), axis=1)) < 0.1
    print("OK  clustering recovers 3 well-separated blobs")


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    X, y = rng.random((60, 3)), rng.integers(0, 2, 60).astype(float)
    m = ANFIS(rng.random((4, 3)), np.full((4, 3), 0.4))
    m._lse(X, y)  # theta fixed from here
    dc, ds = m._grad_premise(X, y)
    h, worst = 1e-6, 0.0
    for name, g in (("c", dc), ("s", ds)):
        P = getattr(m, name)
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + h; lp = m.loss(X, y)
            P[idx] = old - h; lm = m.loss(X, y)
            P[idx] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - g[idx]) / (abs(num) + abs(g[idx]) + 1e-12))
    assert worst < 1e-4, f"gradient mismatch, worst relative error {worst:.2e}"
    print(f"OK  analytic gradient == numerical gradient (worst rel. error {worst:.1e})")


def test_lse_recovers_linear_function():
    rng = np.random.default_rng(2)
    X = rng.random((200, 2))
    y = 2 * X[:, 0] - X[:, 1] + 0.5
    m = ANFIS(np.array([[0.5, 0.5]]), np.full((1, 2), 5.0), ridge=1e-9)  # 1 wide rule
    m._lse(X, y)
    assert np.allclose(m.predict_score(X), y, atol=1e-4)
    print("OK  LSE step recovers an exact linear target with one rule")


def test_hybrid_learning_reduces_loss():
    rng = np.random.default_rng(3)
    X = rng.random((400, 2))
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(float)  # XOR: not linearly separable
    C = np.array([[0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25]])
    m = ANFIS(C, np.full((4, 2), 0.3)).fit(X, y, epochs=60, lr=0.01)
    first, last = m.history[0]["train_loss"], m.history[-1]["train_loss"]
    acc = ((m.predict_score(X) >= 0.5) == y).mean()
    assert last < first and acc > 0.95, (first, last, acc)
    print(f"OK  hybrid learning: loss {first:.4f} -> {last:.4f}, XOR accuracy {acc:.2f}")


if __name__ == "__main__":
    test_clustering_finds_blobs()
    test_gradient_matches_finite_difference()
    test_lse_recovers_linear_function()
    test_hybrid_learning_reduces_loss()
    print("ALL TESTS PASSED")
