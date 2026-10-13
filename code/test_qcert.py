"""Unit tests for qcert (run: python test_qcert.py)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import qcert as Q

RNG = np.random.default_rng(1)
FAILS = []


def check(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not cond:
        FAILS.append(name)


# 1. orthogonal design: w*(e_i) = min(b, z / sqrt(lambda_i))   (break-even proposition)
b = 0.1
lam = np.array([10.0, 200.0, 384.0, 400.0, 5000.0])
S_w = np.diag(np.sqrt(lam))
solver = Q.CertificateSolver(5, 5)
ok = True
for i in range(5):
    c = solver.solve(S_w, np.eye(5)[i], b)
    ok &= abs(c.width - min(b, Q.Z95 / np.sqrt(lam[i]))) < 1e-6
check('orthogonal design closed form', ok)
g = Q.geometry(S_w, b)
check('break-even floor retains lam > (z/b)^2', list(g.keep) == [False, False, False, True, True],
      f'floor={g.floor:.2f}')

# 2. projection identity S_w^T a_P = P q and ||a_P|| = proj_sd
S = RNG.standard_normal((7, 5)) * np.array([5, 50, 100, 200, 400])
g = Q.geometry(S, b)
q = RNG.standard_normal(5)
aP = g.proj_weights(S, q)
check('projection estimator identity', np.allclose(S.T @ aP, g.P @ q, atol=1e-8)
      and abs(np.linalg.norm(aP) - g.proj_sd(q)) < 1e-8)

# 3. sandwich: rank floor <= w* <= projection-estimator width   (Fisher-eigenspace-sandwich proposition)
ok = True
for _ in range(20):
    n, p = RNG.integers(3, 9), RNG.integers(3, 9)
    S = RNG.standard_normal((n, p)) * 40
    E = np.abs(RNG.standard_normal((n, p))) * 0.5
    q = RNG.standard_normal(p)
    g = Q.geometry(S, b)
    c = Q.certificate(S, q, b, E)
    lo = Q.rank_floor(S, q, b)
    hi = Q.fixed_estimator_width(S, g.proj_weights(S, q), q, b, E)['width']
    ok &= (lo <= c.width + 1e-7) and (c.width <= min(hi, b * np.abs(q).sum()) + 1e-7)
check('sandwich bounds', ok)

# 4. monotonicity in layout, envelope and noise   (monotonicity proposition)
S = RNG.standard_normal((10, 6)) * 30
E = np.abs(RNG.standard_normal((10, 6))) * 0.3
q = RNG.standard_normal(6)
w_sub = Q.certificate(S[:6], q, b, E[:6]).width
w_all = Q.certificate(S, q, b, E).width
w_e2 = Q.certificate(S, q, b, 2 * E).width
w_n2 = Q.certificate(S / 2, q, b, E / 2).width
check('adding sensors never widens', w_all <= w_sub + 1e-8, f'{w_sub:.5f} -> {w_all:.5f}')
check('larger envelope never tightens', w_e2 >= w_all - 1e-8)
check('larger noise never tightens', w_n2 >= w_all - 1e-8)

# 5. rank floor is the zero-noise, zero-drift limit
S = RNG.standard_normal((4, 6))
q = RNG.standard_normal(6)
lim = Q.certificate(S * 1e5, q, b).width
check('rank floor = noise-free limit', abs(lim - Q.rank_floor(S, q, b)) < 1e-4,
      f'{lim:.6f} vs {Q.rank_floor(S, q, b):.6f}')
check('rank floor >= b ||P_null q||_2',
      Q.rank_floor(S, q, b) >= b * np.linalg.norm(Q.geometry(S, b).Pnull @ q) - 1e-9)

# 6. coverage of a fixed certified estimator under adversarial theta and Delta (validity proposition)
n, p = 8, 4
S = RNG.standard_normal((n, p)) * 30
E = np.abs(RNG.standard_normal((n, p))) * 2.0
q = np.array([1.0, 0.5, 0, 0])
c = Q.certificate(S, q, b, E)
worst = 1.0
for _ in range(40):
    th = b * RNG.choice([-1.0, 1.0], p)
    D = E * RNG.choice([-1.0, 1.0], (n, p))
    D = D * np.sign(c.a)[:, None] * np.sign(th)[None, :]   # push bias one way
    y = (S + D) @ th + RNG.standard_normal((20000, n))
    cov = np.mean(np.abs(y @ c.a - q @ th) <= c.width)
    worst = min(worst, cov)
check('certified coverage >= 0.95 under adversarial bias', worst >= 0.945, f'worst={worst:.4f}')

# 7. zero estimator fallback reaches the prior exactly
S = np.ones((2, 5)) * 0.01
c = Q.certificate(S, np.eye(5)[0], b)
check('prior fallback', abs(c.width - b) < 1e-12 and not c.informative)

print('\nALL PASS' if not FAILS else f'\n{len(FAILS)} FAILURES: {FAILS}')
sys.exit(1 if FAILS else 0)
