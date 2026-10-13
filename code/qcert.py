"""Query-level certification for linear(ised) vibration-based monitoring.

Observation model (unwhitened)
    y = (S + Delta) theta + eps,   eps ~ N(0, diag(sd^2)),
    theta in Theta_b = {|theta_i| <= b},   |Delta_ji| <= E_ji.

S (n x p) maps the estimand theta to the observables of one sensor layout
(rows = observables available at that layout), sd are the observation noise
standard deviations, b is the declared prior (feasible-set) half-width and
E is the declared model-discrepancy (drift) envelope. A query is a direction
q in theta-space; the scalar of interest is q^T theta.

Everything below works in whitened units: S_w = S / sd, E_w = E / sd.

Three levels of evidence are computed for every query:

1. rank level      - exact row-space membership of q (noise-free, model-exact);
2. noise level     - null participation and reporting state from the Fisher
                     eigenspace retained at lambda >= lambda* = (z / b)^2.
                     The break-even value is exact for an orthogonal design
                     and a coordinate query; for general layouts it is an
                     operational screening threshold;
3. certified level - the bias-aware minimax half-width of the best linear
                     estimator, valid for every theta in the box and every
                     admissible discrepancy (noise + aliasing + drift budget).

The reporting state (REPORT / BOUND / REFUSE) is assigned at the noise level;
the certificate states how precisely that report can be guaranteed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

Z95 = 1.959963984540054          # two-sided 95 % standard-normal quantile
REPORT, BOUND, REFUSE = 'REPORT', 'BOUND', 'REFUSE'
TAU_REPORT, TAU_REFUSE = 0.2, 0.6


# --------------------------------------------------------------------------- #
# geometry                                                                     #
# --------------------------------------------------------------------------- #
def breakeven_floor(box: float, z: float = Z95) -> float:
    """Eigenvalue floor at which the measurement and the prior bound tie.

    For an orthogonal design and a coordinate query, the noise-only 95 %
    half-width is z / sqrt(lam) and the prior half-width is b. Measurement
    improves on the prior when lam > (z / b)^2; equality is a tie and is
    retained by convention. For general layouts this is a screening value,
    not a query-specific optimum (break-even proposition).
    """
    return (z / box) ** 2


@dataclass
class Geometry:
    """Fisher eigen-split of one layout."""
    F: np.ndarray
    lam: np.ndarray          # ascending eigenvalues
    U: np.ndarray            # eigenvectors (columns)
    floor: float
    keep: np.ndarray         # bool mask, lam >= floor
    null: np.ndarray         # bool mask, numerically zero eigenvalues

    @property
    def P(self):
        u = self.U[:, self.keep]
        return u @ u.T

    @property
    def P0(self):
        return np.eye(self.F.shape[0]) - self.P

    @property
    def Pnull(self):
        u = self.U[:, self.null]
        return u @ u.T

    @property
    def rank(self) -> int:
        return int((~self.null).sum())

    @property
    def retained_rank(self) -> int:
        return int(self.keep.sum())

    def pnull(self, q) -> float:
        """Noise-level null participation: share of q outside the retained space."""
        q = np.asarray(q, float)
        return float(np.clip(q @ self.P0 @ q / (q @ q), 0.0, 1.0))

    def pnull_exact(self, q) -> float:
        """Rank-level null participation: share of q in the exact null space."""
        q = np.asarray(q, float)
        return float(np.clip(q @ self.Pnull @ q / (q @ q), 0.0, 1.0))

    def proj_sd(self, q) -> float:
        """Noise-only standard deviation of the retained-space projection of q."""
        q = np.asarray(q, float)
        u = self.U[:, self.keep]
        c = u.T @ q
        return float(np.sqrt(np.sum(c ** 2 / self.lam[self.keep])))

    def proj_weights(self, S_w: np.ndarray, q) -> np.ndarray:
        """Whitened estimator weights a_P of the retained-space projection.

        a_P^T y_w estimates (P q)^T theta without bias when the model is exact
        and theta has no component in the weak space.
        """
        q = np.asarray(q, float)
        u = self.U[:, self.keep]
        c = (u.T @ q) / self.lam[self.keep]
        return S_w @ (u @ c)


def geometry(S_w: np.ndarray, box: float, z: float = Z95,
             floor: float | None = None, rtol: float = 1e-10) -> Geometry:
    S_w = np.atleast_2d(np.asarray(S_w, float))
    F = S_w.T @ S_w
    F = 0.5 * (F + F.T)
    lam, U = np.linalg.eigh(F)
    lam = np.clip(lam, 0.0, None)
    fl = breakeven_floor(box, z) if floor is None else float(floor)
    keep = lam >= fl
    null = lam <= rtol * max(lam.max(), 1e-300)
    return Geometry(F=F, lam=lam, U=U, floor=fl, keep=keep, null=null)


def state_from_pnull(p: float, tau1: float = TAU_REPORT, tau2: float = TAU_REFUSE) -> str:
    if p <= tau1:
        return REPORT
    if p <= tau2:
        return BOUND
    return REFUSE


# --------------------------------------------------------------------------- #
# certificate                                                                  #
# --------------------------------------------------------------------------- #
def _cvxpy():
    import cvxpy as cp
    return cp


def row_envelope(E_w: np.ndarray | None, n: int) -> np.ndarray:
    """Row-aggregated whitened envelope e_j = sum_i E_w[j, i] (worst case over the box)."""
    if E_w is None:
        return np.zeros(n)
    E_w = np.asarray(E_w, float)
    return E_w.sum(1) if E_w.ndim == 2 else E_w


def additive_envelope(rho_w, box: float) -> np.ndarray:
    """Row envelope for an additive discrepancy |r_j| <= rho_j (whitened units).

    The certificate's discrepancy term is b e^T |a|; an additive bound rho on
    the model residual enters as e = rho / b, so that the term equals rho^T |a|.
    A dictionary perturbation |Delta| <= E is the special case rho = b E 1.
    """
    return np.asarray(rho_w, float) / float(box)


@dataclass
class Certificate:
    width: float             # certified half-width w*
    w0: float                # prior half-width b ||q||_1 (zero estimator)
    noise: float             # z ||a*||_2
    alias: float             # b ||S_w^T a* - q||_1
    drift: float             # b e^T |a*|
    a: np.ndarray = field(repr=False)

    @property
    def rel(self) -> float:
        return self.width / self.w0 if self.w0 > 0 else np.nan

    @property
    def informative(self) -> bool:
        return self.width < self.w0 * (1 - 1e-6)

    def limitation(self) -> str:
        """Dominant term of the error budget ('prior' when the box is optimal)."""
        if not self.informative:
            return 'prior'
        terms = {'noise': self.noise, 'alias': self.alias, 'drift': self.drift}
        return max(terms, key=terms.get)


class CertificateSolver:
    """Re-usable second-order cone program for the bias-aware certificate.

        w*(q) = min_a  z ||a||_2 + b ||S_w^T a - q||_1 + b e^T |a|

    S_w, q and e are cvxpy Parameters, so a solver instance can be re-used for
    every query, envelope scaling and noise scaling of one layout shape.
    """

    def __init__(self, n: int, p: int, z: float = Z95, solver: str = 'CLARABEL'):
        cp = _cvxpy()
        self.n, self.p, self.z, self.solver = n, p, z, solver
        self.a = cp.Variable(n)
        # the prior half-width b is folded into the parameters to keep the
        # program disciplined parametrised (DPP): b S_w, b q and b e
        self.Sb = cp.Parameter((n, p))
        self.qb = cp.Parameter(p)
        self.eb = cp.Parameter(n, nonneg=True)
        obj = (z * cp.norm(self.a, 2) + cp.norm(self.Sb.T @ self.a - self.qb, 1)
               + self.eb @ cp.abs(self.a))
        self.prob = cp.Problem(cp.Minimize(obj))

    def solve(self, S_w, q, box, e_w=None) -> Certificate:
        S_w = np.asarray(S_w, float)
        q = np.asarray(q, float)
        e = row_envelope(e_w, S_w.shape[0])
        box = float(box)
        self.Sb.value, self.qb.value, self.eb.value = box * S_w, box * q, box * e
        w0 = float(box * np.abs(q).sum())
        import warnings
        with warnings.catch_warnings():
            # any feasible a yields a valid certificate (validity proposition); solver
            # inaccuracy can only make the reported width conservative
            warnings.simplefilter('ignore')
            try:
                if self.solver == 'CLARABEL':
                    self.prob.solve(solver='CLARABEL', tol_gap_abs=1e-9, tol_gap_rel=1e-9,
                                    tol_feas=1e-9)
                else:
                    self.prob.solve(solver=self.solver)
                ok = self.prob.status in ('optimal', 'optimal_inaccurate')
            except Exception:                               # pragma: no cover
                ok = False
            if not ok or self.a.value is None:
                self.prob.solve(solver='SCS', eps_abs=1e-9, eps_rel=1e-9)
        a = np.asarray(self.a.value, float)
        noise = float(self.z * np.linalg.norm(a))
        alias = float(box * np.abs(S_w.T @ a - q).sum())
        drift = float(box * e @ np.abs(a))
        width = noise + alias + drift
        if width >= w0 - 1e-9 * max(w0, 1.0):
            # the zero estimator (report the prior box) is optimal
            return Certificate(width=w0, w0=w0, noise=0.0, alias=w0, drift=0.0,
                               a=np.zeros_like(a))
        return Certificate(width=width, w0=w0, noise=noise, alias=alias, drift=drift, a=a)


def certificate(S_w, q, box, e_w=None, z: float = Z95) -> Certificate:
    S_w = np.atleast_2d(np.asarray(S_w, float))
    return CertificateSolver(S_w.shape[0], S_w.shape[1], z).solve(S_w, q, box, e_w)


def fixed_estimator_width(S_w, a, q, box, e_w=None, z: float = Z95) -> dict:
    """Bias-aware half-width of a given whitened estimator a (validity proposition)."""
    S_w = np.asarray(S_w, float)
    a = np.asarray(a, float)
    q = np.asarray(q, float)
    e = row_envelope(e_w, S_w.shape[0])
    noise = float(z * np.linalg.norm(a))
    alias = float(box * np.abs(S_w.T @ a - q).sum())
    drift = float(box * e @ np.abs(a))
    return {'width': noise + alias + drift, 'noise': noise, 'alias': alias, 'drift': drift}


def rank_floor(S_w, q, box) -> float:
    """Noise- and drift-free limit of the certificate, b * dist_1(q, row(S)).

    It is the smallest half-width any linear estimator can certify with the
    layout, however much the noise is averaged down and however small the
    discrepancy becomes (rank-floor proposition).
    """
    cp = _cvxpy()
    S_w = np.atleast_2d(np.asarray(S_w, float))
    q = np.asarray(q, float)
    # parametrise the row space by an orthonormal basis to keep the LP well scaled
    u, s, vt = np.linalg.svd(S_w, full_matrices=False)
    r = int((s > 1e-10 * max(s.max(), 1e-300)).sum())
    if r == 0:
        return float(box * np.abs(q).sum())
    V = vt[:r].T                                           # p x r basis of row space
    c = cp.Variable(r)
    prob = cp.Problem(cp.Minimize(cp.norm(V @ c - q, 1)))
    prob.solve(solver='CLARABEL')
    return float(box * prob.value)


# --------------------------------------------------------------------------- #
# convenience: full evaluation of one layout                                   #
# --------------------------------------------------------------------------- #
def evaluate_layout(S, sd, box, queries, E=None, z: float = Z95, solver=None,
                    with_rank_floor: bool = True) -> list[dict]:
    """Rank-, noise- and certified-level evaluation of every query of one layout."""
    S = np.atleast_2d(np.asarray(S, float))
    sd = np.broadcast_to(np.asarray(sd, float), (S.shape[0],))
    S_w = S / sd[:, None]
    E_w = None if E is None else np.asarray(E, float) / sd[:, None]
    g = geometry(S_w, box, z)
    if solver is None or solver.n != S.shape[0] or solver.p != S.shape[1]:
        solver = CertificateSolver(S.shape[0], S.shape[1], z)
    out = []
    for qi, q in enumerate(np.atleast_2d(queries)):
        c = solver.solve(S_w, q, box, E_w)
        c0 = solver.solve(S_w, q, box, None) if E is not None else c
        rec = {
            'query': qi, 'rank': g.rank, 'retained_rank': g.retained_rank,
            'pnull_exact': g.pnull_exact(q), 'pnull': g.pnull(q),
            'state_rank': state_from_pnull(g.pnull_exact(q)),
            'state': state_from_pnull(g.pnull(q)),
            'proj_sd': g.proj_sd(q), 'naive_halfwidth': z * g.proj_sd(q),
            'w0': c.w0, 'width': c.width, 'rel_width': c.rel,
            'noise': c.noise, 'alias': c.alias, 'drift': c.drift,
            'limitation': c.limitation(),
            'width_nodrift': c0.width, 'rel_width_nodrift': c0.rel,
        }
        if with_rank_floor:
            rf = rank_floor(S_w, q, box)
            rec['rank_floor'] = rf
            rec['rel_rank_floor'] = rf / c.w0
        out.append(rec)
    return out
