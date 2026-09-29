"""Same-objective softmax head fitting in invertible coordinates.

Training-only factorization is folded into the original head/bias before export.
No transform or new runtime feature is required for deployment.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class Coordinates:
    mean: np.ndarray
    factor: np.ndarray | None
    scale: np.ndarray | None

    @classmethod
    def prepare(cls, features: np.ndarray, ridge: float, method: str):
        p = features.shape[1]
        if method == 'raw':
            return cls(np.zeros(p), None, None), features
        mean = features.mean(axis=0)
        centered = features - mean
        if method == 'diagonal':
            scale = np.sqrt(np.mean(centered**2, axis=0) + ridge)
            return cls(mean, None, scale), centered / scale
        if method != 'cholesky':
            raise ValueError('method must be raw, diagonal, or cholesky')
        covariance = centered.T @ centered / len(features)
        covariance.flat[::p+1] += ridge
        factor = cholesky(covariance, lower=True, check_finite=False)
        transformed = solve_triangular(factor, centered.T, lower=True,
                                       check_finite=False).T.copy()
        return cls(mean, factor, None), transformed

    def encode(self, head, bias):
        if self.factor is not None:
            weights = self.factor.T @ head
        elif self.scale is not None:
            weights = self.scale[:, None] * head
        else:
            weights = head.copy()
        return weights, bias + self.mean @ head

    def decode(self, weights, offset):
        if self.factor is not None:
            head = solve_triangular(self.factor.T, weights, lower=False,
                                    check_finite=False)
        elif self.scale is not None:
            head = weights / self.scale[:, None]
        else:
            head = weights
        return head, offset - self.mean @ head

    def penalty_gradient(self, head):
        # d ||L^{-T} B||^2 / d B = 2 L^{-1} L^{-T} B.
        if self.factor is not None:
            return solve_triangular(self.factor, head, lower=True,
                                    check_finite=False)
        if self.scale is not None:
            return head / self.scale[:, None]
        return head


def loss_gradient(features, targets, head, bias, ridge):
    logits = features @ head + bias
    normalizer = logsumexp(logits, axis=1)
    value = float(np.mean(normalizer - logits[np.arange(len(targets)), targets])
                  + .5 * ridge * np.sum(head**2))
    residual = np.exp(logits - normalizer[:, None])
    residual[np.arange(len(targets)), targets] -= 1.
    residual /= len(targets)
    return value, features.T @ residual + ridge * head, residual.sum(axis=0)


def fit_head(features, targets, head, bias, *, ridge=1e-6, method='cholesky',
             maxiter=150, ftol=1e-10, gtol=1e-6):
    """Fit one frozen-feature head; label classes must be encoded in [0,C).

    Preparation cost is included in the returned CPU/wall totals. No validation
    data, early-stopping labels, pseudo-labels or test-based hyperparameters enter.
    Status is a solver status, not a certificate of optimality/generalization.
    """
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    f = np.asarray(features, dtype=np.float64, order='C')
    h = np.asarray(head, dtype=np.float64, order='C')
    b = np.asarray(bias, dtype=np.float64)
    y = np.asarray(targets)
    if f.ndim != 2 or not 0 < f.size <= 64_000_000 or h.ndim != 2:
        raise ValueError('nonempty bounded feature matrix and 2D head required')
    n, p = f.shape
    if h.shape[0] != p or not 2 <= h.shape[1] <= 128 or b.shape != (h.shape[1],):
        raise ValueError('inconsistent head/bias shape')
    if y.shape != (n,) or y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= h.shape[1]):
        raise ValueError('encoded integer labels required')
    if any(not np.isfinite(x).all() for x in (f, h, b)):
        raise ValueError('nonfinite feature or parameter')
    if type(ridge) not in (float, int) or not np.isfinite(ridge) or not ridge > 0:
        raise ValueError('positive finite ridge required')
    if type(maxiter) is not int or not 1 <= maxiter <= 5000:
        raise ValueError('maxiter outside supported range')
    for tolerance in (ftol, gtol):
        if type(tolerance) not in (float, int) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('nonnegative finite tolerances required')
    initial_loss, _, _ = loss_gradient(f, y, h, b, ridge)
    prep_cpu = time.process_time()
    coord, z = Coordinates.prepare(f, ridge, method)
    w, offset = coord.encode(h, b)
    preparation_seconds = time.process_time() - prep_cpu
    c = h.shape[1]
    calls = 0
    trace = []
    latest = {}

    def objective(theta):
        nonlocal calls
        weights = theta[:p*c].reshape(p,c)
        beta = theta[p*c:]
        original_head, _ = coord.decode(weights, beta)
        logits = z @ weights + beta
        norm = logsumexp(logits, axis=1)
        value = float(np.mean(norm - logits[np.arange(n), y])
                      + .5 * ridge * np.sum(original_head**2))
        residual = np.exp(logits - norm[:,None])
        residual[np.arange(n),y] -= 1.
        residual /= n
        gradient = z.T @ residual + ridge * coord.penalty_gradient(original_head)
        gb = residual.sum(axis=0)
        calls += 1
        latest.update(objective=value, calls=calls,
                      transformed_gradient_inf=float(max(np.max(np.abs(gradient)), np.max(np.abs(gb)))))
        return value, np.r_[gradient.ravel(),gb]

    def callback(theta):
        trace.append({'iteration':len(trace)+1, **latest,
                      'cpu_seconds':time.process_time()-start_cpu})

    solution = minimize(objective, np.r_[w.ravel(),offset], method='L-BFGS-B', jac=True,
                        callback=callback, options={'maxiter':maxiter, 'ftol':ftol,
                                                   'gtol':gtol, 'maxls':30})
    result_h, result_b = coord.decode(solution.x[:p*c].reshape(p,c),solution.x[p*c:])
    # Serialize original coordinates only, not z/L/mu. Validate true objective after pullback.
    result_h, result_b = result_h.copy(), result_b.copy()
    objective_value, gh, gb = loss_gradient(f, y, result_h, result_b, ridge)
    discrepancy = abs(objective_value - float(solution.fun))
    if not np.isfinite(objective_value) or discrepancy > 1e-8 * max(1.,abs(objective_value)):
        raise ArithmeticError('pullback objective failed numerical consistency check')
    report = {'method':method, 'maxiter':maxiter, 'ridge':ridge,
              'success':bool(solution.success),'status':int(solution.status),
              'message':str(solution.message), 'iterations':int(solution.nit),
              'function_evaluations':calls, 'initial_objective':initial_loss,
              'objective':objective_value, 'pullback_objective_difference':discrepancy,
              'original_gradient_inf':float(max(np.max(np.abs(gh)),np.max(np.abs(gb)))),
              'preparation_cpu_seconds':preparation_seconds,
              'cpu_seconds':time.process_time()-start_cpu,
              'wall_seconds':time.perf_counter()-start_wall, 'trace':trace,
              'training_rows':n,'features':p,'classes':c,
              'scope':'same ridge-softmax objective; training transform folded into existing head'}
    return result_h, result_b, report
