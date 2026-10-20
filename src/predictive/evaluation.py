import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def metrics(y, p, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    keep = w > 0
    y, p, w = y[keep], p[keep], w[keep]
    if len(np.unique(y)) != 2:
        return {k: float('nan') for k in ['roc_auc', 'average_precision', 'brier', 'calibration_intercept', 'calibration_slope']}
    result = dict(roc_auc=roc_auc_score(y, p, sample_weight=w),
                  average_precision=average_precision_score(y, p, sample_weight=w),
                  brier=brier_score_loss(y, p, sample_weight=w))
    z = logit(np.clip(p, 1e-8, 1-1e-8))
    # Near-constant fold prevalence predictions do not identify a stable slope.
    if np.average((z-np.average(z,weights=w))**2,weights=w) < 1e-6:
        result.update(calibration_intercept=float('nan'), calibration_slope=float('nan'))
        return result
    design = np.column_stack([np.ones(len(z)), z])
    def objective(beta):
        eta = design @ beta
        return np.sum(w*(np.logaddexp(0, eta)-y*eta))/w.sum()
    def gradient(beta):
        return design.T @ (w*(expit(design @ beta)-y))/w.sum()
    fit = minimize(objective, [0., 1.], jac=gradient, method='BFGS', options={'gtol': 1e-8})
    valid = fit.success or np.linalg.norm(gradient(fit.x)) < 1e-6
    result.update(calibration_intercept=float(fit.x[0]) if valid else float('nan'),
                  calibration_slope=float(fit.x[1]) if valid else float('nan'))
    return result

def bootstrap(y, p, repetitions=1000, seed=20261012):
    """Ordinary row bootstrap, multinomial over identical (label, score) pairs."""
    pairs, counts = np.unique(np.column_stack([y, p]), axis=0, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        weights = rng.multinomial(len(y), counts / counts.sum())
        draws.append(metrics(pairs[:, 0], pairs[:, 1], weights))
    out = {}
    for key in draws[0]:
        values = np.array([d[key] for d in draws])
        values = values[np.isfinite(values)]
        out[key] = {'low': float(np.quantile(values, .025)) if len(values) else None,
                    'high': float(np.quantile(values, .975)) if len(values) else None,
                    'valid_replicates': len(values)}
    return out
