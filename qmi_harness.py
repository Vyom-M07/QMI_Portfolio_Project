"""Shared code for the QMI portfolio project (exported verbatim from the notebook; do not edit by hand)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252

def port_vol(w, S, ann=TRADING_DAYS):
    """Annualized volatility of weights w, given a DAILY covariance matrix S."""
    return float(np.sqrt(w @ S @ w * ann))

def min_var_weights(S):
    """Global minimum-variance weights, closed form. Weights sum to 1; negative weights (shorts) are allowed."""
    x = np.linalg.solve(S, np.ones(len(S)))      # solve S x = 1
    return x / x.sum()

def simulate_known_truth(S_true, Ts, reps=300, seed=7):
    """Average promised vs true volatility of estimated min-variance weights, for several sample lengths T."""
    rng = np.random.default_rng(seed)
    n = len(S_true)
    best = port_vol(min_var_weights(S_true), S_true)
    rows = []
    for T in Ts:
        promised, actual = [], []
        for _ in range(reps):
            X = rng.multivariate_normal(np.zeros(n), S_true, size=T)
            S_hat = np.cov(X.T)
            w = min_var_weights(S_hat)
            promised.append(port_vol(w, S_hat))     # what the optimizer believes
            actual.append(port_vol(w, S_true))      # what the portfolio really delivers
        rows.append({"T (days)": T, "T / N": T / n, "promised vol": np.mean(promised),
                     "true vol": np.mean(actual), "best possible vol": best})
    out = pd.DataFrame(rows).set_index("T (days)")
    out["true / promised"] = out["true vol"] / out["promised vol"]
    out["excess risk vs best"] = out["true vol"] / out["best possible vol"] - 1
    return out

def evaluate(get_weights, rets, start, end, hold=21, cost_bps=10, max_weight=0.25, max_gross=1.5,
             promise_window=252, min_history=60):
    """Walk-forward test of a weighting rule.

    get_weights(history) -> array of weights. `history` is a DataFrame of DAILY SIMPLE RETURNS,
    every row strictly BEFORE the rebalance date (so the rule cannot peek at the future).
    Returns a dict of metrics plus the daily net return series and the weights chosen at each rebalance.
    """
    idx, n = rets.index, rets.shape[1]
    t_start = idx.searchsorted(pd.Timestamp(start))
    t_stop = idx.searchsorted(pd.Timestamp(end), side="right")
    assert t_start >= min_history, "not enough history before the first rebalance"

    daily, W, dates, promised, turnover, violations = [], [], [], [], [], 0
    w_drift = None                                     # weights after drifting with prices (what we really hold)
    for t in range(t_start, t_stop, hold):
        history = rets.iloc[:t].copy()                 # a COPY of everything strictly before day t
        assert history.index[-1] < idx[t]              # the no-peeking guarantee
        w = np.asarray(get_weights(history), dtype=float).ravel()
        if w.shape != (n,) or not np.all(np.isfinite(w)) or abs(w.sum() - 1) > 1e-6:
            raise ValueError("get_weights must return one finite weight per asset, summing to 1")
        violations += int(np.abs(w).max() > max_weight + 1e-9 or np.abs(w).sum() > max_gross + 1e-9)

        S_ref = np.cov(history.iloc[-promise_window:].values.T)       # the "backtest promise": recent in-sample risk
        promised.append(port_vol(w, S_ref))

        traded = np.abs(w).sum() if w_drift is None else np.abs(w - w_drift).sum()    # dollars traded per $1 of wealth
        turnover.append(traded)

        block = rets.iloc[t:min(t + hold, t_stop)].values
        growth = np.cumprod(1 + block, axis=0)         # growth of $1 in each stock over the holding period
        value = growth @ w                             # value of the portfolio each day (weights drift)
        r = value / np.concatenate([[1.0], value[:-1]]) - 1
        r[0] -= cost_bps / 1e4 * traded                # pay trading costs on the first day of the block
        daily.append(pd.Series(r, index=idx[t:t + len(block)]))
        w_drift = w * growth[-1] / value[-1]
        W.append(w); dates.append(idx[t])

    r = pd.concat(daily)
    equity = (1 + r).cumprod()
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {
        "vol": vol,                                                       # realized annualized volatility, net of costs
        "ann_ret": equity.iloc[-1] ** (TRADING_DAYS / len(r)) - 1,        # annualized return, net of costs
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),             # return per unit of risk (risk-free rate = 0)
        "max_dd": (equity / equity.cummax() - 1).min(),                   # worst peak-to-trough loss
        "promised_vol": float(np.mean(promised)),                         # average in-sample risk of the chosen weights
        "vol_ratio": vol / float(np.mean(promised)) if np.mean(promised) > 0 else float("nan"),   # delivered / promised (1.0 = honest)
        "turnover": float(np.mean(turnover[1:])) if len(turnover) > 1 else float("nan"),   # per rebalance, excl. first purchase
        "avg_max_weight": float(np.mean([np.abs(w).max() for w in W])),
        "avg_gross": float(np.mean([np.abs(w).sum() for w in W])),
        "violation_share": violations / len(W),
        "returns": r,
        "weights": pd.DataFrame(W, index=dates, columns=rets.columns),
    }

def summarize(results):
    """One comparison table from {name: evaluate(...) result}."""
    cols = ["vol", "promised_vol", "vol_ratio", "ann_ret", "sharpe", "max_dd", "turnover",
            "avg_max_weight", "avg_gross", "violation_share"]
    return pd.DataFrame({k: {c: r[c] for c in cols} for k, r in results.items()}).T

def min_var_capped(S, cap=0.20):
    """Long-only minimum-variance weights with every weight <= cap. No closed form, so we solve it numerically."""
    n = len(S)
    assert cap >= 1 / n, "cap too small: weights cannot sum to 1"
    Ss = S / (np.trace(S) / n)                   # rescale so the objective is order 1 (helps the solver)
    w0 = np.ones(n) / n
    res = minimize(lambda w: w @ Ss @ w, w0, jac=lambda w: 2 * Ss @ w, bounds=[(0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-14})
    if not res.success:
        OPT_FAILURES.append(res.message)         # record it instead of hiding it
        return w0
    w = np.clip(res.x, 0, cap)
    return w / w.sum()

OPT_FAILURES = []

def make_strategy(kind, window=252, cap=0.20):
    """Build a get_weights(history) rule.
    kind: 'ew' equal weight | 'raw' closed-form min-var | 'shrunk' min-var on a Ledoit-Wolf covariance |
          'capped' long-only capped min-var | 'shrunk_capped' both fixes together."""
    def get_weights(history):
        n = history.shape[1]
        if kind == "ew":
            return np.ones(n) / n
        X = history.iloc[-window:].values                 # only the most recent `window` days
        S = LedoitWolf().fit(X).covariance_ if kind in ("shrunk", "shrunk_capped") else np.cov(X.T)
        return min_var_weights(S) if kind in ("raw", "shrunk") else min_var_capped(S, cap)
    return get_weights

LECTURE_SOLUTION = make_strategy("shrunk_capped", window=252, cap=0.20)
