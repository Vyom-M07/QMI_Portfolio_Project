"""QMI portfolio project: our get_weights(history).

Goal: lowest realized volatility after trading costs, on unseen data.
Rules: weights sum to 1, |w_i| <= 0.25, sum|w_i| <= 1.5, rebalanced every 21 days, 10 bp per $ traded.

Design (each choice is defended in the notebook):
  1. Dynamic universe: a stock is only eligible once it has ELIG (252) days of real data. Leading NaNs (not yet listed) and leading zero returns (NaNs that were filled with 0)
     are both treated as "not listed yet". Ineligible stocks get weight 0.
  2. Covariance: Ledoit-Wolf shrinkage on (up to) the last WINDOW days of eligible stocks
     (sample covariance is too noisy for 20 assets / 252 days -> optimizer overfits the noise).
  3. Optimizer: minimum variance, long-only, every weight <= CAP.
  4. TIER-2 CHANGE: exponentially weighted (EWMA) days before the Ledoit-Wolf step (half-life 126d,
     truncated at 504d), so recent days count more but older data still adds information.
  5. TIER-3 CHANGE: skip stale stocks (last 10 returns all 0/NaN), found by stress-testing a delisting.
Only numpy / pandas / scipy / scikit-learn are used. The function never reads files or the network,
and it only sees `history` (rows strictly before the rebalance date), so it cannot peek ahead.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

WINDOW = 504        # max days of history used for the covariance (~2 trading years)
HALFLIFE = 126      # EWMA half-life in days (TIER-2 change); None = every day weighted equally (lecture)
ELIG = 252          # a stock needs this many listed days before we will hold it
CAP = 0.20          # max weight per stock (rule allows 0.25; 0.20 keeps a buffer and forces >= 5 names)
MIN_OBS = 126       # fallback: minimum real observations if too few stocks pass ELIG
STALE = 10          # TIER-3 change: skip a stock whose last STALE returns are all 0/NaN (halted/delisted)
RULE_CAP = 0.25     # hard limit from the rules


def eligible_columns(history, elig=ELIG, min_obs=MIN_OBS, stale=STALE):
    """(mask, days_listed): which stocks have enough *real* data to estimate risk.

    A stock's listing date = first row where its return is neither NaN nor exactly 0.
    (Exact-zero leading returns are what you get if someone fills pre-IPO NaNs with 0;
    treating them as real would make the stock look riskless, and min-variance would pile into it.)
    The same trap exists at the END of a series (delisted / halted stock filled with 0s), so a stock
    whose last `stale` returns are all 0/NaN is also excluded.
    """
    vals = history.values
    live = np.isfinite(vals) & (vals != 0)
    has_any = live.any(axis=0)
    first_live = np.where(has_any, live.argmax(axis=0), len(history))
    days_listed = len(history) - first_live
    alive = live[-stale:].any(axis=0) if stale else np.ones(len(has_any), bool)
    full = has_any & alive & (days_listed >= elig)
    if full.sum() >= 1 / RULE_CAP:          # enough names to respect the 25% cap
        return full, days_listed
    return has_any & alive & (days_listed >= min_obs), days_listed


def ewma_lw_cov(X, halflife=HALFLIFE):
    """Ledoit-Wolf covariance of X (T x n), optionally with exponentially decaying day weights.

    Weighting trick: scale each (demeaned) row by sqrt(weight * T); the ordinary covariance of the
    scaled rows is then the weighted covariance, and Ledoit-Wolf shrinks it as usual.
    """
    X = X - X.mean(axis=0)
    if halflife:
        T = len(X)
        age = np.arange(T)[::-1]                         # 0 = most recent day
        wts = 0.5 ** (age / halflife)
        wts = wts / wts.sum()
        X = X * np.sqrt(wts * T)[:, None]
    return LedoitWolf(assume_centered=True).fit(X).covariance_


def min_var_long_only(S, cap):
    """Long-only minimum-variance weights with 0 <= w_i <= cap, sum w = 1 (SLSQP)."""
    n = len(S)
    cap = max(cap, 1.0 / n)                              # if cap is infeasible, loosest feasible value
    Ss = S / (np.trace(S) / n)                           # scale so the objective is ~1 (solver stability)
    w0 = np.ones(n) / n
    res = minimize(lambda w: w @ Ss @ w, w0, jac=lambda w: 2 * Ss @ w,
                   bounds=[(0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-14})
    w = np.clip(res.x, 0, cap) if res.success else w0
    return w / w.sum()


def make_get_weights(window=WINDOW, halflife=HALFLIFE, cap=CAP, elig=ELIG, stale=STALE):
    """Factory so the notebook can test variants with the exact same code path."""
    def get_weights(history):
        n = history.shape[1]
        ok, days_listed = eligible_columns(history, elig, min(MIN_OBS, elig), stale)
        if ok.sum() == 0:                                # nothing usable at all: equal weight
            return np.ones(n) / n
        k = int(min(window, days_listed[ok].min()))      # rows where EVERY eligible stock was listed
        X = np.nan_to_num(history.values[-k:, ok])       # a stray mid-series gap counts as a 0 return
        if ok.sum() == 1 or k < 2:
            w_ok = np.ones(ok.sum()) / ok.sum()
        else:
            w_ok = min_var_long_only(ewma_lw_cov(X, halflife), cap)
        w = np.zeros(n)
        w[ok] = w_ok
        return w
    return get_weights


get_weights = make_get_weights()


if __name__ == "__main__":
    # Self-test: runs standalone on both data files, checks the rules, prints the numbers.
    from qmi_harness import evaluate, summarize, make_strategy, LECTURE_SOLUTION
    clean = pd.read_csv("sp500_20_stocks_1990_2019.csv", index_col=0, parse_dates=True).pct_change().dropna()
    messy = pd.read_csv("messy_20_stocks_1989_2018.csv", index_col=0, parse_dates=True)
    messy = messy.pct_change(fill_method=None).iloc[1:].fillna(0.0)
    pd.set_option("display.width", 200)
    for name, rets, a, b in [("clean", clean, "2010-01-01", "2019-12-31"), ("messy", messy, "2010-01-01", "2018-04-11")]:
        res = {"Equal weight": evaluate(make_strategy("ew"), rets, a, b),
               "Ours": evaluate(get_weights, rets, a, b)}
        if name == "clean":
            res["Lecture"] = evaluate(LECTURE_SOLUTION, rets, a, b)
        print(f"\n{name} {a} -> {b}")
        print(summarize(res)[["vol", "promised_vol", "vol_ratio", "ann_ret", "turnover", "violation_share"]].round(4))
