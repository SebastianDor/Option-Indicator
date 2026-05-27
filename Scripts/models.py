# models.py
# ── Unified model interface for GJR-GARCH variants, EGARCH, GARCH-M and CAViaR
# Every public function returns a ModelResult dict with the same keys so the
# Shiny app can swap models without changing any display logic.

import numpy as np
import pandas as pd
from datetime import datetime
from arch import arch_model
from arch.univariate.distribution import SkewStudent
from scipy.optimize import minimize
from scipy.stats import norm


# ── Shared type (for documentation — plain dict at runtime) ──────────────────
# ModelResult = {
#   "status":        "ok" | "error"
#   "error":         str                    (only when status == "error")
#   "model_id":      str                    e.g. "gjr_skewt"
#   "model_name":    str                    e.g. "GJR-GARCH + Skew-t"
#   "description":   str                    one-liner for UI tooltip
#   "next_date":     pd.Timestamp
#   "mean_fc":       float                  decimal (not %)
#   "vol_fc":        float                  decimal
#   "ann_vol":       float                  decimal
#   "lower_68":      float                  decimal
#   "upper_68":      float                  decimal
#   "lower_95":      float                  decimal
#   "upper_95":      float                  decimal
#   "p_positive":    float                  0–1
#   "p_negative":    float                  0–1
#   "p_up_1pct":     float                  0–1
#   "p_down_1pct":   float                  0–1
#   "nu":            float | None           degrees of freedom
#   "lam":           float | None           skewness parameter
#   "alpha":         float | None           GARCH alpha
#   "gamma":         float | None           GJR gamma
#   "beta":          float | None           GARCH beta
#   "persist":       float | None           volatility persistence
#   "half_life":     float | None           days
#   "aic":           float | None
#   "computed_at":   datetime
#   # CAViaR-only extras (None for all other models)
#   "q05":           float | None           5th percentile forecast (decimal)
#   "q10":           float | None           10th percentile forecast (decimal)
#   "q25":           float | None           25th percentile forecast (decimal)
# }


# ── Shared skew-t helpers ─────────────────────────────────────────────────────
_skewt = SkewStudent()

def _skewt_ppf(p, mean_fc, vol_fc, nu, lam):
    params = np.array([nu, lam])
    return mean_fc + vol_fc * float(
        _skewt.ppf(np.atleast_1d(p), params).flat[0]
    )

def _skewt_cdf(x, mean_fc, vol_fc, nu, lam):
    params = np.array([nu, lam])
    z = (x - mean_fc) / vol_fc
    return float(_skewt.cdf(np.atleast_1d(z), params).flat[0])


def _t_ppf(p, mean_fc, vol_fc, nu):
    from scipy.stats import t as scipy_t
    return float(scipy_t.ppf(p, df=nu, loc=mean_fc, scale=vol_fc))

def _t_cdf(x, mean_fc, vol_fc, nu):
    from scipy.stats import t as scipy_t
    return float(scipy_t.cdf(x, df=nu, loc=mean_fc, scale=vol_fc))

def _norm_ppf(p, mean_fc, vol_fc):
    return float(norm.ppf(p, loc=mean_fc, scale=vol_fc))

def _norm_cdf(x, mean_fc, vol_fc):
    return float(norm.cdf(x, loc=mean_fc, scale=vol_fc))


# ── Shared GARCH result builder ───────────────────────────────────────────────
def _build_from_arch(
    r,
    log_ret,
    model_id,
    model_name,
    description,
    ppf_fn,
    cdf_fn,
):
    """
    Given a fitted arch ModelResult `r`, extract forecast + params,
    compute probabilities using the supplied ppf/cdf callables,
    and return a ModelResult dict.
    """
    fc      = r.forecast(horizon=1, reindex=False)
    mean_fc = float(fc.mean.iloc[-1, 0]) / 100
    vol_fc  = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100

    lower_95 = ppf_fn(0.025)
    upper_95 = ppf_fn(0.975)
    lower_68 = ppf_fn(0.16)
    upper_68 = ppf_fn(0.84)

    p_positive  = 1.0 - cdf_fn(0.0)
    p_negative  =       cdf_fn(0.0)
    p_down_1pct =       cdf_fn(-0.01)
    p_up_1pct   = 1.0 - cdf_fn(0.01)

    params  = r.params
    alpha   = float(params.get("alpha[1]", np.nan))
    gamma   = float(params.get("gamma[1]", np.nan))
    beta    = float(params.get("beta[1]",  np.nan))
    persist = alpha + beta + 0.5 * gamma if not np.isnan(alpha + beta + gamma) else np.nan

    # degrees of freedom / skewness — handle both naming conventions
    nu  = float(params["eta"])    if "eta"    in params else None
    lam = float(params["lambda"]) if "lambda" in params else None

    return {
        "status":      "ok",
        "model_id":    model_id,
        "model_name":  model_name,
        "description": description,
        "next_date":   pd.bdate_range(start=log_ret.index[-1], periods=2, freq="B")[-1],
        "mean_fc":     mean_fc,
        "vol_fc":      vol_fc,
        "ann_vol":     vol_fc * np.sqrt(252),
        "lower_95":    lower_95,
        "upper_95":    upper_95,
        "lower_68":    lower_68,
        "upper_68":    upper_68,
        "p_positive":  p_positive,
        "p_negative":  p_negative,
        "p_down_1pct": p_down_1pct,
        "p_up_1pct":   p_up_1pct,
        "nu":          nu,
        "lam":         lam,
        "alpha":       alpha,
        "gamma":       gamma,
        "beta":        beta,
        "persist":     persist,
        "half_life":   np.log(0.5) / np.log(persist) if (persist is not None and not np.isnan(persist) and 0 < persist < 1) else float("inf"),
        "aic":         r.aic,
        "computed_at": datetime.now(),
        "q05":         None,
        "q10":         None,
        "q25":         None,
    }


def _error_result(model_id, model_name, description, e):
    return {
        "status":      "error",
        "error":       str(e),
        "model_id":    model_id,
        "model_name":  model_name,
        "description": description,
        "computed_at": datetime.now(),
    }


# ── Model 1: GJR-GARCH + Skew-t (current default) ───────────────────────────
def fit_gjr_skewt(log_ret_pct: pd.Series, log_ret: pd.Series) -> dict:
    MODEL_ID   = "gjr_skewt"
    MODEL_NAME = "GJR-GARCH + Skew-t"
    DESC = (
        "Asymmetric volatility: negative shocks raise variance more than "
        "positive shocks (GJR γ term). Left-skewed fat-tailed distribution "
        "puts more weight on downside outcomes. Most realistic for equity indices."
    )
    try:
        m = arch_model(log_ret_pct, mean="Constant", vol="GARCH",
                       p=1, o=1, q=1, dist="skewt", rescale=False)
        r = m.fit(disp="off", options={"maxiter": 1000})

        fc      = r.forecast(horizon=1, reindex=False)
        mean_fc = float(fc.mean.iloc[-1, 0]) / 100
        vol_fc  = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100
        nu      = float(r.params["eta"])
        lam     = float(r.params["lambda"])

        ppf = lambda p: _skewt_ppf(p, mean_fc, vol_fc, nu, lam)
        cdf = lambda x: _skewt_cdf(x, mean_fc, vol_fc, nu, lam)

        return _build_from_arch(r, log_ret, MODEL_ID, MODEL_NAME, DESC, ppf, cdf)
    except Exception as e:
        return _error_result(MODEL_ID, MODEL_NAME, DESC, e)


# ── Model 2: GJR-GARCH + Student-t ───────────────────────────────────────────
def fit_gjr_t(log_ret_pct: pd.Series, log_ret: pd.Series) -> dict:
    MODEL_ID   = "gjr_t"
    MODEL_NAME = "GJR-GARCH + Student-t"
    DESC = (
        "Asymmetric volatility with symmetric fat-tailed distribution. "
        "Captures volatility clustering and heavy tails but treats upside "
        "and downside tail risk equally. Good baseline comparison to Skew-t."
    )
    try:
        m = arch_model(log_ret_pct, mean="Constant", vol="GARCH",
                       p=1, o=1, q=1, dist="t", rescale=False)
        r = m.fit(disp="off", options={"maxiter": 1000})

        fc      = r.forecast(horizon=1, reindex=False)
        mean_fc = float(fc.mean.iloc[-1, 0]) / 100
        vol_fc  = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100

        # arch names the df param "nu" for symmetric t
        nu_key = "nu" if "nu" in r.params else "eta"
        nu     = float(r.params[nu_key])

        ppf = lambda p: _t_ppf(p, mean_fc, vol_fc, nu)
        cdf = lambda x: _t_cdf(x, mean_fc, vol_fc, nu)

        return _build_from_arch(r, log_ret, MODEL_ID, MODEL_NAME, DESC, ppf, cdf)
    except Exception as e:
        return _error_result(MODEL_ID, MODEL_NAME, DESC, e)


# ── Model 3: GJR-GARCH + Normal ───────────────────────────────────────────────
def fit_gjr_normal(log_ret_pct: pd.Series, log_ret: pd.Series) -> dict:
    MODEL_ID   = "gjr_normal"
    MODEL_NAME = "GJR-GARCH + Normal"
    DESC = (
        "Asymmetric volatility with Gaussian distribution. Thin tails mean "
        "extreme events are heavily underestimated. Useful as a lower bound — "
        "shows how much fat tails matter for tail probabilities."
    )
    try:
        m = arch_model(log_ret_pct, mean="Constant", vol="GARCH",
                       p=1, o=1, q=1, dist="normal", rescale=False)
        r = m.fit(disp="off", options={"maxiter": 1000})

        fc      = r.forecast(horizon=1, reindex=False)
        mean_fc = float(fc.mean.iloc[-1, 0]) / 100
        vol_fc  = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100

        ppf = lambda p: _norm_ppf(p, mean_fc, vol_fc)
        cdf = lambda x: _norm_cdf(x, mean_fc, vol_fc)

        return _build_from_arch(r, log_ret, MODEL_ID, MODEL_NAME, DESC, ppf, cdf)
    except Exception as e:
        return _error_result(MODEL_ID, MODEL_NAME, DESC, e)


# ── Model 4: EGARCH + Skew-t ─────────────────────────────────────────────────
def fit_egarch_skewt(log_ret_pct: pd.Series, log_ret: pd.Series) -> dict:
    MODEL_ID   = "egarch_skewt"
    MODEL_NAME = "EGARCH + Skew-t"
    DESC = (
        "EGARCH models log-variance directly, so variance is always positive "
        "without constraints. Asymmetry works differently to GJR: it captures "
        "the sign AND magnitude of shocks. Combined with Skew-t for fat left tails."
    )
    try:
        m = arch_model(log_ret_pct, mean="Constant", vol="EGARCH",
                       p=1, o=1, q=1, dist="skewt", rescale=False)
        r = m.fit(disp="off", options={"maxiter": 1000})

        fc      = r.forecast(horizon=1, reindex=False)
        mean_fc = float(fc.mean.iloc[-1, 0]) / 100
        vol_fc  = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100
        nu      = float(r.params["eta"])
        lam     = float(r.params["lambda"])

        ppf = lambda p: _skewt_ppf(p, mean_fc, vol_fc, nu, lam)
        cdf = lambda x: _skewt_cdf(x, mean_fc, vol_fc, nu, lam)

        return _build_from_arch(r, log_ret, MODEL_ID, MODEL_NAME, DESC, ppf, cdf)
    except Exception as e:
        return _error_result(MODEL_ID, MODEL_NAME, DESC, e)


# ── Model 5: GARCH-in-Mean + Skew-t ──────────────────────────────────────────
def fit_garch_m_skewt(log_ret_pct: pd.Series, log_ret: pd.Series) -> dict:
    MODEL_ID   = "garch_m"
    MODEL_NAME = "GARCH-in-Mean + Skew-t"
    DESC = (
        "GARCH-in-Mean of Engle, Lilien & Robins (1987) with Hansen's skewed "
        "Student-t. The conditional mean is augmented as μ_t = c + δ·σ_t, "
        "so the risk premium is time-varying and proportional to current "
        "volatility. Implemented via iterated GJR-GARCH: σ_t is extracted "
        "from a first-stage GJR fit, then used as a regressor in the mean "
        "equation of a second-stage GJR-GARCH + Skew-t. A negative δ "
        "directly lowers the mean forecast in high-volatility states, "
        "addressing the positive-mean bias of constant-mean specifications."
    )
    try:
        # ── Stage 1: extract conditional volatility path ──────────────────────
        m1 = arch_model(log_ret_pct, mean="Constant", vol="GARCH",
                        p=1, o=1, q=1, dist="skewt", rescale=False)
        r1 = m1.fit(disp="off", options={"maxiter": 1000})
        cond_vol = np.sqrt(r1.conditional_volatility)   # % units, length T

        # ── Stage 2: ARX mean with cond_vol as regressor ──────────────────────
        x_in = cond_vol.values.reshape(-1, 1)
        m2   = arch_model(log_ret_pct, mean="ARX", lags=0,
                          x=x_in,
                          vol="GARCH", p=1, o=1, q=1,
                          dist="skewt", rescale=False)
        r2   = m2.fit(disp="off", options={"maxiter": 1000})

        # ── Manual 1-step forecast — avoids arch's x-forecasting requirement ──
        # Variance forecast: ω + (α + γ·I[ε<0])·ε²_T + β·σ²_T
        params   = r2.params
        omega    = float(params["omega"])
        alpha    = float(params["alpha[1]"])
        gamma    = float(params["gamma[1]"])
        beta     = float(params["beta[1]"])

        last_resid   = float(r2.resid.iloc[-1])
        last_var     = float(r2.conditional_volatility.iloc[-1] ** 2)
        indicator    = 1.0 if last_resid < 0 else 0.0
        var_fc       = omega + (alpha + gamma * indicator) * last_resid**2 + beta * last_var
        vol_fc_pct   = float(np.sqrt(max(var_fc, 1e-12)))   # % units
        vol_fc       = vol_fc_pct / 100                      # decimal

        # Mean forecast: c + δ·σ_{t+1|t}
        # arch names the constant "Const" and the x coefficient "x0[0]"
        const_key = next((k for k in params.index if k.lower() in ("const", "mu", "c")), None)
        x_key     = next((k for k in params.index if k.startswith("x")), None)

        const_val = float(params[const_key]) if const_key else 0.0
        delta     = float(params[x_key])     if x_key     else 0.0

        mean_fc_pct = const_val + delta * vol_fc_pct   # % units
        mean_fc     = mean_fc_pct / 100                 # decimal

        nu  = float(params["eta"])
        lam = float(params["lambda"])

        ppf = lambda p: _skewt_ppf(p, mean_fc, vol_fc, nu, lam)
        cdf = lambda x: _skewt_cdf(x, mean_fc, vol_fc, nu, lam)

        # Build result manually (can't use _build_from_arch — no forecast object)
        lower_95 = ppf(0.025);  upper_95 = ppf(0.975)
        lower_68 = ppf(0.16);   upper_68 = ppf(0.84)

        p_positive  = 1.0 - cdf(0.0)
        p_negative  =       cdf(0.0)
        p_down_1pct =       cdf(-0.01)
        p_up_1pct   = 1.0 - cdf(0.01)

        persist   = alpha + beta + 0.5 * gamma
        half_life = (
            np.log(0.5) / np.log(persist)
            if 0 < persist < 1 else float("inf")
        )

        print(f"  [garch_m] δ={delta:.4f}  c={const_val:.4f}  "
              f"μ_fc={mean_fc_pct:+.4f}%  σ_fc={vol_fc_pct:.4f}%")

        return {
            "status":      "ok",
            "model_id":    MODEL_ID,
            "model_name":  MODEL_NAME,
            "description": DESC,
            "next_date":   pd.bdate_range(
                               start=log_ret.index[-1], periods=2, freq="B"
                           )[-1],
            "mean_fc":     mean_fc,
            "vol_fc":      vol_fc,
            "ann_vol":     vol_fc * np.sqrt(252),
            "lower_95":    lower_95,
            "upper_95":    upper_95,
            "lower_68":    lower_68,
            "upper_68":    upper_68,
            "p_positive":  p_positive,
            "p_negative":  p_negative,
            "p_down_1pct": p_down_1pct,
            "p_up_1pct":   p_up_1pct,
            "nu":          float(params["eta"]),
            "lam":         lam,
            "alpha":       alpha,
            "gamma":       gamma,
            "beta":        beta,
            "persist":     persist,
            "half_life":   half_life,
            "aic":         r2.aic,
            "computed_at": datetime.now(),
            "q05":         None,
            "q10":         None,
            "q25":         None,
        }
    except Exception as e:
        return _error_result(MODEL_ID, MODEL_NAME, DESC, e)


# ── Model 6: CAViaR (Conditional Autoregressive Value at Risk) ────────────────
#
# Engle & Manganelli (2004) SAV (Symmetric Absolute Value) specification:
#   Q_t(τ) = β0 + β1·Q_{t-1}(τ) + β2·|r_{t-1}|
#
# We fit separately for τ = 0.05, 0.10, 0.25 (downside) and
# τ = 0.75, 0.90, 0.95 (upside) to get the full distribution shape.
# Probabilities are derived from the quantile forecasts directly —
# no parametric distribution assumed.

def _caviar_sav_loss(params, returns, tau):
    """Quantile regression loss for SAV-CAViaR."""
    b0, b1, b2 = params
    n   = len(returns)
    q   = np.zeros(n)
    q[0] = np.quantile(returns, tau)   # initialise at unconditional quantile
    for t in range(1, n):
        q[t] = b0 + b1 * q[t - 1] + b2 * abs(returns[t - 1])
    resid = returns - q
    loss  = np.where(resid >= 0, tau * resid, (tau - 1) * resid)
    return float(np.mean(loss))


def _fit_caviar_tau(returns: np.ndarray, tau: float) -> tuple[float, float]:
    """
    Fit SAV-CAViaR for a single quantile level tau.
    Returns (fitted_quantile_forecast, beta1_persistence).
    """
    q0 = np.quantile(returns, tau)

    def _caviar_sav_loss_bounded(params, returns, tau):
        b0, b1, b2 = params
        # Enforce stationarity and sign constraints
        if b1 >= 1.0 or b1 < 0.0 or b2 < 0.0:
            return 1e10
        n   = len(returns)
        q   = np.zeros(n)
        q[0] = q0
        for t in range(1, n):
            val = b0 + b1 * q[t - 1] + b2 * abs(returns[t - 1])
            # Clip to prevent overflow — quantile cannot exceed 5x historical range
            q[t] = np.clip(val, -500.0, 500.0)
        resid = returns - q
        loss  = np.where(resid >= 0, tau * resid, (tau - 1) * resid)
        return float(np.mean(loss))

    best_loss   = np.inf
    best_params = np.array([q0 * 0.05, 0.9, 0.1])

    rng = np.random.default_rng(42)
    starts = [
        np.array([q0 * 0.05, 0.90, 0.10]),
        np.array([q0 * 0.10, 0.85, 0.15]),
        np.array([q0 * 0.05, 0.95, 0.05]),
    ] + [
        np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.7,  0.98),   # cap at 0.98 to avoid unit root
            rng.uniform(0.01, 0.25),
        ])
        for _ in range(5)
    ]

    for x0 in starts:
        try:
            res = minimize(
                _caviar_sav_loss_bounded,
                x0,
                args=(returns, tau),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
            )
            if res.fun < best_loss:
                best_loss   = res.fun
                best_params = res.x
        except Exception:
            continue

    b0, b1, b2 = best_params
    # Final pass with clipping
    n  = len(returns)
    q  = np.zeros(n)
    q[0] = q0
    for t in range(1, n):
        q[t] = np.clip(
            b0 + b1 * q[t - 1] + b2 * abs(returns[t - 1]),
            -500.0, 500.0
        )

    return float(q[-1]), float(b1)


def fit_caviar(log_ret_pct: pd.Series, log_ret: pd.Series) -> dict:
    MODEL_ID   = "caviar"
    MODEL_NAME = "CAViaR"
    DESC = (
        "Conditional Autoregressive Value at Risk (Engle & Manganelli 2004). "
        "Models tail quantiles directly — no distribution assumed. "
        "Each quantile evolves as an AR process driven by past return magnitude. "
        "Probabilities are derived from the quantile forecasts themselves, "
        "making this the most assumption-free model in the set."
    )
    try:
        returns = log_ret_pct.dropna().values

        # Fit quantiles on both tails
        taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        q_fc = {}
        for tau in taus:
            q_fc[tau], _ = _fit_caviar_tau(returns, tau)

        # Convert quantile forecasts from % to decimal
        q_dec = {tau: v / 100 for tau, v in q_fc.items()}

        # Mean forecast: median quantile
        mean_fc = q_dec[0.50]

        # Vol proxy: (Q95 - Q05) / (2 * 1.645)  — IQR-based
        vol_fc = (q_dec[0.95] - q_dec[0.05]) / (2 * 1.645)
        vol_fc = max(vol_fc, 1e-6)   # guard against degenerate fits

        # Probabilities via linear interpolation between fitted quantiles
        # Build piecewise-linear CDF from the 7 quantile points
        q_vals  = np.array([q_dec[t] for t in taus])
        tau_arr = np.array(taus)

        def caviar_cdf(x):
            if x <= q_vals[0]:
                return float(tau_arr[0] * x / q_vals[0]) if q_vals[0] != 0 else 0.05
            if x >= q_vals[-1]:
                return 1.0 - (1.0 - tau_arr[-1]) * (1.0 - x) / (1.0 - q_vals[-1]) \
                    if q_vals[-1] != 1.0 else 0.95
            return float(np.interp(x, q_vals, tau_arr))

        def caviar_ppf(p):
            return float(np.interp(p, tau_arr, q_vals))

        lower_95 = caviar_ppf(0.025)
        upper_95 = caviar_ppf(0.975)
        lower_68 = caviar_ppf(0.16)
        upper_68 = caviar_ppf(0.84)

        p_positive  = 1.0 - caviar_cdf(0.0)
        p_negative  =       caviar_cdf(0.0)
        p_down_1pct =       caviar_cdf(-0.01)
        p_up_1pct   = 1.0 - caviar_cdf(0.01)

        next_date = pd.bdate_range(
            start=log_ret.index[-1], periods=2, freq="B"
        )[-1]

        return {
            "status":      "ok",
            "model_id":    MODEL_ID,
            "model_name":  MODEL_NAME,
            "description": DESC,
            "next_date":   next_date,
            "mean_fc":     mean_fc,
            "vol_fc":      vol_fc,
            "ann_vol":     vol_fc * np.sqrt(252),
            "lower_95":    lower_95,
            "upper_95":    upper_95,
            "lower_68":    lower_68,
            "upper_68":    upper_68,
            "p_positive":  p_positive,
            "p_negative":  p_negative,
            "p_down_1pct": p_down_1pct,
            "p_up_1pct":   p_up_1pct,
            "nu":          None,
            "lam":         None,
            "alpha":       None,
            "gamma":       None,
            "beta":        None,
            "persist":     None,
            "half_life":   None,
            "aic":         None,
            "computed_at": datetime.now(),
            "q05":         q_dec[0.05],
            "q10":         q_dec[0.10],
            "q25":         q_dec[0.25],
        }
    except Exception as e:
        return _error_result(MODEL_ID, MODEL_NAME, DESC, e)


# ── Registry ──────────────────────────────────────────────────────────────────
# Ordered list used by the UI to build the model selector tabs.
# Each entry: (model_id, display_name, fit_function)

MODEL_REGISTRY = [
    ("gjr_skewt",    "GJR + Skew-t",  fit_gjr_skewt),
    ("gjr_t",        "GJR + t",        fit_gjr_t),
    ("gjr_normal",   "GJR + Normal",   fit_gjr_normal),
    ("egarch_skewt", "EGARCH + Skew-t",fit_egarch_skewt),
    ("garch_m",      "GARCH-M + Skew-t",fit_garch_m_skewt),
    ("caviar",       "CAViaR",          fit_caviar),
]

MODEL_IDS = [m[0] for m in MODEL_REGISTRY]


def fit_model(
    model_id:    str,
    log_ret_pct: pd.Series,
    log_ret:     pd.Series,
) -> dict:
    """
    Dispatch to the correct fit function by model_id.
    Returns a ModelResult dict.
    """
    dispatch = {mid: fn for mid, _, fn in MODEL_REGISTRY}
    fn = dispatch.get(model_id)
    if fn is None:
        return _error_result(model_id, model_id, "", ValueError(f"Unknown model_id: {model_id}"))
    return fn(log_ret_pct, log_ret)


def fit_all_models(
    log_ret_pct: pd.Series,
    log_ret:     pd.Series,
    model_ids:   list[str] | None = None,
) -> dict[str, dict]:
    """
    Fit all (or a subset of) models for a single ticker.
    Returns {model_id: ModelResult}.
    """
    ids = model_ids or MODEL_IDS
    return {mid: fit_model(mid, log_ret_pct, log_ret) for mid in ids}