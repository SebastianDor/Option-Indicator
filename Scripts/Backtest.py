import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from datetime import datetime
from scipy.stats import t as t_dist
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


# ── Config ────────────────────────────────────────────────────────────────────
TICKER     = "^STOXX"
YEARS_BACK = 8
ACF_LAGS   = 30


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_prices(ticker: str, years_back: int) -> pd.Series:
    end    = datetime.now()
    start  = end.replace(year=end.year - years_back)
    prices = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)["Close"]
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices.dropna()


prices      = load_prices(TICKER, YEARS_BACK)
log_ret     = np.log(prices / prices.shift(1)).dropna()
log_ret_pct = log_ret * 100

print(f"Loaded {len(log_ret_pct)} log returns for {TICKER}")
print(f"From {log_ret_pct.index[0].date()} to {log_ret_pct.index[-1].date()}")
print(f"Mean: {log_ret_pct.mean():.4f}%  Std: {log_ret_pct.std():.4f}%")


# ── 2. Model definitions ──────────────────────────────────────────────────────
# GJR-GARCH: o=1 adds leverage term gamma
#   sigma2_t = omega + alpha*eps2_{t-1} + gamma*eps2_{t-1}*I(eps<0) + beta*sigma2_{t-1}
#   gamma > 0 confirms bad news raises vol more than good news
# EGARCH: log(sigma2_t) = omega + alpha*|z_{t-1}| + gamma*z_{t-1} + beta*log(sigma2_{t-1})
#   gamma < 0 confirms asymmetry

MODELS = {
    "ARCH(2)":        dict(mean="Constant", vol="ARCH",  p=2,           dist="t"),
    "GARCH(1,1)":     dict(mean="Constant", vol="GARCH", p=1, q=1,      dist="t"),
    "GJR-GARCH(1,1)": dict(mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH(1,1)":    dict(mean="Constant", vol="EGARCH",p=1, q=1,      dist="t"),
    "GJR-GARCH(1,2)": dict(mean="Constant", vol="GARCH", p=1, o=1, q=2, dist="t"),
}

COLORS = ["#4A90D9", "#E8734A", "#4CAF82", "#A855F7", "#F59E0B"]


# ── 3. Fit all models ─────────────────────────────────────────────────────────
fitted = {}
for name, kwargs in MODELS.items():
    print(f"\nFitting {name}...")
    try:
        m = arch_model(log_ret_pct, rescale=False, **kwargs)
        r = m.fit(disp="off", options={"maxiter": 1000})
        fitted[name] = r
        print(f"  AIC: {r.aic:.2f}  BIC: {r.bic:.2f}  Log-L: {r.loglikelihood:.2f}")
    except Exception as e:
        print(f"  Failed: {e}")


# ── 4. Comparison table ───────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  {'Model':<20} {'AIC':>10} {'BIC':>10} {'Log-L':>10} {'LB(10)p':>8}")
print(f"{'─'*65}")

comp_rows = []
for name, r in fitted.items():
    sq_resid = r.std_resid.dropna() ** 2
    lb10_p   = float(acorr_ljungbox(sq_resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
    params   = r.params
    if "gamma[1]" in params:
        asym_str = f"γ={float(params['gamma[1]']):+.4f}"
    elif "gamma" in params:
        asym_str = f"γ={float(params['gamma']):+.4f}"
    else:
        asym_str = "—"
    comp_rows.append({"Model": name, "AIC": round(r.aic, 2), "BIC": round(r.bic, 2),
                      "Log-L": round(r.loglikelihood, 2), "LB(10)p": round(lb10_p, 4),
                      "Asymmetry": asym_str})
    print(f"  {name:<20} {r.aic:>10.2f} {r.bic:>10.2f} {r.loglikelihood:>10.2f} {lb10_p:>8.4f}")

print(f"{'─'*65}")
comp_df  = pd.DataFrame(comp_rows).set_index("Model")
best_aic = comp_df["AIC"].idxmin()
best_bic = comp_df["BIC"].idxmin()
print(f"\n  Best by AIC : {best_aic}")
print(f"  Best by BIC : {best_bic}")
print(f"\n{comp_df.to_string()}")


# ── 5. Parameter details for asymmetric models ───────────────────────────────
for name in ["GJR-GARCH(1,1)", "GJR-GARCH(1,2)", "EGARCH(1,1)"]:
    if name not in fitted:
        continue
    r = fitted[name]
    print(f"\n{'─'*52}")
    print(f"  {name} parameters:")
    print(f"{'─'*52}")
    for pname, val in r.params.items():
        pval = r.pvalues[pname]
        sig  = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {pname:<20} {val:>10.6f}   p={pval:.4f} {sig}")
    params = r.params
    if "alpha[1]" in params and "beta[1]" in params:
        alpha   = float(params["alpha[1]"])
        beta    = float(params["beta[1]"])
        gamma   = float(params.get("gamma[1]", 0))
        persist = alpha + beta + 0.5 * gamma
        print(f"\n  Persistence (α + β + 0.5γ) : {persist:.4f}")
        if persist < 1:
            print(f"  Half-life of vol shock     : {np.log(0.5)/np.log(persist):.1f} days")
        else:
            print(f"  Non-stationary (persistence ≥ 1)")


# ── 6. Residual diagnostics table ────────────────────────────────────────────
print(f"\n{'─'*72}")
print(f"  {'Model':<20} {'LB(5)p':>8} {'LB(10)p':>8} {'LB(20)p':>8}  {'Within±1σ':>10}")
print(f"{'─'*72}")
for name, r in fitted.items():
    sq   = r.std_resid.dropna() ** 2
    lb5  = float(acorr_ljungbox(sq, lags=[5],  return_df=True)["lb_pvalue"].iloc[0])
    lb10 = float(acorr_ljungbox(sq, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
    lb20 = float(acorr_ljungbox(sq, lags=[20], return_df=True)["lb_pvalue"].iloc[0])
    w    = np.mean(np.abs(r.std_resid.dropna()) <= 1) * 100
    print(f"  {name:<20} {lb5:>8.4f} {lb10:>8.4f} {lb20:>8.4f}  {w:>9.1f}%")
print(f"{'─'*72}")
print("  LB p > 0.05 = no remaining ARCH effects (good)")
print("  Within ±1σ target ~68%")


# ── 7. 1-day-ahead forecast from best AIC model ──────────────────────────────
best_r      = fitted[best_aic]
forecast    = best_r.forecast(horizon=1, reindex=False)
mean_fc_pct = float(forecast.mean.iloc[-1, 0])
var_fc_pct  = float(forecast.variance.iloc[-1, 0])
mean_fc     = mean_fc_pct / 100
vol_fc      = np.sqrt(var_fc_pct) / 100
nu          = float(best_r.params["nu"])
lower_fc    = mean_fc + t_dist.ppf(0.025, df=nu) * vol_fc
upper_fc    = mean_fc + t_dist.ppf(0.975, df=nu) * vol_fc
next_date   = pd.bdate_range(start=log_ret.index[-1], periods=2, freq="B")[-1]

print(f"\n{'─'*52}")
print(f"  Best model ({best_aic}) — forecast for {next_date.date()}")
print(f"{'─'*52}")
print(f"  Forecast log return  : {mean_fc*100:+.4f}%")
print(f"  Forecast volatility  : {vol_fc*100:.4f}%  (1-day σ)")
print(f"  Annualised vol       : {vol_fc*np.sqrt(252)*100:.2f}%")
print(f"  95% CI               : [{lower_fc*100:+.4f}%, {upper_fc*100:+.4f}%]")
print(f"  Last known return    : {float(log_ret.iloc[-1])*100:+.4f}%")
print(f"  Student-t d.o.f. (ν) : {nu:.2f}")
print(f"{'─'*52}")

# ── 8. Walk-forward backtest ──────────────────────────────────────────────────
print("\nRunning walk-forward backtest for all models (takes a few minutes)...")
min_train = 504   # 2 years minimum

wf_all = {}
for name, kwargs in MODELS.items():
    print(f"  {name}...")
    rows = []
    for i in range(min_train, len(log_ret_pct) - 1):
        try:
            m  = arch_model(log_ret_pct.iloc[:i], rescale=False, **kwargs)
            r  = m.fit(disp="off", options={"maxiter": 200})
            fc = r.forecast(horizon=1, reindex=False)
            rows.append({
                "date":      log_ret.index[i + 1],
                "pred_mean": float(fc.mean.iloc[-1, 0]) / 100,
                "pred_vol":  float(np.sqrt(fc.variance.iloc[-1, 0])) / 100,
                "actual":    float(log_ret.iloc[i + 1]),
            })
        except Exception:
            continue
    wf_all[name] = pd.DataFrame(rows).set_index("date")

print(f"\n{'─'*72}")
print(f"  {'Model':<20} {'Hit%':>7} {'MAE%':>8} {'RMSE%':>8} {'Within±1σ':>11}")
print(f"{'─'*72}")
for name, wf in wf_all.items():
    hit    = np.mean(np.sign(wf["pred_mean"]) == np.sign(wf["actual"])) * 100
    mae    = np.mean(np.abs(wf["actual"] - wf["pred_mean"])) * 100
    rmse   = np.sqrt(np.mean((wf["actual"] - wf["pred_mean"])**2)) * 100
    within = np.mean(np.abs(wf["actual"]) <= wf["pred_vol"]) * 100
    print(f"  {name:<20} {hit:>7.1f} {mae:>8.4f} {rmse:>8.4f} {within:>10.1f}%")
print(f"{'─'*72}")


# ── 9. Plots ──────────────────────────────────────────────────────────────────

# Figure 1: Conditional volatility comparison
fig1, axes = plt.subplots(len(fitted), 1, figsize=(14, 3 * len(fitted)), sharex=True)
fig1.suptitle(f"Conditional Volatility Comparison — {TICKER} ({YEARS_BACK}y)",
              fontsize=13, fontweight="bold")
for ax, (name, r), color in zip(axes, fitted.items(), COLORS):
    cv = r.conditional_volatility / 100
    ax.plot(cv.index, cv * 100, color=color, linewidth=0.8,
            label=f"{name}  AIC={r.aic:.0f}")
    ax.fill_between(cv.index, 0, cv * 100, color=color, alpha=0.12)
    ax.set_ylabel("σ (%)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("garch_vol_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# Figure 2: Post-fit ACF/PACF of squared std residuals
fig2, axes = plt.subplots(len(fitted), 2, figsize=(14, 3 * len(fitted)))
fig2.suptitle(f"Post-Fit ACF/PACF — Squared Std Residuals — {TICKER}",
              fontsize=13, fontweight="bold")
for row, ((name, r), color) in enumerate(zip(fitted.items(), COLORS)):
    sq = r.std_resid.dropna() ** 2
    plot_acf(sq, lags=ACF_LAGS, ax=axes[row, 0], alpha=0.05,
             title=f"ACF² — {name}", color=color,
             vlines_kwargs={"colors": color})
    plot_pacf(sq, lags=ACF_LAGS, ax=axes[row, 1], alpha=0.05, method="ywm",
              title=f"PACF² — {name}", color=color,
              vlines_kwargs={"colors": color})
    for ax in axes[row]:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Lag")
plt.tight_layout()
plt.savefig("garch_resid_acf.png", dpi=150, bbox_inches="tight")
plt.show()


# Figure 3: Walk-forward vol forecast vs |actual return|
fig3, axes = plt.subplots(len(wf_all), 1, figsize=(14, 3 * len(wf_all)), sharex=True)
fig3.suptitle(f"Walk-Forward: Predicted σ vs |Actual Return| — {TICKER}",
              fontsize=13, fontweight="bold")
for ax, (name, wf), color in zip(axes, wf_all.items(), COLORS):
    ax.plot(wf.index, wf["pred_vol"] * 100,
            color=color, linewidth=0.9, label=f"Forecast σ — {name}")
    ax.plot(wf.index, wf["actual"].abs() * 100,
            color="gray", linewidth=0.5, alpha=0.5, label="|Actual return|")
    ax.set_ylabel("(%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("garch_wf_vol.png", dpi=150, bbox_inches="tight")
plt.show()


# Figure 4: AIC / BIC bar chart
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig4.suptitle("Model Selection — AIC & BIC", fontsize=13, fontweight="bold")

names  = list(comp_df.index)
aics   = comp_df["AIC"].values
bics   = comp_df["BIC"].values
x      = np.arange(len(names))
width  = 0.5

bars1 = ax1.bar(x, aics, width, color=COLORS, alpha=0.85, edgecolor="white")
ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax1.set_ylabel("AIC"); ax1.set_title("AIC (lower = better)")
ax1.bar_label(bars1, fmt="%.0f", fontsize=8, padding=3)
ax1.grid(True, alpha=0.3, axis="y")

bars2 = ax2.bar(x, bics, width, color=COLORS, alpha=0.85, edgecolor="white")
ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax2.set_ylabel("BIC"); ax2.set_title("BIC (lower = better, penalises params more)")
ax2.bar_label(bars2, fmt="%.0f", fontsize=8, padding=3)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("garch_aic_bic.png", dpi=150, bbox_inches="tight")
plt.show()


# Figure 5: Overlay of all conditional vols on one axis
fig5, ax = plt.subplots(figsize=(14, 5))
fig5.suptitle(f"All Models — Conditional Volatility Overlay — {TICKER}",
              fontsize=13, fontweight="bold")
for (name, r), color in zip(fitted.items(), COLORS):
    cv = r.conditional_volatility / 100
    ax.plot(cv.index, cv * 100, color=color, linewidth=0.9, alpha=0.8, label=name)
ax.set_ylabel("Daily σ (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("garch_vol_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAll plots saved:")
print("  garch_vol_comparison.png")
print("  garch_resid_acf.png")
print("  garch_wf_vol.png")
print("  garch_aic_bic.png")
print("  garch_vol_overlay.png")