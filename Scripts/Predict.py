import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from datetime import datetime
from scipy.stats import t as t_dist
from statsmodels.stats.diagnostic import acorr_ljungbox


# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = {
    "^STOXX":    "STOXX 600",
    "^STOXX50E": "STOXX 50",
    "^AEX":      "AEX",
    "^GSPC":     "S&P 500",
    "^GDAXI":    "DAX",
    "^NDX":      "Nasdaq 100",
}
YEARS_BACK = 8
COLORS     = ["#4A90D9", "#E8734A", "#4CAF82", "#A855F7", "#F59E0B", "#EC4899"]


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_log_returns(ticker: str, years_back: int) -> pd.Series:
    end    = datetime.now()
    start  = end.replace(year=end.year - years_back)
    prices = yf.Ticker(ticker).history(
        start=start, end=end, interval="1d", auto_adjust=True
    )["Close"]
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return np.log(prices.dropna() / prices.dropna().shift(1)).dropna()


print("Loading data...")
returns = {}
for ticker in TICKERS:
    returns[ticker] = load_log_returns(ticker, YEARS_BACK)
    print(f"  {TICKERS[ticker]:<12} {len(returns[ticker])} obs  "
          f"({returns[ticker].index[0].date()} → {returns[ticker].index[-1].date()})")


# ── 2. Fit GJR-GARCH(1,1)-t ──────────────────────────────────────────────────
def fit_gjr_garch(log_ret: pd.Series) -> object:
    m = arch_model(log_ret * 100, mean="Constant", vol="GARCH",
                   p=1, o=1, q=1, dist="t", rescale=False)
    return m.fit(disp="off", options={"maxiter": 1000})


print("\nFitting GJR-GARCH(1,1)-t for all indexes...")
fitted = {}
for ticker, label in TICKERS.items():
    print(f"  {label}...", end=" ")
    try:
        fitted[ticker] = fit_gjr_garch(returns[ticker])
        r = fitted[ticker]
        print(f"AIC={r.aic:.1f}  Log-L={r.loglikelihood:.1f}")
    except Exception as e:
        print(f"FAILED: {e}")


# ── 3. Forecast + directional probabilities ───────────────────────────────────
def get_forecast(result, log_ret: pd.Series) -> dict:
    fc       = result.forecast(horizon=1, reindex=False)
    mean_fc  = float(fc.mean.iloc[-1, 0]) / 100
    vol_fc   = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100
    nu       = float(result.params["nu"])

    # Confidence intervals — 68% and 95%
    lower_95 = mean_fc + t_dist.ppf(0.025, df=nu) * vol_fc
    upper_95 = mean_fc + t_dist.ppf(0.975, df=nu) * vol_fc
    lower_68 = mean_fc + t_dist.ppf(0.16,  df=nu) * vol_fc
    upper_68 = mean_fc + t_dist.ppf(0.84,  df=nu) * vol_fc

    # Directional probabilities using the fitted Student-t
    # P(r > 0) and P(r < 0) given forecast mean and vol
    p_positive = 1 - t_dist.cdf(0, df=nu, loc=mean_fc, scale=vol_fc)
    p_negative =     t_dist.cdf(0, df=nu, loc=mean_fc, scale=vol_fc)

    # Tail probabilities — useful for risk
    p_down_1pct = t_dist.cdf(-0.01, df=nu, loc=mean_fc, scale=vol_fc)   # P(r < -1%)
    p_up_1pct   = 1 - t_dist.cdf(0.01, df=nu, loc=mean_fc, scale=vol_fc) # P(r > +1%)

    p     = result.params
    alpha = float(p.get("alpha[1]", np.nan))
    gamma = float(p.get("gamma[1]", np.nan))
    beta  = float(p.get("beta[1]",  np.nan))

    return {
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
        "alpha":       alpha,
        "gamma":       gamma,
        "beta":        beta,
        "persist":     alpha + beta + 0.5 * gamma,
        "last_ret":    float(log_ret.iloc[-1]),
        "aic":         result.aic,
    }


forecasts = {t: get_forecast(fitted[t], returns[t]) for t in fitted}


# ── 4. Summary tables ─────────────────────────────────────────────────────────
print(f"\n{'─'*80}")
print(f"  {'Index':<12} {'Forecast%':>10} {'Vol%/day':>9} {'AnnVol%':>9} "
      f"{'P(↑)':>7} {'P(↓)':>7} {'P(<-1%)':>8} {'P(>+1%)':>8}")
print(f"{'─'*80}")
for ticker, label in TICKERS.items():
    if ticker not in forecasts: continue
    f = forecasts[ticker]
    print(f"  {label:<12} {f['mean_fc']*100:>+10.4f} {f['vol_fc']*100:>9.4f} "
          f"{f['ann_vol']*100:>9.2f} "
          f"{f['p_positive']*100:>7.1f}% {f['p_negative']*100:>7.1f}% "
          f"{f['p_down_1pct']*100:>8.1f}% {f['p_up_1pct']*100:>8.1f}%")
print(f"{'─'*80}")

print(f"\n{'─'*72}")
print(f"  {'Index':<12} {'68% CI':>24} {'95% CI':>26}")
print(f"{'─'*72}")
for ticker, label in TICKERS.items():
    if ticker not in forecasts: continue
    f = forecasts[ticker]
    ci68 = f"[{f['lower_68']*100:+.3f}%, {f['upper_68']*100:+.3f}%]"
    ci95 = f"[{f['lower_95']*100:+.3f}%, {f['upper_95']*100:+.3f}%]"
    print(f"  {label:<12} {ci68:>24} {ci95:>26}")
print(f"{'─'*72}")

print(f"\n{'─'*78}")
print(f"  {'Index':<12} {'Alpha':>8} {'Gamma':>8} {'Beta':>8} "
      f"{'nu':>6}  {'Half-life':>10}  {'AIC':>10}")
print(f"{'─'*78}")
for ticker, label in TICKERS.items():
    if ticker not in forecasts: continue
    f  = forecasts[ticker]
    hl = f"{np.log(0.5)/np.log(f['persist']):.1f}d" if f["persist"] < 1 else "∞"
    print(f"  {label:<12} {f['alpha']:>8.4f} {f['gamma']:>8.4f} {f['beta']:>8.4f} "
          f"{f['nu']:>6.2f}  {hl:>10}  {f['aic']:>10.1f}")
print(f"{'─'*78}")


# ── 5. Residual diagnostics ───────────────────────────────────────────────────
print(f"\n{'─'*72}")
print(f"  {'Index':<12} {'LB(5)p':>8} {'LB(10)p':>8} {'LB(20)p':>8}  {'Within±1σ':>10}")
print(f"{'─'*72}")
for ticker, label in TICKERS.items():
    if ticker not in fitted: continue
    r    = fitted[ticker]
    sq   = r.std_resid.dropna() ** 2
    lb5  = float(acorr_ljungbox(sq, lags=[5],  return_df=True)["lb_pvalue"].iloc[0])
    lb10 = float(acorr_ljungbox(sq, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
    lb20 = float(acorr_ljungbox(sq, lags=[20], return_df=True)["lb_pvalue"].iloc[0])
    w    = np.mean(np.abs(r.std_resid.dropna()) <= 1) * 100
    print(f"  {label:<12} {lb5:>8.4f} {lb10:>8.4f} {lb20:>8.4f}  {w:>9.1f}%")
print(f"{'─'*72}")
print("  LB p > 0.05 = no remaining ARCH effects (good)")


# ── 6. Walk-forward backtest ──────────────────────────────────────────────────
print("\nRunning walk-forward backtest for all indexes (takes several minutes)...")
min_train = 504
wf_all    = {}

for ticker, label in TICKERS.items():
    if ticker not in fitted: continue
    print(f"  {label}...")
    log_ret_pct = returns[ticker] * 100
    rows = []
    for i in range(min_train, len(log_ret_pct) - 1):
        try:
            m  = arch_model(log_ret_pct.iloc[:i], mean="Constant", vol="GARCH",
                            p=1, o=1, q=1, dist="t", rescale=False)
            r  = m.fit(disp="off", options={"maxiter": 200})
            fc = r.forecast(horizon=1, reindex=False)
            nu_wf   = float(r.params["nu"])
            mu_wf   = float(fc.mean.iloc[-1, 0]) / 100
            vol_wf  = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100
            p_pos   = 1 - t_dist.cdf(0, df=nu_wf, loc=mu_wf, scale=vol_wf)
            rows.append({
                "date":      returns[ticker].index[i + 1],
                "pred_mean": mu_wf,
                "pred_vol":  vol_wf,
                "p_pos":     p_pos,
                "actual":    float(returns[ticker].iloc[i + 1]),
            })
        except Exception:
            continue
    wf_all[ticker] = pd.DataFrame(rows).set_index("date")

print(f"\n{'─'*80}")
print(f"  {'Index':<12} {'Hit%':>7} {'ProbHit%':>9} {'MAE%':>8} "
      f"{'RMSE%':>8} {'Within±1σ':>11}")
print(f"{'─'*80}")
for ticker, label in TICKERS.items():
    if ticker not in wf_all: continue
    wf       = wf_all[ticker]
    hit      = np.mean(np.sign(wf["pred_mean"]) == np.sign(wf["actual"])) * 100
    # Probability-weighted hit: did the direction with higher prob win?
    prob_hit = np.mean(
        ((wf["p_pos"] > 0.5) & (wf["actual"] > 0)) |
        ((wf["p_pos"] < 0.5) & (wf["actual"] < 0))
    ) * 100
    mae      = np.mean(np.abs(wf["actual"] - wf["pred_mean"])) * 100
    rmse     = np.sqrt(np.mean((wf["actual"] - wf["pred_mean"])**2)) * 100
    within   = np.mean(np.abs(wf["actual"]) <= wf["pred_vol"]) * 100
    print(f"  {label:<12} {hit:>7.1f} {prob_hit:>9.1f} {mae:>8.4f} "
          f"{rmse:>8.4f} {within:>10.1f}%")
print(f"{'─'*80}")
print("  ProbHit% = directional accuracy using P(↑) > 50% as signal")


# ── 7. Plots ──────────────────────────────────────────────────────────────────

# Figure 1: Forecast per index — returns + vol bands + forecast dot + prob gauge
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle("GJR-GARCH(1,1) — 1-Day-Ahead Forecasts", fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, (ticker, label) in enumerate(TICKERS.items()):
    if ticker not in forecasts: continue
    ax      = axes[i]
    color   = COLORS[i]
    f       = forecasts[ticker]
    r       = fitted[ticker]
    log_ret = returns[ticker]
    window  = 60

    bar_colors = ["#4CAF82" if v >= 0 else "#E8734A"
                  for v in log_ret.values[-window:]]
    ax.bar(log_ret.index[-window:], log_ret.values[-window:] * 100,
           color=bar_colors, alpha=0.6, width=0.8)

    cv = r.conditional_volatility.iloc[-window:] / 100
    ax.fill_between(cv.index, -cv * 100, cv * 100,
                    color=color, alpha=0.15, label="±1σ GARCH")

    # 68% CI (inner) and 95% CI (outer)
    ax.errorbar(f["next_date"], f["mean_fc"] * 100,
                yerr=[[(f["mean_fc"] - f["lower_95"]) * 100],
                      [(f["upper_95"] - f["mean_fc"]) * 100]],
                fmt="none", color="#FFD700", capsize=4, linewidth=1.0,
                alpha=0.5, label="95% CI")
    ax.errorbar(f["next_date"], f["mean_fc"] * 100,
                yerr=[[(f["mean_fc"] - f["lower_68"]) * 100],
                      [(f["upper_68"] - f["mean_fc"]) * 100]],
                fmt="o", color="#FFD700", markersize=9, capsize=5,
                linewidth=2.0, zorder=5, label="68% CI")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    p_up  = f["p_positive"] * 100
    p_dn  = f["p_negative"] * 100
    arrow = "↑" if p_up >= 50 else "↓"
    prob  = max(p_up, p_dn)
    ax.set_title(
        f"{label}  |  σ={f['vol_fc']*100:.2f}%/day  ({f['ann_vol']*100:.1f}% ann.)\n"
        f"P(↑)={p_up:.1f}%  P(↓)={p_dn:.1f}%  "
        f"P(<-1%)={f['p_down_1pct']*100:.1f}%  P(>+1%)={f['p_up_1pct']*100:.1f}%",
        fontsize=8
    )
    ax.set_ylabel("Log return (%)", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=7, rotation=20)

plt.tight_layout()
plt.savefig("gjr_forecasts.png", dpi=150, bbox_inches="tight")
plt.show()


# Figure 2: Directional probability heatmap across all indexes
fig2, ax = plt.subplots(figsize=(10, 4))
fig2.suptitle("GJR-GARCH(1,1) — Directional Probability Summary",
              fontsize=13, fontweight="bold")

prob_data = np.array([
    [forecasts[t]["p_positive"] * 100,
     forecasts[t]["p_negative"] * 100,
     forecasts[t]["p_up_1pct"]  * 100,
     forecasts[t]["p_down_1pct"]* 100]
    for t in TICKERS if t in forecasts
])
col_labels  = ["P(↑)", "P(↓)", "P(>+1%)", "P(<-1%)"]
row_labels  = [TICKERS[t] for t in TICKERS if t in forecasts]

im = ax.imshow(prob_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=10)
for r_idx in range(len(row_labels)):
    for c_idx in range(len(col_labels)):
        val = prob_data[r_idx, c_idx]
        ax.text(c_idx, r_idx, f"{val:.1f}%", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="black" if 25 < val < 75 else "white")
plt.colorbar(im, ax=ax, label="Probability (%)")
plt.tight_layout()
plt.savefig("gjr_prob_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# Figure 3: Conditional volatility overlay
fig3, ax = plt.subplots(figsize=(14, 5))
fig3.suptitle("GJR-GARCH(1,1) — Conditional Volatility — All Indexes",
              fontsize=13, fontweight="bold")
for i, (ticker, label) in enumerate(TICKERS.items()):
    if ticker not in fitted: continue
    cv = fitted[ticker].conditional_volatility / 100
    ax.plot(cv.index, cv * 100, color=COLORS[i], linewidth=0.9,
            alpha=0.85, label=label)
ax.set_ylabel("Daily σ (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gjr_vol_overlay.png", dpi=150, bbox_inches="tight")
plt.show()


# Figure 4: Walk-forward P(↑) over time for each index
fig4, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
fig4.suptitle("Walk-Forward: P(↑) Over Time — All Indexes",
              fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, (ticker, label) in enumerate(TICKERS.items()):
    if ticker not in wf_all: continue
    ax  = axes[i]
    wf  = wf_all[ticker]
    hit = np.mean(np.sign(wf["pred_mean"]) == np.sign(wf["actual"])) * 100

    # Color the P(↑) line by whether the actual return was positive
    ax.plot(wf.index, wf["p_pos"] * 100,
            color=COLORS[i], linewidth=0.8, alpha=0.9, label="P(↑)")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="50%")
    ax.fill_between(wf.index, 50, wf["p_pos"] * 100,
                    where=wf["p_pos"] > 0.5,
                    color="#4CAF82", alpha=0.15, label="Bullish")
    ax.fill_between(wf.index, wf["p_pos"] * 100, 50,
                    where=wf["p_pos"] < 0.5,
                    color="#E8734A", alpha=0.15, label="Bearish")
    ax.set_ylim(0, 100)
    ax.set_ylabel("P(↑) %", fontsize=8)
    ax.set_title(f"{label}  |  Hit={hit:.1f}%", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=7, rotation=20)

plt.tight_layout()
plt.savefig("gjr_prob_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAll plots saved:")
print("  gjr_forecasts.png         — 60-day returns + 68/95% CI + probabilities")
print("  gjr_prob_heatmap.png      — probability heatmap across all indexes")
print("  gjr_vol_overlay.png       — conditional vol overlay")
print("  gjr_prob_timeseries.png   — walk-forward P(↑) over time")
