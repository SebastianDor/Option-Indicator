from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import yfinance as yf
import numpy as np
import psutil
import time
import threading
import pickle
import os
from arch import arch_model
from scipy.stats import t as t_dist


# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.pkl")

def cache_save(key: str, data) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump({"date": datetime.now().date(), "data": data}, f)

def cache_load(key: str):
    """Returns data if cached today, else None."""
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj["date"] == datetime.now().date():
            print(f"[cache] HIT  — {key}")
            return obj["data"]
        print(f"[cache] STALE — {key}")
    except Exception as e:
        print(f"[cache] ERROR — {key}: {e}")
    return None


# ── GARCH ─────────────────────────────────────────────────────────────────────
def compute_gjr_garch_forecasts(tickers: list[str], years_back: int = 8) -> dict:
    """
    Fits GJR-GARCH(1,1)-t for each ticker.
    Checks disk cache first — only recomputes if cache is stale or missing.
    Safe to call from a background thread.
    """
    cached = cache_load("garch_forecasts")
    if cached is not None:
        return cached

    print("[GARCH] Cache miss — fitting models...")
    results = {}
    end   = datetime.now()
    start = end.replace(year=end.year - years_back)

    for ticker in tickers:
        try:
            prices = yf.Ticker(ticker).history(
                start=start, end=end, interval="1d", auto_adjust=True
            )["Close"]
            if hasattr(prices.index, "tz") and prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            log_ret     = np.log(prices.dropna() / prices.dropna().shift(1)).dropna()
            log_ret_pct = log_ret * 100

            m = arch_model(log_ret_pct, mean="Constant", vol="GARCH",
                           p=1, o=1, q=1, dist="t", rescale=False)
            r = m.fit(disp="off", options={"maxiter": 1000})

            fc       = r.forecast(horizon=1, reindex=False)
            mean_fc  = float(fc.mean.iloc[-1, 0]) / 100
            vol_fc   = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100
            nu       = float(r.params["nu"])

            lower_95 = mean_fc + t_dist.ppf(0.025, df=nu) * vol_fc
            upper_95 = mean_fc + t_dist.ppf(0.975, df=nu) * vol_fc
            lower_68 = mean_fc + t_dist.ppf(0.16,  df=nu) * vol_fc
            upper_68 = mean_fc + t_dist.ppf(0.84,  df=nu) * vol_fc

            p_positive  = float(1 - t_dist.cdf(0,     df=nu, loc=mean_fc, scale=vol_fc))
            p_negative  = float(    t_dist.cdf(0,     df=nu, loc=mean_fc, scale=vol_fc))
            p_down_1pct = float(    t_dist.cdf(-0.01, df=nu, loc=mean_fc, scale=vol_fc))
            p_up_1pct   = float(1 - t_dist.cdf( 0.01, df=nu, loc=mean_fc, scale=vol_fc))

            params  = r.params
            alpha   = float(params.get("alpha[1]", np.nan))
            gamma   = float(params.get("gamma[1]", np.nan))
            beta    = float(params.get("beta[1]",  np.nan))
            persist = alpha + beta + 0.5 * gamma

            results[ticker] = {
                "status":      "ok",
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
                "persist":     persist,
                "half_life":   np.log(0.5) / np.log(persist) if persist < 1 else float("inf"),
                "aic":         r.aic,
                "computed_at": datetime.now(),
            }
            print(f"  [GARCH] {ticker} done — "
                  f"P(↑)={p_positive*100:.1f}%  σ={vol_fc*100:.3f}%/day")

        except Exception as e:
            print(f"  [GARCH] {ticker} FAILED: {e}")
            results[ticker] = {"status": "error", "error": str(e)}

    cache_save("garch_forecasts", results)
    print("[GARCH] Results cached to disk.")
    return results


# ── Market data ───────────────────────────────────────────────────────────────
def get_index_returns(
    tickers: list[str],
    years_back: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads daily log returns and cumulative returns.
    Checks disk cache first — only re-downloads if cache is stale or missing.
    """
    cached = cache_load("market_returns")
    if cached is not None:
        return cached

    print("[data] Cache miss — downloading market data...")
    end_date   = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date.replace(year=end_date.year - years_back)
    returns = pd.DataFrame(); cum_returns = pd.DataFrame()

    for ticker in tickers:
        try:
            prices = yf.Ticker(ticker).history(
                start=start_date, end=end_date, interval="1d", auto_adjust=True
            )["Close"]
            if hasattr(prices.index, "tz") and prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            if prices.dropna().empty:
                returns[ticker] = np.nan; cum_returns[ticker] = np.nan; continue
            log_ret = np.log(prices / prices.shift(1))
            returns[ticker]     = log_ret
            cum_returns[ticker] = np.exp(log_ret.cumsum()) - 1
        except Exception as e:
            print(f"[warning] Failed {ticker}: {e}")
            returns[ticker] = np.nan; cum_returns[ticker] = np.nan

    result = (returns, cum_returns)
    cache_save("market_returns", result)
    print("[data] Market data cached to disk.")
    return result

def get_live_prices(tickers: list[str]) -> dict:
    prices = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="5d", interval="1m")
            prices[ticker] = float(hist["Close"].dropna().iloc[-1]) if not hist.empty else None
        except Exception:
            prices[ticker] = None
    return prices

def compute_live_log_return(live_prices: dict, tickers: list[str]) -> dict:
    live_ret = {}
    for ticker in tickers:
        try:
            live_px = live_prices.get(ticker)
            if live_px is None: live_ret[ticker] = None; continue
            hist   = yf.Ticker(ticker).history(period="5d", interval="1d")
            closes = hist["Close"].dropna()
            if len(closes) < 2: live_ret[ticker] = None; continue
            live_ret[ticker] = float(np.log(live_px / float(closes.iloc[-2])))
        except Exception:
            live_ret[ticker] = None
    return live_ret

def compute_live_cum_return(cum_returns: pd.DataFrame,
                             live_prices: dict,
                             tickers: list[str]) -> dict:
    live_cum = {}
    for ticker in tickers:
        try:
            live_px = live_prices.get(ticker)
            if live_px is None: live_cum[ticker] = None; continue
            last_cum   = cum_returns[ticker].dropna().iloc[-1]
            last_close = float(
                yf.Ticker(ticker).history(period="5d", interval="1d")["Close"].dropna().iloc[-1]
            )
            live_cum[ticker] = live_px / (last_close / (1 + last_cum)) - 1
        except Exception:
            live_cum[ticker] = None
    return live_cum

def compute_forecast_cumulative(
    cum_returns: pd.DataFrame,
    garch_res:   dict,
    tickers:     list[str],
) -> dict:
    """
    For each ticker, compute the absolute cumulative return level
    that the GARCH forecast implies, anchored to the last known
    historical close. Cached to disk so it survives app restarts.
    """
    cached = cache_load("forecast_cumulative")
    if cached is not None:
        return cached

    result = {}
    for ticker in tickers:
        fc = garch_res.get(ticker, {})
        if fc.get("status") != "ok":
            continue
        try:
            last_cum   = float(cum_returns[ticker].dropna().iloc[-1])
            last_date  = cum_returns[ticker].dropna().index[-1]
            fc_return  = fc["mean_fc"]
            fc_cum     = (1 + last_cum) * (1 + fc_return) - 1
            result[ticker] = {
                "anchor_date": last_date,        # last known historical date
                "anchor_cum":  last_cum * 100,   # cumulative % at anchor
                "fc_date":     fc["next_date"],  # predicted date (e.g. 26th)
                "fc_cum":      fc_cum * 100,     # predicted cumulative %
                "fc_return":   fc_return * 100,  # predicted daily return %
                "fc_68_low":   fc["lower_68"] * 100,
                "fc_68_high":  fc["upper_68"] * 100,
            }
        except Exception as e:
            print(f"[forecast_cum] {ticker} failed: {e}")

    cache_save("forecast_cumulative", result)
    print("[forecast_cum] Cached forecast cumulative levels.")
    return result


# ── Config ────────────────────────────────────────────────────────────────────
TICKERS       = ["^STOXX", "^STOXX50E", "^AEX", "^GSPC", "^GDAXI", "^NDX"]
TICKER_LABELS = {
    "^STOXX": "STOXX 600", "^STOXX50E": "STOXX 50", "^AEX": "AEX",
    "^GSPC": "S&P 500", "^GDAXI": "DAX", "^NDX": "Nasdaq 100",
}
TICKER_COLORS = ["#4A90D9", "#E8734A", "#4CAF82", "#A855F7", "#F59E0B", "#EC4899"]
CUM_HIDDEN    = {"^STOXX50E", "^NDX", "^GDAXI", "^GSPC"}
REFRESH_CHOICES = {"0": "Off", "5": "5s", "10": "10s", "30": "30s", "60": "60s"}

THEMES = {
    "light": {
        "primary": "#4A90D9", "primary_dark": "#2C5F8A",
        "sidebar_bg": "#1E2A38", "sidebar_text": "#CBD5E1",
        "sidebar_hover": "#2E3D50", "sidebar_active": "#4A90D9",
        "page_bg": "#F4F6F9", "card_bg": "#FFFFFF", "border": "#E2E8F0",
        "text_main": "#1E293B", "text_muted": "#64748B",
        "settings_bg": "#FFFFFF", "settings_border": "#E2E8F0", "divider": "#E2E8F0",
        "plot_bg": "#FFFFFF", "plot_paper": "#F4F6F9",
        "plot_grid": "#E2E8F0", "plot_font": "#1E293B",
        "positive": "#16A34A", "negative": "#DC2626",
    },
    "dark": {
        "primary": "#4A90D9", "primary_dark": "#2C5F8A",
        "sidebar_bg": "#0F1720", "sidebar_text": "#CBD5E1",
        "sidebar_hover": "#1A2535", "sidebar_active": "#4A90D9",
        "page_bg": "#1A1F2E", "card_bg": "#242B3D", "border": "#2E3A4E",
        "text_main": "#E8EDF5", "text_muted": "#8A9BB5",
        "settings_bg": "#242B3D", "settings_border": "#2E3A4E", "divider": "#2E3A4E",
        "plot_bg": "#242B3D", "plot_paper": "#1A1F2E",
        "plot_grid": "#2E3A4E", "plot_font": "#E8EDF5",
        "positive": "#4ADE80", "negative": "#F87171",
    },
}

def make_css(c):
    return f"""
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: {c['page_bg']}; color: {c['text_main']}; height: 100vh; overflow: hidden; transition: background 0.3s, color 0.3s; }}
    .app-shell {{ display: flex; height: 100vh; }}
    #active_theme, label[for=active_theme], #proc_page, label[for=proc_page] {{ display: none; }}
    .sidebar {{ width: 220px; min-width: 220px; background: {c['sidebar_bg']}; color: {c['sidebar_text']}; display: flex; flex-direction: column; padding: 1rem 0; overflow: hidden; transition: width 0.3s ease, min-width 0.3s ease, padding 0.3s ease; white-space: nowrap; }}
    .sidebar.collapsed {{ width: 0; min-width: 0; padding: 0; }}
    .sidebar .app-title {{ font-size: 1.1rem; font-weight: 700; color: #fff; padding: 0.5rem 1.25rem 1.25rem; border-bottom: 1px solid {c['primary_dark']}; margin-bottom: 0.5rem; overflow: hidden; }}
    .sidebar .sidebar-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: {c['text_muted']}; padding: 0.75rem 1.25rem 0.25rem; overflow: hidden; }}
    .sidebar .sidebar-item {{ display: flex; align-items: center; gap: 0.6rem; padding: 0.6rem 1.25rem; cursor: pointer; font-size: 0.9rem; color: {c['sidebar_text']}; border-left: 3px solid transparent; transition: background 0.15s, border-color 0.15s; text-decoration: none; background: none; border-right: none; border-top: none; border-bottom: none; width: 100%; text-align: left; overflow: hidden; }}
    .sidebar .sidebar-item:hover {{ background: {c['sidebar_hover']}; }}
    .sidebar .sidebar-item.active {{ background: {c['sidebar_hover']}; border-left-color: {c['sidebar_active']}; color: #fff; }}
    .sidebar-trigger {{ position: fixed; left: 0; top: 0; width: 18px; height: 100vh; z-index: 100; }}
    .sidebar-toggle {{ position: fixed; top: 50%; transform: translateY(-50%); left: 220px; transition: left 0.3s ease; z-index: 200; background: {c['primary']}; border: none; color: #fff; width: 18px; height: 48px; border-radius: 0 6px 6px 0; cursor: pointer; font-size: 0.65rem; display: flex; align-items: center; justify-content: center; opacity: 0.75; }}
    .sidebar-toggle:hover {{ opacity: 1; }}
    .sidebar-toggle.collapsed {{ left: 0; }}
    .refresh-control {{ padding: 0.5rem 1.25rem 0.75rem; overflow: hidden; }}
    .refresh-control label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: {c['text_muted']}; display: block; margin-bottom: 0.4rem; }}
    .refresh-control select {{ width: 100%; background: {c['sidebar_hover']}; color: {c['sidebar_text']}; border: 1px solid {c['primary_dark']}; border-radius: 5px; padding: 0.3rem 0.5rem; font-size: 0.82rem; cursor: pointer; outline: none; }}
    .main-content {{ flex: 1; overflow-y: auto; padding: 2rem; background: {c['page_bg']}; transition: background 0.3s; }}
    .page {{ display: none; }}
    .page.active {{ display: block; }}
    .page-header {{ margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid {c['border']}; }}
    .page-header h1 {{ font-size: 1.5rem; font-weight: 600; color: {c['text_main']}; }}
    .page-header p {{ font-size: 0.9rem; color: {c['text_muted']}; margin-top: 0.25rem; }}
    .placeholder-card {{ background: {c['card_bg']}; border-radius: 8px; padding: 2rem; margin-bottom: 1.5rem; min-height: 180px; display: flex; align-items: center; justify-content: center; color: {c['text_muted']}; font-size: 0.95rem; border: 2px dashed {c['border']}; }}
    .plot-card {{ position: relative; background: {c['card_bg']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; min-width: 0; overflow: hidden; }}
    .plot-card h3 {{ font-size: 0.95rem; font-weight: 600; color: {c['text_main']}; margin-bottom: 0.25rem; padding-right: 2rem; }}
    .plot-card p {{ font-size: 0.8rem; color: {c['text_muted']}; margin-bottom: 1rem; }}
    .plot-row {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.25rem; margin-bottom: 1.5rem; }}
    .plot-row > * {{ min-width: 0; }}
    .dist-icon-btn {{ position: absolute; top: 0.85rem; right: 0.85rem; background: none; border: none; cursor: pointer; font-size: 1.05rem; color: {c['text_muted']}; line-height: 1; padding: 0.2rem; transition: color 0.15s, transform 0.3s; border-radius: 50%; z-index: 10; }}
    .dist-icon-btn:hover {{ color: {c['primary']}; }}
    .dist-icon-btn.active {{ color: {c['primary']}; transform: rotate(180deg); }}
    .chart-slot {{ display: block; }}
    .chart-slot.hidden {{ display: none; }}
    .stat-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.25rem; margin-bottom: 1.5rem; }}
    .stat-card {{ background: {c['card_bg']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 1.25rem 1.5rem; }}
    .stat-card .stat-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: {c['text_muted']}; margin-bottom: 0.4rem; }}
    .stat-card .stat-value {{ font-size: 1.6rem; font-weight: 700; color: {c['text_main']}; }}
    .stat-card .stat-sub {{ font-size: 0.78rem; color: {c['text_muted']}; margin-top: 0.2rem; }}
    .proc-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .proc-table th {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 2px solid {c['border']}; color: {c['text_muted']}; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .proc-table td {{ padding: 0.45rem 0.75rem; border-bottom: 1px solid {c['border']}; color: {c['text_main']}; }}
    .proc-table tr:last-child td {{ border-bottom: none; }}
    .settings-page {{ max-width: 620px; }}
    .settings-section {{ background: {c['settings_bg']}; border: 1px solid {c['settings_border']}; border-radius: 10px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; }}
    .settings-section h2 {{ font-size: 1rem; font-weight: 600; color: {c['text_main']}; margin-bottom: 0.25rem; }}
    .settings-section .section-desc {{ font-size: 0.82rem; color: {c['text_muted']}; margin-bottom: 1.25rem; }}
    .settings-divider {{ border: none; border-top: 1px solid {c['divider']}; margin: 1.25rem 0; }}
    .settings-placeholder {{ font-size: 0.88rem; color: {c['text_muted']}; font-style: italic; padding: 0.5rem 0; }}
    .theme-options {{ display: flex; gap: 1rem; margin-top: 0.25rem; }}
    .theme-option {{ display: flex; align-items: center; gap: 0.5rem; cursor: pointer; padding: 0.6rem 1.1rem; border-radius: 8px; border: 2px solid {c['border']}; background: {c['page_bg']}; color: {c['text_main']}; font-size: 0.9rem; transition: border-color 0.2s, background 0.2s; user-select: none; }}
    .theme-option input[type=radio] {{ accent-color: {c['primary']}; }}
    .theme-option.selected {{ border-color: {c['primary']}; background: {c['card_bg']}; }}
    .garch-loading {{ display: flex; align-items: center; gap: 0.75rem; color: {c['text_muted']}; font-size: 0.9rem; padding: 1.5rem; }}
    .garch-spinner {{ width: 18px; height: 18px; border: 2px solid {c['border']}; border-top-color: {c['primary']}; border-radius: 50%; animation: spin 0.8s linear infinite; flex-shrink: 0; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .forecast-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; margin-bottom: 1.5rem; }}
    .forecast-card {{ background: {c['card_bg']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 1.25rem 1.5rem; }}
    .forecast-card .fc-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem; }}
    .forecast-card .fc-label {{ font-size: 0.8rem; font-weight: 600; color: {c['text_muted']}; text-transform: uppercase; letter-spacing: 0.05em; }}
    .forecast-card .fc-date {{ font-size: 0.72rem; color: {c['text_muted']}; }}
    .forecast-card .fc-return {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem; }}
    .forecast-card .fc-return.positive {{ color: {c['positive']}; }}
    .forecast-card .fc-return.negative {{ color: {c['negative']}; }}
    .forecast-card .fc-vol {{ font-size: 0.82rem; color: {c['text_muted']}; margin-bottom: 0.75rem; }}
    .forecast-card .fc-ci {{ font-size: 0.75rem; color: {c['text_muted']}; margin-bottom: 0.2rem; }}
    .forecast-card .fc-ci span {{ color: {c['text_main']}; font-weight: 500; }}
    .ci-bar-wrap {{ position: relative; height: 6px; background: {c['border']}; border-radius: 3px; margin: 0.5rem 0 0.75rem; }}
    .ci-bar-95 {{ position: absolute; height: 6px; background: {c['primary']}; opacity: 0.25; border-radius: 3px; }}
    .ci-bar-68 {{ position: absolute; height: 6px; background: {c['primary']}; opacity: 0.6; border-radius: 3px; }}
    .ci-dot {{ position: absolute; top: 50%; transform: translate(-50%, -50%); width: 10px; height: 10px; border-radius: 50%; background: {c['primary']}; border: 2px solid {c['card_bg']}; }}
    .prob-row {{ display: flex; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap; }}
    .prob-pill {{ flex: 1; text-align: center; padding: 0.3rem 0.4rem; border-radius: 6px; font-size: 0.72rem; font-weight: 600; min-width: 60px; }}
    .prob-pill.up   {{ background: rgba(22,163,74,0.12);  color: {c['positive']}; }}
    .prob-pill.down {{ background: rgba(220,38,38,0.12);  color: {c['negative']}; }}
    .prob-pill.tail {{ background: {c['border']};          color: {c['text_muted']}; }}
    .prob-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .prob-table th {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 2px solid {c['border']}; color: {c['text_muted']}; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .prob-table td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid {c['border']}; color: {c['text_main']}; vertical-align: middle; }}
    .prob-table tr:last-child td {{ border-bottom: none; }}
    .prob-bar-cell {{ width: 130px; }}
    .prob-bar-wrap-sm {{ display: flex; align-items: center; gap: 0.4rem; }}
    .prob-bar-bg {{ background: {c['border']}; border-radius: 3px; height: 7px; flex: 1; }}
    .prob-bar-fill {{ height: 7px; border-radius: 3px; }}
    .badge {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 4px; font-size: 0.72rem; font-weight: 700; }}
    .badge.bull {{ background: rgba(22,163,74,0.12);  color: {c['positive']}; }}
    .badge.bear {{ background: rgba(220,38,38,0.12);  color: {c['negative']}; }}
    .cache-badge {{ display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.72rem; color: {c['text_muted']}; background: {c['border']}; padding: 0.2rem 0.6rem; border-radius: 4px; margin-left: 0.75rem; }}
    .cum-modal-overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:1000; align-items:center; justify-content:center; }}
    .cum-modal-overlay.open {{ display:flex; }}
    .cum-modal-box {{ background:{c['card_bg']}; border-radius:12px; padding:0; width:92vw; height:88vh; display:flex; flex-direction:column; overflow:hidden; position:relative; }}
    .cum-modal-header {{ display:flex; align-items:center; justify-content:space-between; padding:0.75rem 1rem; border-bottom:1px solid {c['border']}; flex-shrink:0; }}
    .cum-modal-title {{ font-size:0.95rem; font-weight:600; color:{c['text_main']}; }}
    .cum-modal-close {{ background:none; border:none; font-size:1.3rem; cursor:pointer; color:{c['text_muted']}; line-height:1; padding:0.2rem 0.4rem; border-radius:4px; }}
    .cum-modal-close:hover {{ color:{c['text_main']}; background:{c['border']}; }}
    #cum-modal-plot {{ flex:1; min-height:0; }}
    """

def plotly_layout(theme: dict, height: int = 300) -> dict:
    return dict(
        paper_bgcolor=theme["plot_paper"], plot_bgcolor=theme["plot_bg"],
        font=dict(color=theme["plot_font"], family="Segoe UI", size=12),
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(gridcolor=theme["plot_grid"], zeroline=False),
        yaxis=dict(gridcolor=theme["plot_grid"], zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"), height=height,
    )

# ── UI helpers ────────────────────────────────────────────────────────────────
def make_page(page_id, title, subtitle, active=False):
    return ui.tags.div(
        {"class": f"page {'active' if active else ''}", "id": f"page-{page_id}"},
        ui.tags.div({"class": "page-header"}, ui.tags.h1(title), ui.tags.p(subtitle)),
        ui.tags.div("Placeholder", **{"class": "placeholder-card"}),
        ui.tags.div("Placeholder", **{"class": "placeholder-card"}),
    )

def make_analysis_page():
    return ui.tags.div(
        {"class": "page", "id": "page-analysis"},
        ui.tags.div(
            {"class": "cum-modal-overlay", "id": "cumModalOverlay",
             "onclick": "if(event.target===this) closeCumModal()"},
            ui.tags.div(
                {"class": "cum-modal-box"},
                ui.tags.div(
                    {"class": "cum-modal-header"},
                    ui.tags.span("Cumulative Returns", **{"class": "cum-modal-title"}),
                    ui.tags.button("✕", **{
                        "class":   "cum-modal-close",
                        "onclick": "closeCumModal()",
                    }),
                ),
                ui.tags.div({"id": "cum-modal-plot"}),
            ),
        ),        
        ui.tags.div({"class": "page-header"},
            ui.tags.h1("Analysis"),
            ui.tags.p("Index return distributions, cumulative performance and GJR-GARCH(1,1) forecasts.")),
        # ── Distribution / box plots ──────────────────────────────────────────
        ui.tags.div(
            {"class": "plot-row"},
            *[
                ui.tags.div(
                    {"class": "plot-card"},
                    ui.tags.h3(f"{TICKER_LABELS[t]} — Return Distribution"),
                    ui.tags.p("Daily log returns"),
                    ui.tags.button("↻", **{
                        "class": "dist-icon-btn",
                        "id":    f"toggle-btn-{t.replace('^','')}",
                        "onclick": f"toggleChart('{t.replace('^','')}')",
                        "title": "Toggle distribution view",
                    }),
                    ui.tags.div(
                        {"class": "chart-slot", "id": f"slot-box-{t.replace('^','')}"},
                        output_widget(f"box_{t.replace('^','')}")),
                    ui.tags.div(
                        {"class": "chart-slot hidden", "id": f"slot-dist-{t.replace('^','')}"},
                        output_widget(f"dist_{t.replace('^','')}")),
                )
                for t in TICKERS
            ],
        ),
        ui.tags.div({"class": "plot-card"},
            ui.tags.h3("Cumulative Returns"),
            ui.tags.button("⛶", **{
                "id":      "btn-popout-cum",
                "onclick": "popoutCumulative()",
                "title":   "Pop out chart",
                "style":   "position:absolute;top:0.85rem;right:0.85rem;"
                           "background:none;border:none;cursor:pointer;"
                           "font-size:1.1rem;opacity:0.5;"
                           "transition:opacity 0.15s;",
                "onmouseover": "this.style.opacity='1'",
                "onmouseout":  "this.style.opacity='0.5'",
            }),
            output_widget("line_cum")),
    )

def make_directional_page():
    return ui.tags.div(
        {"class": "page", "id": "page-directional"},
        ui.tags.div({"class": "page-header"},
            ui.tags.h1("Directional Probabilities"),
            ui.tags.p("GJR-GARCH(1,1)-t derived probabilities for tomorrow's return direction and tail events. Cached daily.")),
        ui.output_ui("garch_prob_content"),
    )

def make_server_page():
    return ui.tags.div(
        {"class": "page", "id": "page-server"},
        ui.tags.div({"class": "page-header"},
            ui.tags.h1("Server"),
            ui.tags.p("Live system resource usage and running processes.")),
        ui.tags.div({"class": "stat-row"},
            ui.output_ui("stat_cpu"),
            ui.output_ui("stat_ram"),
            ui.output_ui("stat_disk"),
            ui.output_ui("stat_uptime")),
        ui.tags.div({"class": "plot-card"},
            ui.tags.h3("Top Processes"),
            ui.tags.p("Top 15 processes by CPU usage"),
            ui.output_ui("proc_table")),
    )

def make_settings_page():
    return ui.tags.div(
        {"class": "page", "id": "page-settings"},
        ui.tags.div({"class": "page-header"},
            ui.tags.h1("Settings"),
            ui.tags.p("Manage your application preferences.")),
        ui.tags.div(
            {"class": "settings-page"},
            ui.tags.div(
                {"class": "settings-section"},
                ui.tags.h2("Appearance"),
                ui.tags.p("Choose how the application looks.", **{"class": "section-desc"}),
                ui.tags.div(
                    {"class": "theme-options", "id": "themeOptions"},
                    ui.tags.label(
                        {"class": "theme-option selected", "id": "opt-light", "onclick": "setTheme('light')"},
                        ui.tags.input(type="radio", name="theme", value="light", checked=True),
                        "☀️  Light mode"),
                    ui.tags.label(
                        {"class": "theme-option", "id": "opt-dark", "onclick": "setTheme('dark')"},
                        ui.tags.input(type="radio", name="theme", value="dark"),
                        "🌙  Dark mode"),
                ),
            ),
            ui.tags.div({"class": "settings-section"},
                ui.tags.h2("Cache"),
                ui.tags.p("Cached files are stored in .cache/ next to app.py and expire at midnight.", **{"class": "section-desc"}),
                ui.tags.hr(**{"class": "settings-divider"}),
                ui.output_ui("cache_status"),
            ),
            ui.tags.div({"class": "settings-section"},
                ui.tags.h2("Account"),
                ui.tags.p("Manage your account details.", **{"class": "section-desc"}),
                ui.tags.hr(**{"class": "settings-divider"}),
                ui.tags.p("— Placeholder —", **{"class": "settings-placeholder"})),
            ui.tags.div({"class": "settings-section"},
                ui.tags.h2("Notifications"),
                ui.tags.p("Configure notification preferences.", **{"class": "section-desc"}),
                ui.tags.hr(**{"class": "settings-divider"}),
                ui.tags.p("— Placeholder —", **{"class": "settings-placeholder"})),
        ),
    )


# ── Precompute CSS strings for JS injection ───────────────────────────────────
_light_css = make_css(THEMES["light"]).replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
_dark_css  = make_css(THEMES["dark"]).replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

# ── app_ui ────────────────────────────────────────────────────────────────────
app_ui = ui.tags.div(
    ui.tags.style(make_css(THEMES["light"]), id="themeStyle"),
    ui.input_text("active_theme", label="", value="light"),
    ui.input_numeric("proc_page", label="", value=1),

    ui.tags.div(
        {"class": "app-shell"},
        ui.tags.div({"class": "sidebar-trigger", "id": "sidebarTrigger"}),
        ui.tags.button("◀", id="sidebarToggle", **{"class": "sidebar-toggle"}),

        ui.tags.div(
            {"class": "sidebar", "id": "sidebar"},
            ui.tags.div("MyApp", **{"class": "app-title"}),

            ui.tags.span("Menu", **{"class": "sidebar-label"}),
            ui.tags.button("🏠  Dashboard",   **{"class": "sidebar-item active", "onclick": "setPage('dashboard')"}),
            ui.tags.button("📊  Analysis",    **{"class": "sidebar-item",        "onclick": "setPage('analysis')"}),
            ui.tags.button("🎯  Directional", **{"class": "sidebar-item",        "onclick": "setPage('directional')"}),
            ui.tags.button("🖥️  Server",      **{"class": "sidebar-item",        "onclick": "setPage('server')"}),
            ui.tags.button("👤  Users",       **{"class": "sidebar-item",        "onclick": "setPage('users')"}),

            ui.tags.span("Other", **{"class": "sidebar-label"}),
            ui.tags.button("⚙️  Settings",    **{"class": "sidebar-item",        "onclick": "setPage('settings')"}),
            ui.tags.button("❓  Help",        **{"class": "sidebar-item",        "onclick": "setPage('help')"}),

            ui.tags.span("Refresh", **{
                "class": "sidebar-label",
                "id":    "refresh-label",
                "style": "display:none",
            }),
            ui.tags.div(
                {"class": "refresh-control", "id": "refresh-control", "style": "display:none"},
                ui.input_select("refresh_interval", label="Auto-refresh",
                                choices=REFRESH_CHOICES, selected="0"),
            ),
        ),

        ui.tags.div(
            {"class": "main-content"},
            make_page("dashboard", "Dashboard", "Welcome to your dashboard.", active=True),
            make_analysis_page(),
            make_directional_page(),
            make_server_page(),
            make_page("users",    "Users",    "Manage users and permissions."),
            make_settings_page(),
            make_page("help",     "Help",     "Documentation and support."),
        ),
    ),

    # Theme + sidebar + navigation script (f-string)
    ui.tags.script(f"""
        const lightCSS = `{_light_css}`;
        const darkCSS  = `{_dark_css}`;

        function setTheme(theme) {{
            document.getElementById('themeStyle').textContent = theme === 'dark' ? darkCSS : lightCSS;
            document.getElementById('opt-light').classList.toggle('selected', theme === 'light');
            document.getElementById('opt-dark').classList.toggle('selected',  theme === 'dark');
            const el = document.getElementById('active_theme');
            el.value = theme; el.dispatchEvent(new Event('change'));
        }}

        const sidebar = document.getElementById('sidebar');
        const toggle  = document.getElementById('sidebarToggle');
        const trigger = document.getElementById('sidebarTrigger');
        let pinned = true, hideTimer = null;

        function showSidebar() {{
            clearTimeout(hideTimer);
            sidebar.classList.remove('collapsed');
            toggle.classList.remove('collapsed');
            toggle.textContent = '◀';
        }}
        function hideSidebar() {{
            sidebar.classList.add('collapsed');
            toggle.classList.add('collapsed');
            toggle.textContent = '▶';
        }}
        toggle.addEventListener('click', () => {{
            pinned = !pinned;
            pinned ? showSidebar() : hideSidebar();
        }});
        trigger.addEventListener('mouseenter', () => {{ if (!pinned) showSidebar(); }});
        sidebar.addEventListener('mouseleave', () => {{ if (!pinned) hideTimer = setTimeout(hideSidebar, 300); }});
        sidebar.addEventListener('mouseenter', () => clearTimeout(hideTimer));

        const REFRESH_PAGES = ['analysis', 'server', 'directional'];
        function setPage(name) {{
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById('page-' + name).classList.add('active');
            document.querySelectorAll('.sidebar-item').forEach(b => b.classList.remove('active'));
            event.currentTarget.classList.add('active');
            const show = REFRESH_PAGES.includes(name);
            document.getElementById('refresh-control').style.display = show ? 'block' : 'none';
            document.getElementById('refresh-label').style.display   = show ? 'block' : 'none';
        }}

        function updatePage(delta) {{
            const el = document.getElementById('proc_page');
            el.value = Math.max(1, parseInt(el.value || 1) + delta);
            el.dispatchEvent(new Event('change'));
        }}

        function toggleChart(key) {{
            const box  = document.getElementById('slot-box-'  + key);
            const dist = document.getElementById('slot-dist-' + key);
            const btn  = document.getElementById('toggle-btn-' + key);
            const showingBox = !box.classList.contains('hidden');
            box.classList.toggle('hidden',   showingBox);
            dist.classList.toggle('hidden', !showingBox);
            btn.classList.toggle('active',   showingBox);
        }}
    """),

    # Popout cumulative chart script (plain string, no f-string)
    ui.tags.script("""
        function popoutCumulative() {
            const container = document.getElementById('line_cum');
            const plotDiv   = container ? container.querySelector('.js-plotly-plot') : null;
            if (!plotDiv || !plotDiv.data) {
                alert('Chart not ready yet.');
                return;
            }

            const overlay = document.getElementById('cumModalOverlay');
            const target  = document.getElementById('cum-modal-plot');
            overlay.classList.add('open');

            // Copy layout including current zoom/pan state
            const layout = {
                ...plotDiv.layout,
                autosize: true,
                modebar: { orientation: 'v', bgcolor: 'rgba(0,0,0,0)' },
            };
            delete layout.width;
            delete layout.height;

            Plotly.newPlot(target, plotDiv.data, layout,
                {responsive: true, displayModeBar: true, displaylogo: false})
            .then(function() {
                window._cumModalPlot = target;

                // Sync zoom/pan from modal back to the original chart
                target.on('plotly_relayout', function(ed) {
                    if (!ed['xaxis.range[0]']) return;
                    Plotly.relayout(plotDiv, {
                        'xaxis.range[0]': ed['xaxis.range[0]'],
                        'xaxis.range[1]': ed['xaxis.range[1]'],
                    });
                });

                // Sync legend visibility from modal back to original
                target.on('plotly_restyle', function(ed) {
                    Plotly.restyle(plotDiv, ed[0], ed[1]);
                });
            });
        }

        function closeCumModal() {
            const overlay = document.getElementById('cumModalOverlay');
            const target  = document.getElementById('cum-modal-plot');
            const container = document.getElementById('line_cum');
            const plotDiv   = container ? container.querySelector('.js-plotly-plot') : null;

            // Sync zoom state back to the small chart on close
            if (plotDiv && target && target.layout) {
                const xl = target.layout.xaxis || {};
                const yl = target.layout.yaxis || {};
                const update = {};
                if (xl.range) {
                    update['xaxis.range[0]'] = xl.range[0];
                    update['xaxis.range[1]'] = xl.range[1];
                }
                if (yl.range) {
                    update['yaxis.range[0]'] = yl.range[0];
                    update['yaxis.range[1]'] = yl.range[1];
                }
                if (Object.keys(update).length) {
                    Plotly.relayout(plotDiv, update);
                }

                // Sync legend visibility (which traces are hidden)
                if (target.data && plotDiv.data) {
                    const visUpdate = target.data.map(function(t) { return t.visible; });
                    Plotly.restyle(plotDiv, {visible: visUpdate});
                }
            }

            // Purge modal plot to free memory
            if (window._cumModalPlot) {
                Plotly.purge(target);
                window._cumModalPlot = null;
            }

            overlay.classList.remove('open');
        }

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeCumModal();
        });
    """),
)

def server(input, output, session):

    # ── Load market data synchronously — fast, no threading needed ────────────
    r, c = get_index_returns(TICKERS, years_back=4)
    r = r.dropna()
    c = c.loc[r.index]
    market_data   = reactive.Value((r, c))        # set immediately, never None

    _initial_prices = get_live_prices(TICKERS)              # plain variable, no reactive
    live_data       = reactive.Value(_initial_prices)
    live_log_rets   = reactive.Value(
        compute_live_log_return(_initial_prices, TICKERS)   # use plain variable
    )
    garch_results = reactive.Value(None)           # None until thread finishes

    # ── Background threads ────────────────────────────────────────────────────
    # ── GARCH on background thread — safe session callback on completion ──────
    # ── GARCH background thread ───────────────────────────────────────────────
    _garch_ready = threading.Event()
    _garch_data  = {}

    def _run_garch():
        print("[GARCH] Background thread started...")
        res = compute_gjr_garch_forecasts(TICKERS, years_back=8)
        _garch_data.update(res)
        _garch_ready.set()
        print("[GARCH] Done.")

    threading.Thread(target=_run_garch, daemon=True).start()

    @reactive.Effect
    def _poll_garch():
        if garch_results.get() is None:
            reactive.invalidate_later(2)        # check every 2s
            if _garch_ready.is_set():
                garch_results.set(dict(_garch_data))

    # ── Auto-refresh live prices ──────────────────────────────────────────────
    @reactive.Effect
    def _auto_refresh():
        interval = int(input.refresh_interval())
        if interval > 0:
            reactive.invalidate_later(interval)
            prices = get_live_prices(TICKERS)
            live_data.set(prices)
            live_log_rets.set(compute_live_log_return(prices, TICKERS))

    def current_theme() -> dict:
        return THEMES.get(input.active_theme(), THEMES["light"])

    # ── GARCH forecast cards (Analysis page) ─────────────────────────────────
    @render.ui
    def garch_forecast_cards():
        c   = current_theme()
        res = garch_results.get()

        if res is None:
            return ui.tags.div({"class": "garch-loading"},
                ui.tags.div({"class": "garch-spinner"}),
                ui.tags.span("Computing GJR-GARCH(1,1) forecasts… this takes ~30–60 seconds."))

        # Determine if results came from cache
        sample    = next((v for v in res.values() if v.get("status") == "ok"), None)
        cache_ts  = sample["computed_at"].strftime("%d %b %Y %H:%M") if sample else "—"
        from_disk = (sample["computed_at"].date() == datetime.now().date()
                     and (datetime.now() - sample["computed_at"]).seconds > 10) if sample else False

        cards = []
        for ticker in TICKERS:
            label = TICKER_LABELS[ticker]
            color = TICKER_COLORS[TICKERS.index(ticker)]
            f     = res.get(ticker, {})

            if f.get("status") != "ok":
                cards.append(ui.tags.div({"class": "forecast-card"},
                    ui.tags.div({"class": "fc-header"},
                        ui.tags.span(label, **{"class": "fc-label"})),
                    ui.tags.p("Model failed.",
                        style=f"color:{c['text_muted']};font-size:0.85rem;")))
                continue

            ret_pct  = f["mean_fc"] * 100
            ret_cls  = "positive" if ret_pct >= 0 else "negative"
            ret_sign = "+" if ret_pct >= 0 else ""

            # CI bar: map values into 0–100% within a ±window span
            window   = max(abs(f["lower_95"]) * 100, abs(f["upper_95"]) * 100, 1.5)
            def to_pct(v): return (v * 100 + window) / (2 * window) * 100
            pct_95_l = to_pct(f["lower_95"]); pct_95_r = to_pct(f["upper_95"])
            pct_68_l = to_pct(f["lower_68"]); pct_68_r = to_pct(f["upper_68"])
            pct_dot  = to_pct(f["mean_fc"])

            p_up = f["p_positive"] * 100
            p_dn = f["p_negative"] * 100

            cards.append(ui.tags.div({"class": "forecast-card"},
                ui.tags.div({"class": "fc-header"},
                    ui.tags.span(label, **{"class": "fc-label",
                        "style": f"color:{color};"}),
                    ui.tags.span(
                        f"→ {f['next_date'].strftime('%d %b')}",
                        **{"class": "fc-date"})),
                ui.tags.div(f"{ret_sign}{ret_pct:.3f}%",
                    **{"class": f"fc-return {ret_cls}"}),
                ui.tags.div(
                    f"σ = {f['vol_fc']*100:.3f}%/day  "
                    f"({f['ann_vol']*100:.1f}% ann.)",
                    **{"class": "fc-vol"}),
                ui.tags.div({"class": "ci-bar-wrap"},
                    ui.tags.div({"class": "ci-bar-95",
                        "style": f"left:{pct_95_l:.1f}%;"
                                 f"width:{pct_95_r - pct_95_l:.1f}%"}),
                    ui.tags.div({"class": "ci-bar-68",
                        "style": f"left:{pct_68_l:.1f}%;"
                                 f"width:{pct_68_r - pct_68_l:.1f}%"}),
                    ui.tags.div({"class": "ci-dot",
                        "style": f"left:{pct_dot:.1f}%"})),
                ui.tags.div({"class": "fc-ci"},
                    ui.tags.span("68% CI  "),
                    ui.tags.span(
                        f"[{f['lower_68']*100:+.3f}%, "
                        f"{f['upper_68']*100:+.3f}%]")),
                ui.tags.div({"class": "fc-ci"},
                    ui.tags.span("95% CI  "),
                    ui.tags.span(
                        f"[{f['lower_95']*100:+.3f}%, "
                        f"{f['upper_95']*100:+.3f}%]")),
                ui.tags.div({"class": "prob-row"},
                    ui.tags.div(f"↑ {p_up:.1f}%",
                        **{"class": "prob-pill up"}),
                    ui.tags.div(f"↓ {p_dn:.1f}%",
                        **{"class": "prob-pill down"}),
                    ui.tags.div(f">+1%: {f['p_up_1pct']*100:.1f}%",
                        **{"class": "prob-pill tail"}),
                    ui.tags.div(f"<-1%: {f['p_down_1pct']*100:.1f}%",
                        **{"class": "prob-pill tail"})),
            ))

        return ui.tags.div(
            ui.tags.div({"class": "forecast-grid"}, *cards),
            ui.tags.div({"style": "margin-top:0.5rem;"},
                ui.tags.span({"class": "cache-badge"},
                    "💾 cached" if from_disk else "⚡ computed",
                    f" {cache_ts}")),
        )

    forecast_cum = reactive.Value(None)

    @reactive.Effect
    def _compute_forecast_cum():
        res = garch_results.get()
        if res is None:
            return
        if forecast_cum.get() is not None:
            return
        d = market_data()
        fc = compute_forecast_cumulative(d[1], res, TICKERS)
        forecast_cum.set(fc)

    # ── Directional probability page ──────────────────────────────────────────
    @render.ui
    def garch_prob_content():
        c   = current_theme()
        res = garch_results.get()

        if res is None:
            return ui.tags.div({"class": "garch-loading"},
                ui.tags.div({"class": "garch-spinner"}),
                ui.tags.span("Computing GJR-GARCH(1,1) forecasts… please wait."))

        sample   = next((v for v in res.values() if v.get("status") == "ok"), None)
        cache_ts = sample["computed_at"].strftime("%d %b %Y %H:%M") if sample else "—"

        # ── Summary table ─────────────────────────────────────────────────────
        def prob_bar_cell(pct, bar_color):
            return ui.tags.td({"class": "prob-bar-cell"},
                ui.tags.div({"class": "prob-bar-wrap-sm"},
                    ui.tags.div({"class": "prob-bar-bg"},
                        ui.tags.div({"class": "prob-bar-fill",
                            "style": f"width:{min(pct,100):.1f}%;"
                                     f"background:{bar_color};"})),
                    ui.tags.span(f"{pct:.1f}%",
                        style="font-size:0.75rem;white-space:nowrap;")))

        header = ui.tags.thead(ui.tags.tr(
            ui.tags.th("Index"),
            ui.tags.th("Forecast"),
            ui.tags.th("P(↑)"),
            ui.tags.th("P(↓)"),
            ui.tags.th("P(>+1%)"),
            ui.tags.th("P(<-1%)"),
            ui.tags.th("Ann. Vol"),
            ui.tags.th("Persist."),
            ui.tags.th("Half-life"),
            ui.tags.th("Signal"),
        ))

        rows = []
        for ticker in TICKERS:
            label = TICKER_LABELS[ticker]
            color = TICKER_COLORS[TICKERS.index(ticker)]
            f     = res.get(ticker, {})

            if f.get("status") != "ok":
                rows.append(ui.tags.tr(
                    ui.tags.td(label),
                    ui.tags.td("—", colspan="9")))
                continue

            ret_pct  = f["mean_fc"] * 100
            ret_sign = "+" if ret_pct >= 0 else ""
            ret_col  = c["positive"] if ret_pct >= 0 else c["negative"]
            p_up     = f["p_positive"]  * 100
            p_dn     = f["p_negative"]  * 100
            sig_cls  = "bull" if p_up >= 50 else "bear"
            signal   = "BULL" if p_up >= 50 else "BEAR"
            hl       = f"{f['half_life']:.1f}d" if f["half_life"] < 500 else "∞"

            rows.append(ui.tags.tr(
                ui.tags.td(ui.tags.span(label,
                    style=f"font-weight:600;color:{color};")),
                ui.tags.td(ui.tags.span(
                    f"{ret_sign}{ret_pct:.3f}%",
                    style=f"font-weight:600;color:{ret_col};")),
                prob_bar_cell(p_up, c["positive"]),
                prob_bar_cell(p_dn, c["negative"]),
                prob_bar_cell(f["p_up_1pct"]   * 100, c["primary"]),
                prob_bar_cell(f["p_down_1pct"] * 100, c["primary"]),
                ui.tags.td(f"{f['ann_vol']*100:.1f}%"),
                ui.tags.td(f"{f['persist']:.4f}"),
                ui.tags.td(hl),
                ui.tags.td(ui.tags.span(signal,
                    **{"class": f"badge {sig_cls}"})),
            ))

        table_card = ui.tags.div({"class": "plot-card"},
            ui.tags.h3("Directional Probability Summary"),
            ui.tags.p(
                f"GJR-GARCH(1,1)-t  |  8-year window  |  "
                f"Computed: {cache_ts}"),
            ui.tags.table({"class": "prob-table"},
                header, ui.tags.tbody(*rows)))

        # ── Per-index gauge cards ─────────────────────────────────────────────
        gauge_cards = []
        for ticker in TICKERS:
            label = TICKER_LABELS[ticker]
            color = TICKER_COLORS[TICKERS.index(ticker)]
            f     = res.get(ticker, {})
            if f.get("status") != "ok":
                continue

            p_up = f["p_positive"] * 100
            p_dn = f["p_negative"] * 100
            sig_cls = "bull" if p_up >= 50 else "bear"
            signal  = "BULL" if p_up >= 50 else "BEAR"

            gauge_cards.append(ui.tags.div({"class": "forecast-card"},
                ui.tags.div({"class": "fc-header"},
                    ui.tags.span(label, **{
                        "class": "fc-label",
                        "style": f"color:{color};"}),
                    ui.tags.span(signal,
                        **{"class": f"badge {sig_cls}"})),
                ui.tags.div(
                    {"style": "display:flex;gap:0.5rem;margin:0.75rem 0;"},
                    ui.tags.div(
                        {"style": "flex:1;background:rgba(22,163,74,0.1);"
                                  "border-radius:8px;padding:0.75rem;"
                                  "text-align:center;"},
                        ui.tags.div("P(↑)",
                            style=f"font-size:0.7rem;"
                                  f"color:{c['text_muted']};"),
                        ui.tags.div(f"{p_up:.1f}%",
                            style=f"font-size:1.4rem;font-weight:700;"
                                  f"color:{c['positive']};")),
                    ui.tags.div(
                        {"style": "flex:1;background:rgba(220,38,38,0.1);"
                                  "border-radius:8px;padding:0.75rem;"
                                  "text-align:center;"},
                        ui.tags.div("P(↓)",
                            style=f"font-size:0.7rem;"
                                  f"color:{c['text_muted']};"),
                        ui.tags.div(f"{p_dn:.1f}%",
                            style=f"font-size:1.4rem;font-weight:700;"
                                  f"color:{c['negative']};"))),
                ui.tags.div({"class": "prob-row"},
                    ui.tags.div(
                        f">+1%: {f['p_up_1pct']*100:.1f}%",
                        **{"class": "prob-pill tail"}),
                    ui.tags.div(
                        f"<-1%: {f['p_down_1pct']*100:.1f}%",
                        **{"class": "prob-pill tail"}),
                    ui.tags.div(
                        f"ν = {f['nu']:.1f}",
                        **{"class": "prob-pill tail"})),
                ui.tags.div(
                    (f"σ={f['vol_fc']*100:.3f}%/day  |  "
                     f"persist={f['persist']:.4f}  |  "
                     f"half-life={f['half_life']:.1f}d"
                     if f["half_life"] < 500 else
                     f"σ={f['vol_fc']*100:.3f}%/day  |  "
                     f"persist={f['persist']:.4f}  |  half-life=∞"),
                    style=f"font-size:0.75rem;color:{c['text_muted']};"
                          f"margin-top:0.5rem;"),
            ))

        gauges = ui.tags.div(
            {"class": "forecast-grid", "style": "margin-top:1.5rem;"},
            *gauge_cards)

        cache_note = ui.tags.div(
            {"style": "margin-top:0.5rem;"},
            ui.tags.span({"class": "cache-badge"}, f"💾 {cache_ts}"))

        return ui.tags.div(table_card, gauges, cache_note)


    # ── Cache status (Settings page) ──────────────────────────────────────────
    @render.ui
    def cache_status():
        c     = current_theme()
        rows  = []
        keys  = {
            "market_returns":      "Market returns",
            "garch_forecasts":     "GARCH forecasts",
            "forecast_cumulative": "Forecast cumulative",
        }
        for key, label in keys.items():
            path = _cache_path(key)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                    ts    = obj["date"]
                    fresh = ts == datetime.now().date()
                    size  = os.path.getsize(path) / 1024
                    rows.append(ui.tags.tr(
                        ui.tags.td(label),
                        ui.tags.td(str(ts)),
                        ui.tags.td(
                            ui.tags.span(
                                "Fresh" if fresh else "Stale",
                                **{"class": "badge " + ("bull" if fresh else "bear")})),
                        ui.tags.td(f"{size:.1f} KB"),
                    ))
                except Exception:
                    rows.append(ui.tags.tr(
                        ui.tags.td(label),
                        ui.tags.td("—"), ui.tags.td("Error"), ui.tags.td("—")))
            else:
                rows.append(ui.tags.tr(
                    ui.tags.td(label),
                    ui.tags.td("—"),
                    ui.tags.td(ui.tags.span("Missing",
                        **{"class": "badge bear"})),
                    ui.tags.td("—")))

        return ui.tags.table({"class": "prob-table"},
            ui.tags.thead(ui.tags.tr(
                ui.tags.th("Dataset"),
                ui.tags.th("Cached date"),
                ui.tags.th("Status"),
                ui.tags.th("Size"))),
            ui.tags.tbody(*rows))


    # ── Plot helpers ──────────────────────────────────────────────────────────
    def make_box_fig(col, label, color, live_log_ret=None, garch_fc=None):
        theme    = current_theme()
        hist_pct = col.values * 100
        all_pct  = np.append(hist_pct, live_log_ret * 100) \
                   if live_log_ret is not None else hist_pct
        mn  = float(np.min(all_pct))
        q1  = float(np.percentile(all_pct, 25))
        med = float(np.median(all_pct))
        mn2 = float(np.mean(all_pct))
        q3  = float(np.percentile(all_pct, 75))
        mx  = float(np.max(all_pct))
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=all_pct, x0=0, name=label,
            marker_color=color, marker=dict(color=color, size=4),
            boxmean="sd", boxpoints="outliers",
            hoverinfo="none", width=0.4))
        fig.add_trace(go.Scatter(
            x=[0], y=[med], mode="markers",
            marker=dict(size=12, opacity=0, color=color),
            showlegend=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Max: {mx:.2f}%<br>Q3: {q3:.2f}%<br>"
                f"Mean: {mn2:.2f}%<br>Median: {med:.2f}%<br>"
                f"Q1: {q1:.2f}%<br>Min: {mn:.2f}%<extra></extra>")))
        # Today's live return
        if live_log_ret is not None:
            lp = live_log_ret * 100
            fig.add_trace(go.Scatter(
                x=[0], y=[lp], mode="markers+text",
                marker=dict(symbol="x", size=12, color="#FFD700",
                            line=dict(color="#FFD700", width=2)),
                text=["  today"], textposition="middle right",
                textfont=dict(size=9, color="#FFD700"),
                showlegend=False,
                hovertemplate=f"<b>Today</b><br>{lp:.3f}%<extra></extra>"))
        # GARCH forecast point — diamond, distinct from today
        if garch_fc and garch_fc.get("status") == "ok":
            fp    = garch_fc["mean_fc"] * 100
            fv    = garch_fc["vol_fc"]  * 100
            f68l  = garch_fc["lower_68"] * 100
            f68h  = garch_fc["upper_68"] * 100
            fdate = garch_fc["next_date"].strftime("%d %b")
            fig.add_trace(go.Scatter(
                x=[0], y=[fp], mode="markers+text",
                marker=dict(symbol="diamond", size=13, color="#A855F7",
                            line=dict(color="white", width=1.5)),
                text=[f"  forecast {fdate}"], textposition="middle right",
                textfont=dict(size=9, color="#A855F7"),
                showlegend=False,
                hovertemplate=(
                    f"<b>GARCH forecast</b><br>"
                    f"μ = {fp:+.3f}%<br>"
                    f"σ = {fv:.3f}%/day<br>"
                    f"68% CI: [{f68l:+.3f}%, {f68h:+.3f}%]"
                    f"<extra></extra>")))
        layout = plotly_layout(theme)
        layout.update(
            showlegend=False,
            margin=dict(l=40, r=10, t=10, b=30),
            autosize=True,
            xaxis=dict(visible=False, range=[-1, 1],
                       gridcolor=theme["plot_grid"]),
            yaxis=dict(title="Daily return (%)", ticksuffix="%",
                       gridcolor=theme["plot_grid"],
                       zeroline=True, zerolinecolor=theme["plot_grid"]))
        fig.update_layout(**layout)
        return fig

    def make_dist_fig(col, label, color, live_log_ret=None):
        from scipy.stats import gaussian_kde
        theme    = current_theme()
        hist_pct = col.dropna().values * 100
        kde      = gaussian_kde(hist_pct, bw_method="scott")
        x_range  = np.linspace(hist_pct.min() - 0.5, hist_pct.max() + 0.5, 400)
        kde_vals = kde(x_range)
        live_pct = pct_rank = None
        if live_log_ret is not None:
            live_pct = live_log_ret * 100
            pct_rank = float(np.mean(hist_pct <= live_pct) * 100)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=hist_pct, histnorm="probability density",
            name="Historical", marker_color=color,
            opacity=0.45, nbinsx=60,
            hovertemplate="Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_vals, mode="lines", name="KDE",
            line=dict(color=color, width=2.5), hoverinfo="skip"))
        if live_pct is not None:
            y_max = float(kde_vals.max()) * 1.15
            fig.add_shape(type="line",
                x0=live_pct, x1=live_pct, y0=0, y1=y_max,
                line=dict(color="#FFD700", width=2, dash="dash"))
            fig.add_trace(go.Scatter(
                x=[live_pct], y=[y_max * 0.5],
                mode="markers+text",
                marker=dict(symbol="x", size=12, color="#FFD700",
                            line=dict(color="#FFD700", width=2)),
                text=[f"  Today: {live_pct:.2f}%<br>"
                      f"  Pctile: {pct_rank:.0f}th"],
                textposition="middle right",
                textfont=dict(size=10, color="#FFD700"),
                name="Today",
                hovertemplate=(
                    f"<b>Today's return</b><br>"
                    f"{live_pct:.2f}%<br>"
                    f"Percentile: {pct_rank:.1f}th<extra></extra>")))
        mean_v = float(np.mean(hist_pct))
        std_v  = float(np.std(hist_pct))
        med_v  = float(np.median(hist_pct))
        layout = plotly_layout(theme, height=300)
        layout.update(
            bargap=0.05, showlegend=True,
            margin=dict(l=40, r=10, t=10, b=30),
            xaxis=dict(title="Daily log return (%)", ticksuffix="%",
                       gridcolor=theme["plot_grid"],
                       zeroline=True, zerolinecolor=theme["plot_grid"]),
            yaxis=dict(title="Density", gridcolor=theme["plot_grid"],
                       zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(
                xref="paper", yref="paper", x=0.01, y=0.97,
                showarrow=False, align="left",
                text=f"Mean: {mean_v:.3f}%  |  "
                     f"Std: {std_v:.3f}%  |  "
                     f"Median: {med_v:.3f}%",
                font=dict(size=10, color=theme["text_muted"]))])
        fig.update_layout(**layout)
        return fig


    # ── Box renderers ─────────────────────────────────────────────────────────
    @render_widget("box_STOXX")
    def box_STOXX():
        d   = market_data()
        lr  = live_log_rets().get("^STOXX")
        res = garch_results.get()
        fc  = res.get("^STOXX") if res else None
        return make_box_fig(d[0]["^STOXX"].dropna(), "STOXX 600", TICKER_COLORS[0], lr, fc)

    @render_widget("box_STOXX50E")
    def box_STOXX50E():
        d   = market_data()
        lr  = live_log_rets().get("^STOXX50E")
        res = garch_results.get()
        fc  = res.get("^STOXX50E") if res else None
        return make_box_fig(d[0]["^STOXX50E"].dropna(), "STOXX 50", TICKER_COLORS[1], lr, fc)

    @render_widget("box_AEX")
    def box_AEX():
        d   = market_data()
        lr  = live_log_rets().get("^AEX")
        res = garch_results.get()
        fc  = res.get("^AEX") if res else None
        return make_box_fig(d[0]["^AEX"].dropna(), "AEX", TICKER_COLORS[2], lr, fc)

    @render_widget("box_GSPC")
    def box_GSPC():
        d   = market_data()
        lr  = live_log_rets().get("^GSPC")
        res = garch_results.get()
        fc  = res.get("^GSPC") if res else None
        return make_box_fig(d[0]["^GSPC"].dropna(), "S&P 500", TICKER_COLORS[3], lr, fc)

    @render_widget("box_GDAXI")
    def box_GDAXI():
        d   = market_data()
        lr  = live_log_rets().get("^GDAXI")
        res = garch_results.get()
        fc  = res.get("^GDAXI") if res else None
        return make_box_fig(d[0]["^GDAXI"].dropna(), "DAX", TICKER_COLORS[4], lr, fc)

    @render_widget("box_NDX")
    def box_NDX():
        d   = market_data()
        lr  = live_log_rets().get("^NDX")
        res = garch_results.get()
        fc  = res.get("^NDX") if res else None
        return make_box_fig(d[0]["^NDX"].dropna(), "Nasdaq 100", TICKER_COLORS[5], lr, fc)

    # ── Dist renderers ────────────────────────────────────────────────────────
    @render_widget("dist_STOXX")
    def dist_STOXX():
        d = market_data(); lv = live_data()
        lr = compute_live_log_return(lv, ["^STOXX"])["^STOXX"] if lv else None
        return make_dist_fig(d[0]["^STOXX"].dropna(), "STOXX 600", TICKER_COLORS[0], lr)

    @render_widget("dist_STOXX50E")
    def dist_STOXX50E():
        d = market_data(); lv = live_data()
        lr = compute_live_log_return(lv, ["^STOXX50E"])["^STOXX50E"] if lv else None
        return make_dist_fig(d[0]["^STOXX50E"].dropna(), "STOXX 50", TICKER_COLORS[1], lr)

    @render_widget("dist_AEX")
    def dist_AEX():
        d = market_data(); lv = live_data()
        lr = compute_live_log_return(lv, ["^AEX"])["^AEX"] if lv else None
        return make_dist_fig(d[0]["^AEX"].dropna(), "AEX", TICKER_COLORS[2], lr)

    @render_widget("dist_GSPC")
    def dist_GSPC():
        d = market_data(); lv = live_data()
        lr = compute_live_log_return(lv, ["^GSPC"])["^GSPC"] if lv else None
        return make_dist_fig(d[0]["^GSPC"].dropna(), "S&P 500", TICKER_COLORS[3], lr)

    @render_widget("dist_GDAXI")
    def dist_GDAXI():
        d = market_data(); lv = live_data()
        lr = compute_live_log_return(lv, ["^GDAXI"])["^GDAXI"] if lv else None
        return make_dist_fig(d[0]["^GDAXI"].dropna(), "DAX", TICKER_COLORS[4], lr)

    @render_widget("dist_NDX")
    def dist_NDX():
        d = market_data(); lv = live_data()
        lr = compute_live_log_return(lv, ["^NDX"])["^NDX"] if lv else None
        return make_dist_fig(d[0]["^NDX"].dropna(), "Nasdaq 100", TICKER_COLORS[5], lr)


    # ── Cumulative line ───────────────────────────────────────────────────────
    @render_widget("line_cum")
    def line_cum():
        d  = market_data()
        lv = live_data()
        if d is None:
            return go.Figure()

        theme    = current_theme()
        fig      = go.Figure()
        fc_cum   = forecast_cum.get()
        live_cum = compute_live_cum_return(d[1], lv, TICKERS) if lv else {}
        llr      = live_log_rets()
        today    = datetime.now()

        for i, ticker in enumerate(TICKERS):
            label   = TICKER_LABELS[ticker]
            color   = TICKER_COLORS[i]
            visible = "legendonly" if ticker in CUM_HIDDEN else True
            grp     = f"grp_{ticker.replace('^','')}"

            # ── Historical line ───────────────────────────────────────────────
            col         = d[1][ticker].dropna()
            ret_aligned = d[0][ticker].dropna().reindex(col.index)
            fig.add_trace(go.Scatter(
                x=col.index.to_pydatetime(),
                y=col.values * 100,
                name=label,
                mode="lines",
                visible=visible,
                legendgroup=grp,
                line=dict(color=color, width=2),
                customdata=ret_aligned.values * 100,
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    "Cumulative: <b>%{y:.2f}%</b><br>"
                    "Day return: <b>%{customdata:.2f}%</b>"
                    f"<extra>{label}</extra>"
                ),
            ))

            # ── Live connector + dot ──────────────────────────────────────────
            lc       = live_cum.get(ticker)
            live_ret = llr.get(ticker) if llr else None

            if lc is not None:
                last_date = col.index[-1].to_pydatetime()
                last_val  = float(col.iloc[-1]) * 100

                fig.add_trace(go.Scatter(
                    x=[last_date, today],
                    y=[last_val, lc * 100],
                    mode="lines",
                    visible=visible,
                    legendgroup=grp,
                    showlegend=False,
                    line=dict(color=color, width=1.5, dash="dot"),
                    hoverinfo="skip",
                ))

                fig.add_trace(go.Scatter(
                    x=[today],
                    y=[lc * 100],
                    mode="markers+text",
                    visible=visible,
                    legendgroup=grp,
                    showlegend=False,
                    marker=dict(symbol="circle", size=10, color=color,
                                line=dict(color="white", width=1.5)),
                    text=[f"  {label}"],
                    textposition="middle right",
                    textfont=dict(size=9, color=color),
                    customdata=[[live_ret * 100 if live_ret is not None else float("nan")]],
                    hovertemplate=(
                        "%{x|%d %b %Y %H:%M}<br>"
                        "Cumulative: <b>%{y:.2f}%</b><br>"
                        "Day return: <b>%{customdata[0]:.2f}%</b>"
                        f"<extra>{label} — live</extra>"
                    ),
                ))

            # ── Forecast connector + diamond ──────────────────────────────────
            if fc_cum:
                fc = fc_cum.get(ticker)
                if fc is not None:
                    if lc is not None:
                        conn_x = [today,            fc["fc_date"]]
                        conn_y = [lc * 100,          fc["fc_cum"]]
                    else:
                        conn_x = [fc["anchor_date"], fc["fc_date"]]
                        conn_y = [fc["anchor_cum"],  fc["fc_cum"]]

                    fig.add_trace(go.Scatter(
                        x=conn_x, y=conn_y,
                        mode="lines",
                        visible=visible,
                        legendgroup=grp,
                        showlegend=False,
                        line=dict(color=color, width=1, dash="dash"),
                        hoverinfo="skip",
                    ))

                    fig.add_trace(go.Scatter(
                        x=[fc["fc_date"]],
                        y=[fc["fc_cum"]],
                        mode="markers",
                        visible=visible,
                        legendgroup=grp,
                        showlegend=False,
                        marker=dict(symbol="diamond", size=11, color=color,
                                    line=dict(color="white", width=1.5)),
                        hovertemplate=(
                            "%{x|%d %b %Y}<br>"
                            f"Forecast cumulative: <b>%{{y:.2f}}%</b><br>"
                            f"Forecast return: <b>{fc['fc_return']:+.3f}%</b><br>"
                            f"68% CI: [{fc['fc_68_low']:+.3f}%, "
                            f"{fc['fc_68_high']:+.3f}%]"
                            f"<extra>{label} — GARCH forecast</extra>"
                        ),
                    ))

        layout = plotly_layout(theme, height=380)
        layout.update(
            xaxis=dict(type="date", gridcolor=theme["plot_grid"], zeroline=False),
            yaxis=dict(title="Cumulative return (%)", ticksuffix="%",
                       gridcolor=theme["plot_grid"],
                       zeroline=True, zerolinecolor=theme["plot_grid"]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_layout(**layout)
        return fig


    # ── Server stat cards ─────────────────────────────────────────────────────
    def _stat_card(label, value, sub=""):
        return ui.tags.div({"class": "stat-card"},
            ui.tags.div(label, **{"class": "stat-label"}),
            ui.tags.div(value, **{"class": "stat-value"}),
            ui.tags.div(sub,   **{"class": "stat-sub"}) if sub
            else ui.tags.span())

    @render.ui
    def stat_cpu():
        interval = int(input.refresh_interval())
        if interval > 0: reactive.invalidate_later(interval)
        return _stat_card("CPU Usage",
            f"{psutil.cpu_percent(interval=0.2):.1f}%",
            f"{psutil.cpu_count(logical=True)} logical cores")

    @render.ui
    def stat_ram():
        interval = int(input.refresh_interval())
        if interval > 0: reactive.invalidate_later(interval)
        vm = psutil.virtual_memory()
        return _stat_card("RAM Usage", f"{vm.percent:.1f}%",
            f"{vm.used/1024**3:.1f} / {vm.total/1024**3:.1f} GB")

    @render.ui
    def stat_disk():
        interval = int(input.refresh_interval())
        if interval > 0: reactive.invalidate_later(interval)
        dk = psutil.disk_usage("/")
        return _stat_card("Disk Usage", f"{dk.percent:.1f}%",
            f"{dk.used/1024**3:.1f} / {dk.total/1024**3:.1f} GB")

    @render.ui
    def stat_uptime():
        interval = int(input.refresh_interval())
        if interval > 0: reactive.invalidate_later(interval)
        boot   = psutil.boot_time()
        uptime = time.time() - boot
        h, rem = divmod(int(uptime), 3600)
        m, _   = divmod(rem, 60)
        return _stat_card("Uptime", f"{h}h {m}m",
            f"Boot: {datetime.fromtimestamp(boot).strftime('%d %b %H:%M')}")

    @render.ui
    def proc_table():
        interval = int(input.refresh_interval())
        if interval > 0: reactive.invalidate_later(interval)
        SKIP      = {"system idle process", "idle"}
        proc_objs = list(psutil.process_iter(
            ["pid", "name", "memory_percent", "status"]))
        for p in proc_objs:
            try:   p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        time.sleep(0.3)
        procs_info = []
        cpu_count  = psutil.cpu_count(logical=True) or 1
        for p in proc_objs:
            try:
                name = p.info["name"] or "—"
                if name.lower() in SKIP: continue
                procs_info.append({
                    "pid":    p.info["pid"],
                    "name":   name,
                    "cpu":    p.cpu_percent(interval=None) / cpu_count,
                    "mem":    p.info["memory_percent"] or 0.0,
                    "status": p.info["status"] or "—",
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs_info.sort(key=lambda x: (x["cpu"], x["mem"]), reverse=True)
        page_size = 15
        total     = len(procs_info)
        n_pages   = max(1, (total + page_size - 1) // page_size)
        try:    current_page = int(input.proc_page())
        except: current_page = 1
        current_page = max(1, min(current_page, n_pages))
        chunk = procs_info[
            (current_page - 1) * page_size : current_page * page_size
        ]
        rows = [
            ui.tags.tr(
                ui.tags.td(str(p["pid"])),
                ui.tags.td(p["name"]),
                ui.tags.td(f'{p["cpu"]:.1f}%'),
                ui.tags.td(f'{p["mem"]:.1f}%'),
                ui.tags.td(p["status"]))
            for p in chunk
        ]
        pagination = ui.tags.div(
            {"style": "display:flex;gap:0.5rem;align-items:center;"
                      "margin-top:1rem;font-size:0.85rem;"},
            ui.tags.button("← Prev", id="proc_prev",
                onclick="updatePage(-1)",
                disabled=current_page <= 1,
                style="padding:0.3rem 0.75rem;cursor:pointer;"
                      "border-radius:5px;border:1px solid #ccc;"),
            ui.tags.span(f"Page {current_page} of {n_pages}"),
            ui.tags.button("Next →", id="proc_next",
                onclick="updatePage(1)",
                disabled=current_page >= n_pages,
                style="padding:0.3rem 0.75rem;cursor:pointer;"
                      "border-radius:5px;border:1px solid #ccc;"),
            ui.tags.span(f"({total} processes)",
                style="margin-left:0.5rem;color:#888;"),
        )
        return ui.tags.div(
            ui.tags.table({"class": "proc-table"},
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("PID"), ui.tags.th("Name"),
                    ui.tags.th("CPU %"), ui.tags.th("MEM %"),
                    ui.tags.th("Status"))),
                ui.tags.tbody(*rows)),
            pagination)


app = App(app_ui, server)