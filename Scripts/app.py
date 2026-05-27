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

from models import (
    MODEL_REGISTRY,
    MODEL_IDS,
    fit_all_models,
)

# Module-level lock — prevents two sessions from fitting simultaneously
_fit_lock    = threading.Lock()
_fit_started = threading.Event()
_current_fitting_model = [""]   # mutable so background thread can write to it


# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.pkl")

def cache_save(key: str, data) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump({"date": datetime.now().date(), "data": data}, f)

def cache_load(key: str):
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


# ── GARCH / model fitting ─────────────────────────────────────────────────────
def compute_all_models(tickers: list[str], years_back: int = 15) -> dict:
    # Check if all models already cached
    all_cached = {}
    for mid in MODEL_IDS:
        cached = cache_load(f"models_{mid}")
        if cached is not None:
            all_cached[mid] = cached

    if len(all_cached) == len(MODEL_IDS):
        print("[models] All models loaded from cache.")
        return _reformat_results(all_cached)

    # Only one thread fits at a time
    if not _fit_lock.acquire(blocking=False):
        print("[models] Fitting already in progress, waiting...")
        _fit_lock.acquire(blocking=True)
        _fit_lock.release()
        # Re-load whatever was cached by the other thread
        all_cached = {}
        for mid in MODEL_IDS:
            cached = cache_load(f"models_{mid}")
            if cached is not None:
                all_cached[mid] = cached
        return _reformat_results(all_cached)

    try:
        print("[models] Cache miss — fitting models...")
        end   = datetime.now()
        start = end.replace(year=end.year - years_back)

        # Fetch price data once
        price_data = {}
        for ticker in tickers:
            try:
                prices = yf.Ticker(ticker).history(
                    start=start, end=end, interval="1d", auto_adjust=True
                )["Close"]
                if hasattr(prices.index, "tz") and prices.index.tz is not None:
                    prices.index = prices.index.tz_localize(None)
                log_ret     = np.log(prices.dropna() / prices.dropna().shift(1)).dropna()
                log_ret_pct = log_ret * 100
                price_data[ticker] = (log_ret_pct, log_ret)
                print(f"  [data] {ticker} loaded — {len(log_ret)} obs")
            except Exception as e:
                print(f"  [data] {ticker} FAILED: {e}")

        results_by_model = {}

        # Fit fast models first (gjr_skewt through garch_m), CAViaR last
        FAST_MODELS = ["gjr_skewt", "gjr_t", "gjr_normal", "egarch_skewt", "garch_m"]
        SLOW_MODELS = ["caviar"]

        for mid in FAST_MODELS + SLOW_MODELS:
            # Skip if already cached
            cached = cache_load(f"models_{mid}")
            if cached is not None:
                results_by_model[mid] = cached
                all_cached[mid] = cached
                continue
            
            _current_fitting_model[0] = mid
            model_results = {}
            for ticker in tickers:
                if ticker not in price_data:
                    continue
                log_ret_pct, log_ret = price_data[ticker]
                from models import fit_model
                res = fit_model(mid, log_ret_pct, log_ret)
                model_results[ticker] = res
                if res.get("status") == "ok":
                    print(f"  [{mid}] {ticker} — "
                          f"P(↑)={res['p_positive']*100:.1f}%  "
                          f"σ={res['vol_fc']*100:.3f}%/day")
                else:
                    print(f"  [{mid}] {ticker} FAILED: {res.get('error','?')}")

            results_by_model[mid] = model_results
            # Cache immediately so the page can use it
            cache_save(f"models_{mid}", model_results)
            print(f"  [{mid}] cached.")

        _current_fitting_model[0] = ""
        return _reformat_results(results_by_model)

    finally:
        _fit_lock.release()

def _reformat_results(results_by_model: dict) -> dict:
    """
    Convert {model_id: {ticker: result}}
    to      {ticker: {model_id: result}}
    """
    out = {}
    for mid, ticker_dict in results_by_model.items():
        for ticker, res in ticker_dict.items():
            if ticker not in out:
                out[ticker] = {}
            out[ticker][mid] = res
    return out


def get_default_model_result(all_results: dict, ticker: str) -> dict | None:
    """
    Returns the gjr_skewt result for a ticker,
    falling back to the first available model.
    Used wherever the old single-model result was used
    (box plots, cumulative chart forecast dot, snapshot).
    """
    ticker_results = all_results.get(ticker, {})
    return (
        ticker_results.get("gjr_skewt")
        or next((v for v in ticker_results.values() if v.get("status") == "ok"), None)
    )

# ── Market data ───────────────────────────────────────────────────────────────
def get_index_returns(
    tickers: list[str],
    years_back: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cached = cache_load("market_returns")
    if cached is not None:
        return cached

    print("[data] Cache miss — downloading market data...")
    end_date   = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date.replace(year=end_date.year - years_back)
    returns = pd.DataFrame()
    cum_returns = pd.DataFrame()

    for ticker in tickers:
        try:
            prices = yf.Ticker(ticker).history(
                start=start_date, end=end_date, interval="1d", auto_adjust=True
            )["Close"]
            if hasattr(prices.index, "tz") and prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            if prices.dropna().empty:
                returns[ticker] = np.nan
                cum_returns[ticker] = np.nan
                continue
            log_ret = np.log(prices / prices.shift(1))
            returns[ticker]     = log_ret
            cum_returns[ticker] = np.exp(log_ret.cumsum()) - 1
        except Exception as e:
            print(f"[warning] Failed {ticker}: {e}")
            returns[ticker] = np.nan
            cum_returns[ticker] = np.nan

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
            if live_px is None:
                live_ret[ticker] = None
                continue
            hist   = yf.Ticker(ticker).history(period="5d", interval="1d")
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                live_ret[ticker] = None
                continue
            live_ret[ticker] = float(np.log(live_px / float(closes.iloc[-2])))
        except Exception:
            live_ret[ticker] = None
    return live_ret


def compute_live_cum_return(
    cum_returns: pd.DataFrame,
    live_prices: dict,
    tickers: list[str],
) -> dict:
    live_cum = {}
    for ticker in tickers:
        try:
            live_px = live_prices.get(ticker)
            if live_px is None:
                live_cum[ticker] = None
                continue
            last_cum   = cum_returns[ticker].dropna().iloc[-1]
            last_close = float(
                yf.Ticker(ticker).history(
                    period="5d", interval="1d"
                )["Close"].dropna().iloc[-1]
            )
            live_cum[ticker] = live_px / (last_close / (1 + last_cum)) - 1
        except Exception:
            live_cum[ticker] = None
    return live_cum


def compute_forecast_cumulative(
    cum_returns: pd.DataFrame,
    all_results: dict,
    tickers:     list[str],
) -> dict:
    """
    Anchors the gjr_skewt forecast to the last known cumulative return level.
    Cached separately — invalidated when models cache is refreshed.
    """
    cached = cache_load("forecast_cumulative")
    if cached is not None:
        return cached

    result = {}
    for ticker in tickers:
        fc = get_default_model_result(all_results, ticker)
        if fc is None or fc.get("status") != "ok":
            continue
        try:
            last_cum  = float(cum_returns[ticker].dropna().iloc[-1])
            last_date = cum_returns[ticker].dropna().index[-1]
            fc_return = fc["mean_fc"]
            fc_cum    = (1 + last_cum) * (1 + fc_return) - 1
            result[ticker] = {
                "anchor_date": last_date,
                "anchor_cum":  last_cum * 100,
                "fc_date":     fc["next_date"],
                "fc_cum":      fc_cum * 100,
                "fc_return":   fc_return * 100,
                "fc_68_low":   fc["lower_68"] * 100,
                "fc_68_high":  fc["upper_68"] * 100,
            }
        except Exception as e:
            print(f"[forecast_cum] {ticker} failed: {e}")

    cache_save("forecast_cumulative", result)
    print("[forecast_cum] Cached forecast cumulative levels.")
    return result


# ── Previous-day forecast snapshot ───────────────────────────────────────────
def save_forecast_snapshot(
    all_results:  dict,
    cum_returns:  pd.DataFrame,
) -> None:
    """
    Saves the gjr_skewt forecast for each ticker so tomorrow
    we can compare it to the actual close.
    Only writes once per calendar day.
    """
    key  = "forecast_snapshot"
    path = _cache_path(key)
    today = datetime.now().date()

    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                existing = pickle.load(f)
            if existing.get("snapshot_date") == today:
                return
        except Exception:
            pass

    snapshot = {"snapshot_date": today, "tickers": {}}
    for ticker in all_results:
        fc = get_default_model_result(all_results, ticker)
        if fc is None or fc.get("status") != "ok":
            continue
        try:
            last_cum = float(cum_returns[ticker].dropna().iloc[-1])
            fc_cum   = (1 + last_cum) * (1 + fc["mean_fc"]) - 1
            snapshot["tickers"][ticker] = {
                "fc_date":    fc["next_date"],
                "fc_cum":     fc_cum * 100,
                "fc_return":  fc["mean_fc"] * 100,
                "fc_68_low":  fc["lower_68"] * 100,
                "fc_68_high": fc["upper_68"] * 100,
                "anchor_cum": last_cum * 100,
            }
        except Exception as e:
            print(f"[snapshot] {ticker} failed: {e}")

    with open(path, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"[snapshot] Saved forecast snapshot for {today}.")


def load_prev_forecast_snapshot(
    cum_returns: pd.DataFrame,
) -> dict | None:
    path = _cache_path("forecast_snapshot")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            snap = pickle.load(f)
    except Exception as e:
        print(f"[snapshot] Load error: {e}")
        return None

    result = {}
    for ticker, fc in snap.get("tickers", {}).items():
        try:
            col     = cum_returns[ticker].dropna()
            fc_date = pd.Timestamp(fc["fc_date"])

            candidates = col.index[col.index >= fc_date]
            if candidates.empty:
                continue
            actual_date = candidates[0]
            actual_cum  = float(col.loc[actual_date]) * 100

            log_ret_col = pd.Series(
                np.log(col.values / np.roll(col.values, 1)),
                index=col.index,
            )
            actual_ret = float(log_ret_col.loc[actual_date]) * 100

            result[ticker] = {
                "fc_date":       actual_date,
                "fc_cum":        fc["fc_cum"],
                "actual_cum":    actual_cum,
                "fc_return":     fc["fc_return"],
                "actual_return": actual_ret,
                "error":         actual_cum - fc["fc_cum"],
                "fc_68_low":     fc["fc_68_low"],
                "fc_68_high":    fc["fc_68_high"],
                "anchor_cum":    fc["anchor_cum"],
            }
        except Exception as e:
            print(f"[snapshot] {ticker} resolve error: {e}")

    return result if result else None

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS       = ["^STOXX", "^STOXX50E", "^AEX", "^GSPC", "^GDAXI", "^NDX"]
TICKER_LABELS = {
    "^STOXX": "STOXX 600", "^STOXX50E": "STOXX 50", "^AEX": "AEX",
    "^GSPC": "S&P 500", "^GDAXI": "DAX", "^NDX": "Nasdaq 100",
}
TICKER_COLORS = ["#4A90D9", "#E8734A", "#4CAF82", "#A855F7", "#F59E0B", "#EC4899"]
CUM_HIDDEN    = {"^STOXX50E", "^NDX", "^GDAXI", "^GSPC"}
REFRESH_CHOICES = {"0": "Off", "5": "5s", "10": "10s", "30": "30s", "60": "60s"}

# Model selector labels for the UI tabs
MODEL_TAB_LABELS = {
    "gjr_skewt":    "GJR + Skew-t",
    "gjr_t":        "GJR + t",
    "gjr_normal":   "GJR + Normal",
    "egarch_skewt": "EGARCH + Skew-t",
    "garch_m":      "GARCH-M",
    "caviar":       "CAViaR",
}

# Short econometric descriptions shown in the info card
MODEL_DESCRIPTIONS = {
    "gjr_skewt": (
        "GJR-GARCH(1,1) with Hansen's skewed Student-t. "
        "The GJR γ term allows the conditional variance to respond "
        "asymmetrically to signed innovations: negative residuals (ε < 0) "
        "receive weight α + γ versus α for positive residuals, capturing "
        "the empirical leverage effect. The skewed-t adds a location-scale "
        "skewness parameter λ, shifting probability mass toward the left tail. "
        "This is the most complete single-equation specification for equity returns."
    ),
    "gjr_t": (
        "GJR-GARCH(1,1) with symmetric Student-t innovations. "
        "Retains the asymmetric variance response (leverage effect) via the γ term "
        "but imposes symmetry on the conditional return distribution. "
        "The degrees-of-freedom parameter ν controls tail thickness — "
        "lower ν implies heavier tails. "
        "Useful as a benchmark: differences vs Skew-t isolate the contribution "
        "of distributional skewness to downside probabilities."
    ),
    "gjr_normal": (
        "GJR-GARCH(1,1) with Gaussian innovations. "
        "Preserves asymmetric volatility dynamics but assumes normally distributed "
        "standardised residuals. Under the CLT this is asymptotically consistent, "
        "but in finite samples the thin tails of the normal severely underestimate "
        "the probability of extreme returns. "
        "Serves as a lower bound on tail risk — any excess over this baseline "
        "is attributable to fat-tail and skewness effects."
    ),
    "egarch_skewt": (
        "EGARCH(1,1) of Nelson (1991) with Hansen's skewed Student-t. "
        "Models log conditional variance, guaranteeing positivity without "
        "parameter constraints. The asymmetry term captures both the sign "
        "and magnitude of lagged standardised innovations, unlike GJR which "
        "only conditions on the sign. "
        "Differences vs GJR-Skewt reveal whether the leverage effect is "
        "better described by a threshold (GJR) or a continuous function (EGARCH)."
    ),
    "garch_m": (
        "GARCH-in-Mean(1,1) with Hansen's skewed Student-t, following "
        "Engle, Lilien & Robins (1987). The conditional mean is augmented: "
        "μ_t = c + δ·σ_t, so the risk premium is time-varying and proportional "
        "to current volatility. A negative δ (risk-off regime) directly lowers "
        "the mean forecast during high-volatility periods, addressing the "
        "positive-mean bias of constant-mean GARCH models. "
        "The mean forecast here is endogenous to the variance state."
    ),
    "caviar": (
        "Conditional Autoregressive Value at Risk, SAV specification "
        "of Engle & Manganelli (2004). Directly models the τ-quantile "
        "of the return distribution as an AR(1) process in lagged quantile "
        "and lagged absolute return: Q_t(τ) = β₀ + β₁·Q_{t-1}(τ) + β₂·|r_{t-1}|. "
        "No parametric distribution is assumed — probabilities are derived "
        "from a piecewise-linear interpolation across fitted quantiles "
        "(τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}). "
        "This makes CAViaR the most assumption-free model in the set and "
        "particularly reliable for tail probability estimation."
    ),
}

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
    .model-selector {{ display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1.25rem; }}
    .model-tab {{ padding:0.4rem 0.9rem; border-radius:6px; border:1px solid {c['border']}; background:{c['card_bg']}; color:{c['text_muted']}; font-size:0.8rem; font-weight:600; cursor:pointer; transition:all 0.15s; white-space:nowrap; }}
    .model-tab:hover {{ border-color:{c['primary']}; color:{c['primary']}; }}
    .model-tab.active {{ background:{c['primary']}; border-color:{c['primary']}; color:#fff; }}
    .model-desc-card {{ background:{c['card_bg']}; border:1px solid {c['border']}; border-left: 3px solid {c['primary']}; border-radius:8px; padding:0.9rem 1.25rem; margin-bottom:1.25rem; font-size:0.82rem; color:{c['text_muted']}; line-height:1.6; }}
    .model-desc-card strong {{ color:{c['text_main']}; }}
    .comparison-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
    .comparison-table th {{ text-align:left; padding:0.4rem 0.6rem; border-bottom:2px solid {c['border']}; color:{c['text_muted']}; font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; }}
    .comparison-table td {{ padding:0.4rem 0.6rem; border-bottom:1px solid {c['border']}; color:{c['text_main']}; }}
    .comparison-table tr:last-child td {{ border-bottom:none; }}
    .comparison-table tr.active-model td {{ background:{c['page_bg']}; font-weight:600; }}
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
    # Same tab buttons as Directional — share input.selected_model
    tabs = [
        ui.tags.button(
            label,
            **{
                "class":   "model-tab" + (" active" if mid == "gjr_skewt" else ""),
                "id":      f"model-tab-{mid}",
                "onclick": f"selectModel('{mid}')",
            }
        )
        for mid, label in MODEL_TAB_LABELS.items()
    ]

    return ui.tags.div(
        {"class": "page", "id": "page-analysis"},
        # Cumulative pop-out modal (unchanged) ...
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
        # Page header
        ui.tags.div({"class": "page-header"},
            ui.tags.h1("Analysis"),
            ui.tags.p("Index return distributions, cumulative performance and model forecasts.")
        ),
        # NEW: shared loading/progress banner above model tabs
        ui.output_ui("model_loading_banner_analysis"),
        # Model selector (shared with Directional)
        ui.tags.div({"class": "model-selector"}, *tabs),
        # Box / dist grid ...
        ui.tags.div(
            {"class": "plot-row"},
            *[
                ui.tags.div(
                    {"class": "plot-card"},
                    ui.tags.h3(f"{TICKER_LABELS[t]} — Return Distribution"),
                    ui.tags.p("Daily log returns"),
                    ui.tags.button("↻", **{
                        "class":   "dist-icon-btn",
                        "id":      f"toggle-btn-{t.replace('^','')}",
                        "onclick": f"toggleChart('{t.replace('^','')}')",
                        "title":   "Toggle distribution view",
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
        # Cumulative line card (unchanged)
        ui.tags.div({"class": "plot-card"},
            ui.tags.h3("Cumulative Returns"),
            ui.tags.button("⛶", **{
                "id":      "btn-popout-cum",
                "onclick": "popoutCumulative()",
                "title":   "Pop out chart",
                "style":   "position:absolute;top:0.85rem;right:0.85rem;"
                           "background:none;border:none;cursor:pointer;"
                           "font-size:1.1rem;opacity:0.5;transition:opacity 0.15s;",
                "onmouseover": "this.style.opacity='1'",
                "onmouseout":  "this.style.opacity='0.5'",
            }),
            output_widget("line_cum")),
    )



def make_directional_page():
    # Build model selector tabs in the UI statically
    tabs = [
        ui.tags.button(
            label,
            **{
                "class":   "model-tab" + (" active" if mid == "gjr_skewt" else ""),
                "id":      f"model-tab-{mid}",
                "onclick": f"selectModel('{mid}')",
            }
        )
        for mid, label in MODEL_TAB_LABELS.items()
    ]
    return ui.tags.div(
        {"class": "page", "id": "page-directional"},
        ui.tags.div({"class": "page-header"},
            ui.tags.h1("Directional Probabilities"),
            ui.tags.p(
                "Next-day return direction probabilities and tail risk estimates "
                "across six econometric specifications. Select a model to update "
                "all cards and the comparison strip."
            )),
        # Hidden input that Shiny reads
        ui.tags.input(
            id="selected_model",
            type="text",
            value="gjr_skewt",
            style="display:none;position:absolute;",
        ),
        # NEW: loading / progress banner above tabs
        ui.output_ui("model_loading_banner_directional"),
        # Model selector tabs
        ui.tags.div({"class": "model-selector"}, *tabs),
        # Description card — rendered server-side
        ui.output_ui("model_desc_card"),
        # Main probability content
        ui.output_ui("garch_prob_content"),
        # Comparison strip across all models
        ui.tags.div({"class": "plot-card", "style": "margin-top:1.5rem;"},
            ui.tags.h3("Model Comparison"),
            ui.tags.p(
                "P(↑), P(↓) and P(<−1%) for all models simultaneously. "
                "Highlights how distributional and mean-equation assumptions "
                "shift tail probabilities."
            ),
            ui.output_ui("model_comparison_table")),
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
                        {"class": "theme-option selected", "id": "opt-light",
                         "onclick": "setTheme('light')"},
                        ui.tags.input(type="radio", name="theme", value="light", checked=True),
                        "☀️  Light mode"),
                    ui.tags.label(
                        {"class": "theme-option", "id": "opt-dark",
                         "onclick": "setTheme('dark')"},
                        ui.tags.input(type="radio", name="theme", value="dark"),
                        "🌙  Dark mode"),
                ),
            ),
            ui.tags.div({"class": "settings-section"},
                ui.tags.h2("Cache"),
                ui.tags.p(
                    "Cached files are stored in .cache/ next to app.py and expire at midnight.",
                    **{"class": "section-desc"}),
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

    # ── Theme + sidebar + navigation + model selector ─────────────────────────
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

        // ── Model selector ────────────────────────────────────────────────────
        function selectModel(mid) {{
            // Update tab highlight
            document.querySelectorAll('.model-tab').forEach(t => {{
                t.classList.toggle('active', t.id === 'model-tab-' + mid);
            }});
            // Push value into hidden Shiny input
            const el = document.getElementById('selected_model');
            el.value = mid;
            el.dispatchEvent(new Event('change'));
        }}
    """),

    # ── Popout cumulative chart ───────────────────────────────────────────────
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
                target.on('plotly_relayout', function(ed) {
                    if (!ed['xaxis.range[0]']) return;
                    Plotly.relayout(plotDiv, {
                        'xaxis.range[0]': ed['xaxis.range[0]'],
                        'xaxis.range[1]': ed['xaxis.range[1]'],
                    });
                });
                target.on('plotly_restyle', function(ed) {
                    Plotly.restyle(plotDiv, ed[0], ed[1]);
                });
            });
        }

        function closeCumModal() {
            const overlay   = document.getElementById('cumModalOverlay');
            const target    = document.getElementById('cum-modal-plot');
            const container = document.getElementById('line_cum');
            const plotDiv   = container ? container.querySelector('.js-plotly-plot') : null;
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
                if (Object.keys(update).length) Plotly.relayout(plotDiv, update);
                if (target.data && plotDiv.data) {
                    const visUpdate = target.data.map(function(t) { return t.visible; });
                    Plotly.restyle(plotDiv, {visible: visUpdate});
                }
            }
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

    # ── Market data ───────────────────────────────────────────────────────────
    r, c = get_index_returns(TICKERS, years_back=4)
    r = r.dropna()
    c = c.loc[r.index]
    market_data      = reactive.Value((r, c))
    _cum_returns_ref = c
    _prev_snap       = load_prev_forecast_snapshot(c)
    prev_snapshot    = reactive.Value(_prev_snap)

    _initial_prices = get_live_prices(TICKERS)
    live_data       = reactive.Value(_initial_prices)
    live_log_rets   = reactive.Value(
        compute_live_log_return(_initial_prices, TICKERS)
    )

    # all_results: {ticker: {model_id: ModelResult}}
    all_model_results = reactive.Value(None)

    # ── Background thread — fit all models ────────────────────────────────────
    _models_ready = threading.Event()
    _models_data  = {}

    _models_partial = {}   # accumulates results as models complete

    def _run_models():
        print("[models] Background thread started...")
        res = compute_all_models(TICKERS, years_back=15)
        _models_data.update(res)
        save_forecast_snapshot(res, _cum_returns_ref)
        _models_ready.set()
        print("[models] All models done — including CAViaR.")

    threading.Thread(target=_run_models, daemon=True).start()

    _last_ok_count = [0]  # mutable container so nonlocal isn't needed

    @reactive.Effect
    def _poll_models():
        if _models_ready.is_set():
            if _models_data:
                all_model_results.set(dict(_models_data))
                print("[poll] Done — all models loaded.")
            return

        # Still fitting — check cache for new results
        partial = {}
        ok_count = 0
        for mid in MODEL_IDS:
            path = _cache_path(f"models_{mid}")
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                    if obj["date"] == datetime.now().date():
                        cached = obj["data"]
                        for ticker, res in cached.items():
                            if ticker not in partial:
                                partial[ticker] = {}
                            partial[ticker][mid] = res
                            if res.get("status") == "ok":
                                ok_count += 1
                except Exception:
                    pass

        # Only update reactive value if count changed — avoids re-renders
        if ok_count != _last_ok_count[0]:
            _last_ok_count[0] = ok_count
            if partial:
                all_model_results.set(partial)
                print(f"[poll] {ok_count} ok results ready")

        reactive.invalidate_later(3)

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

    def selected_model_id() -> str:
        return input.selected_model() or "gjr_skewt"

    # ── Forecast cumulative ───────────────────────────────────────────────────
    forecast_cum = reactive.Value(None)

    @reactive.Effect
    def _compute_forecast_cum():
        res = all_model_results.get()
        if res is None:
            return
        if forecast_cum.get() is not None:
            return
        d  = market_data()
        fc = compute_forecast_cumulative(d[1], res, TICKERS)
        forecast_cum.set(fc)

    # ── Shared banner logic ───────────────────────────────────────────────────
    def _build_loading_banner():
        c   = current_theme()
        res = all_model_results.get()   # reactive dependency — fires on every poll update

        if _models_ready.is_set():
            return ui.tags.div()

        done_models = []
        if res is not None:
            done_models = [
                m for m in MODEL_IDS
                if any(
                    res.get(t, {}).get(m, {}).get("status") == "ok"
                    for t in TICKERS
                )
            ]

        fitting = _current_fitting_model[0]
        label   = MODEL_TAB_LABELS.get(fitting, fitting)
        msg     = f"Working on: {label}…" if fitting else "Initialising models…"

        pills = [
            ui.tags.span(
                MODEL_TAB_LABELS[m],
                style=(
                    f"display:inline-block;padding:0.2rem 0.6rem;"
                    f"border-radius:4px;font-size:0.75rem;font-weight:600;margin:0.2rem;"
                    f"background:{'rgba(74,144,217,0.15)' if m in done_models else c['border']};"
                    f"color:{'#4A90D9' if m in done_models else c['text_muted']};"
                )
            )
            for m in MODEL_IDS
        ]

        return ui.tags.div(
            {"class": "plot-card", "style": "margin-bottom:1.0rem;"},
            ui.tags.div(
                {"style": "display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;"},
                ui.tags.div({"class": "garch-spinner"}),
                ui.tags.span(msg, style=f"color:{c['text_main']};font-weight:500;font-size:0.9rem;"),
            ),
            ui.tags.div(
                {"style": "display:flex;flex-wrap:wrap;gap:0.25rem;margin-bottom:0.5rem;"},
                *pills,
            ),
            ui.tags.p(
                "The page updates automatically as each model completes.",
                style=f"font-size:0.78rem;color:{c['text_muted']};margin:0;",
            ),
        )

    @render.ui
    def model_loading_banner_analysis():
        return _build_loading_banner()

    @render.ui
    def model_loading_banner_directional():
        return _build_loading_banner()

    # ── Model description card ────────────────────────────────────────────────
    @render.ui
    def model_desc_card():
        _  = input.active_theme()
        mid  = selected_model_id()
        desc = MODEL_DESCRIPTIONS.get(mid, "")
        name = MODEL_TAB_LABELS.get(mid, mid)
        return ui.tags.div(
            {"class": "model-desc-card"},
            ui.tags.strong(f"{name}  — "),
            desc,
        )

    # ── Model comparison table ────────────────────────────────────────────────
    @render.ui
    def model_comparison_table():
        _   = input.active_theme()
        c   = current_theme()
        res = all_model_results.get()
        mid = selected_model_id()

        if res is None:
            return ui.tags.div({"class": "garch-loading"},
                ui.tags.div({"class": "garch-spinner"}),
                ui.tags.span("Fitting models…"))

        # One row per model, columns = metrics averaged across tickers
        header = ui.tags.thead(ui.tags.tr(
            ui.tags.th("Model"),
            *[ui.tags.th(TICKER_LABELS[t]) for t in TICKERS],
            ui.tags.th("Avg P(↑)"),
            ui.tags.th("Avg P(<−1%)"),
            ui.tags.th("Avg σ/day"),
        ))

        rows = []
        for m_id, m_label in MODEL_TAB_LABELS.items():
            p_ups, p_dn1s, vols = [], [], []
            ticker_cells = []
            for ticker in TICKERS:
                f = res.get(ticker, {}).get(m_id, {})
                if f.get("status") == "ok":
                    pu  = f["p_positive"] * 100
                    pd1 = f["p_down_1pct"] * 100
                    v   = f["vol_fc"] * 100
                    p_ups.append(pu)
                    p_dn1s.append(pd1)
                    vols.append(v)
                    col = c["positive"] if pu >= 50 else c["negative"]
                    ticker_cells.append(
                        ui.tags.td(
                            ui.tags.span(
                                f"{pu:.1f}%",
                                style=f"font-weight:600;color:{col};"
                            )
                        )
                    )
                else:
                    ticker_cells.append(ui.tags.td("—"))

            avg_up  = f"{np.mean(p_ups):.1f}%"  if p_ups  else "—"
            avg_dn1 = f"{np.mean(p_dn1s):.1f}%" if p_dn1s else "—"
            avg_vol = f"{np.mean(vols):.3f}%"   if vols   else "—"

            is_active = (m_id == mid)
            rows.append(ui.tags.tr(
                {"class": "active-model" if is_active else ""},
                ui.tags.td(
                    ui.tags.span(
                        m_label,
                        style=f"font-weight:{'700' if is_active else '400'};"
                              f"color:{c['primary'] if is_active else c['text_main']};"
                    )
                ),
                *ticker_cells,
                ui.tags.td(avg_up),
                ui.tags.td(avg_dn1),
                ui.tags.td(avg_vol),
            ))

        return ui.tags.div(
            {"style": "overflow-x:auto;"},
            ui.tags.table(
                {"class": "comparison-table"},
                header,
                ui.tags.tbody(*rows),
            )
        )

    @render.ui
    def garch_prob_content():
        c   = current_theme()
        res = all_model_results.get()
        mid = selected_model_id()

        # Banner is now handled by model_loading_banner — just return empty div
        if res is None:
            return ui.tags.div()

        def get_f(ticker):
            return res.get(ticker, {}).get(mid, {})

        sample = next(
            (get_f(t) for t in TICKERS if get_f(t).get("status") == "ok"),
            None
        )
        cache_ts = (
            sample["computed_at"].strftime("%d %b %Y %H:%M")
            if sample else "—"
        )

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
            ui.tags.th("μ forecast"),
            ui.tags.th("P(↑)"),
            ui.tags.th("P(↓)"),
            ui.tags.th("P(>+1%)"),
            ui.tags.th("P(<−1%)"),
            ui.tags.th("σ ann."),
            ui.tags.th("Persist."),
            ui.tags.th("Half-life"),
            ui.tags.th("Signal"),
        ))

        rows = []
        for ticker in TICKERS:
            label = TICKER_LABELS[ticker]
            color = TICKER_COLORS[TICKERS.index(ticker)]
            f     = get_f(ticker)

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

            persist   = f.get("persist")
            half_life = f.get("half_life")
            hl_str = (
                f"{half_life:.1f}d"
                if half_life is not None and half_life < 500
                else ("∞" if half_life is not None else "—")
            )
            persist_str = f"{persist:.4f}" if persist is not None else "—"

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
                ui.tags.td(persist_str),
                ui.tags.td(hl_str),
                ui.tags.td(ui.tags.span(signal,
                    **{"class": f"badge {sig_cls}"})),
            ))

        table_card = ui.tags.div({"class": "plot-card"},
            ui.tags.h3("Directional Probability Summary"),
            ui.tags.p(
                f"{MODEL_TAB_LABELS.get(mid, mid)}  |  "
                f"15-year estimation window  |  "
                f"Computed: {cache_ts}"
            ),
            ui.tags.table({"class": "prob-table"},
                header, ui.tags.tbody(*rows)))

        # ── Per-index gauge cards ─────────────────────────────────────────────
        gauge_cards = []
        for ticker in TICKERS:
            label = TICKER_LABELS[ticker]
            color = TICKER_COLORS[TICKERS.index(ticker)]
            f     = get_f(ticker)
            if f.get("status") != "ok":
                continue

            p_up    = f["p_positive"]  * 100
            p_dn    = f["p_negative"]  * 100
            sig_cls = "bull" if p_up >= 50 else "bear"
            signal  = "BULL" if p_up >= 50 else "BEAR"

            extra_pills = []
            if f.get("q05") is not None:
                extra_pills += [
                    ui.tags.div(f"Q5%: {f['q05']*100:+.3f}%",  **{"class": "prob-pill tail"}),
                    ui.tags.div(f"Q10%: {f['q10']*100:+.3f}%", **{"class": "prob-pill tail"}),
                ]
            else:
                nu_val = f.get("nu")
                if nu_val is not None:
                    extra_pills.append(ui.tags.div(f"ν = {nu_val:.1f}", **{"class": "prob-pill tail"}))
                lam_val = f.get("lam")
                if lam_val is not None:
                    extra_pills.append(ui.tags.div(f"λ = {lam_val:.3f}", **{"class": "prob-pill tail"}))

            persist   = f.get("persist")
            half_life = f.get("half_life")
            if persist is not None:
                hl_str = (
                    f"{half_life:.1f}d"
                    if half_life is not None and half_life < 500
                    else "∞"
                )
                footer_str = (
                    f"σ={f['vol_fc']*100:.3f}%/day  |  "
                    f"persist={persist:.4f}  |  "
                    f"half-life={hl_str}"
                )
            else:
                footer_str = f"σ={f['vol_fc']*100:.3f}%/day  |  SAV-CAViaR quantile model"

            gauge_cards.append(ui.tags.div({"class": "forecast-card"},
                ui.tags.div({"class": "fc-header"},
                    ui.tags.span(label, **{"class": "fc-label", "style": f"color:{color};"}),
                    ui.tags.span(signal, **{"class": f"badge {sig_cls}"})),
                ui.tags.div(
                    {"style": "display:flex;gap:0.5rem;margin:0.75rem 0;"},
                    ui.tags.div(
                        {"style": "flex:1;background:rgba(22,163,74,0.1);border-radius:8px;padding:0.75rem;text-align:center;"},
                        ui.tags.div("P(↑)", style=f"font-size:0.7rem;color:{c['text_muted']};"),
                        ui.tags.div(f"{p_up:.1f}%", style=f"font-size:1.4rem;font-weight:700;color:{c['positive']};")),
                    ui.tags.div(
                        {"style": "flex:1;background:rgba(220,38,38,0.1);border-radius:8px;padding:0.75rem;text-align:center;"},
                        ui.tags.div("P(↓)", style=f"font-size:0.7rem;color:{c['text_muted']};"),
                        ui.tags.div(f"{p_dn:.1f}%", style=f"font-size:1.4rem;font-weight:700;color:{c['negative']};"))),
                ui.tags.div({"class": "prob-row"},
                    ui.tags.div(f">+1%: {f['p_up_1pct']*100:.1f}%",   **{"class": "prob-pill tail"}),
                    ui.tags.div(f"<−1%: {f['p_down_1pct']*100:.1f}%", **{"class": "prob-pill tail"}),
                    *extra_pills,
                ),
                ui.tags.div(footer_str,
                    style=f"font-size:0.75rem;color:{c['text_muted']};margin-top:0.5rem;"),
            ))

        gauges = ui.tags.div(
            {"class": "forecast-grid", "style": "margin-top:1.5rem;"},
            *gauge_cards)

        cache_note = ui.tags.div(
            {"style": "margin-top:0.5rem;"},
            ui.tags.span({"class": "cache-badge"}, f"💾 {cache_ts}"))

        return ui.tags.div(table_card, gauges, cache_note)



    # ── Cache status ──────────────────────────────────────────────────────────
    @render.ui
    def cache_status():
        rows = []
        keys = {"market_returns": "Market returns"}
        for mid, label in MODEL_TAB_LABELS.items():
            keys[f"models_{mid}"] = f"Model: {label}"
        keys["forecast_cumulative"] = "Forecast cumulative"

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
                        ui.tags.td(ui.tags.span(
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
        if garch_fc and garch_fc.get("status") == "ok":
            fp   = garch_fc["mean_fc"] * 100
            fv   = garch_fc["vol_fc"]  * 100
            f68l = garch_fc["lower_68"] * 100
            f68h = garch_fc["upper_68"] * 100
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

    def make_dist_fig(col, label, color, live_log_ret=None, garch_fc=None):
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

        # ── 68% CI shading for forecast ───────────────────────────────────────
        if garch_fc and garch_fc.get("status") == "ok":
            f68l = garch_fc["lower_68"] * 100
            f68h = garch_fc["upper_68"] * 100
            f95l = garch_fc["lower_95"] * 100
            f95h = garch_fc["upper_95"] * 100
            y_max_shade = float(kde_vals.max()) * 1.3

            # 95% CI band
            fig.add_shape(type="rect",
                x0=f95l, x1=f95h, y0=0, y1=y_max_shade,
                fillcolor="#A855F7", opacity=0.07,
                line=dict(width=0), layer="below")

            # 68% CI band
            fig.add_shape(type="rect",
                x0=f68l, x1=f68h, y0=0, y1=y_max_shade,
                fillcolor="#A855F7", opacity=0.15,
                line=dict(width=0), layer="below")

        # ── Histogram + KDE ───────────────────────────────────────────────────
        fig.add_trace(go.Histogram(
            x=hist_pct, histnorm="probability density",
            name="Historical", marker_color=color,
            opacity=0.45, nbinsx=60,
            hovertemplate="Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>"))

        fig.add_trace(go.Scatter(
            x=x_range, y=kde_vals, mode="lines", name="KDE",
            line=dict(color=color, width=2.5), hoverinfo="skip"))

        # ── Today marker ──────────────────────────────────────────────────────
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

        # ── Forecast diamond + vertical line ──────────────────────────────────
        if garch_fc and garch_fc.get("status") == "ok":
            fp    = garch_fc["mean_fc"] * 100
            fv    = garch_fc["vol_fc"]  * 100
            f68l  = garch_fc["lower_68"] * 100
            f68h  = garch_fc["upper_68"] * 100
            fdate = garch_fc["next_date"].strftime("%d %b")
            y_max = float(kde_vals.max()) * 1.15

            fig.add_shape(type="line",
                x0=fp, x1=fp, y0=0, y1=y_max,
                line=dict(color="#A855F7", width=1.5, dash="dot"))

            fig.add_trace(go.Scatter(
                x=[fp], y=[y_max * 0.75],
                mode="markers+text",
                marker=dict(symbol="diamond", size=13, color="#A855F7",
                            line=dict(color="white", width=1.5)),
                text=[f"  forecast {fdate}"],
                textposition="middle right",
                textfont=dict(size=9, color="#A855F7"),
                name="Forecast",
                hovertemplate=(
                    f"<b>GARCH forecast</b><br>"
                    f"μ = {fp:+.3f}%<br>"
                    f"σ = {fv:.3f}%/day<br>"
                    f"68% CI: [{f68l:+.3f}%, {f68h:+.3f}%]"
                    f"<extra></extra>")))

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
                text=(f"Mean: {mean_v:.3f}%  |  "
                      f"Std: {std_v:.3f}%  |  "
                      f"Median: {med_v:.3f}%"),
                font=dict(size=10, color=theme["text_muted"]))])
        fig.update_layout(**layout)
        return fig


    # ── Box renderers ─────────────────────────────────────────────────────────
    def _get_selected_fc(ticker):
        res = all_model_results.get()
        if res is None:
            return None
        mid = selected_model_id()
        f   = res.get(ticker, {}).get(mid, {})
        return f if f.get("status") == "ok" else get_default_model_result(res, ticker)

    @render_widget("box_STOXX")
    def box_STOXX():
        d = market_data()
        return make_box_fig(d[0]["^STOXX"].dropna(), "STOXX 600",
            TICKER_COLORS[0], live_log_rets().get("^STOXX"), _get_selected_fc("^STOXX"))

    @render_widget("box_STOXX50E")
    def box_STOXX50E():
        d = market_data()
        return make_box_fig(d[0]["^STOXX50E"].dropna(), "STOXX 50",
            TICKER_COLORS[1], live_log_rets().get("^STOXX50E"), _get_selected_fc("^STOXX50E"))

    @render_widget("box_AEX")
    def box_AEX():
        d = market_data()
        return make_box_fig(d[0]["^AEX"].dropna(), "AEX",
            TICKER_COLORS[2], live_log_rets().get("^AEX"), _get_selected_fc("^AEX"))

    @render_widget("box_GSPC")
    def box_GSPC():
        d = market_data()
        return make_box_fig(d[0]["^GSPC"].dropna(), "S&P 500",
            TICKER_COLORS[3], live_log_rets().get("^GSPC"), _get_selected_fc("^GSPC"))

    @render_widget("box_GDAXI")
    def box_GDAXI():
        d = market_data()
        return make_box_fig(d[0]["^GDAXI"].dropna(), "DAX",
            TICKER_COLORS[4], live_log_rets().get("^GDAXI"), _get_selected_fc("^GDAXI"))

    @render_widget("box_NDX")
    def box_NDX():
        d = market_data()
        return make_box_fig(d[0]["^NDX"].dropna(), "Nasdaq 100",
            TICKER_COLORS[5], live_log_rets().get("^NDX"), _get_selected_fc("^NDX"))

    # ── Dist renderers ────────────────────────────────────────────────────────
    @render_widget("dist_STOXX")
    def dist_STOXX():
        d = market_data(); lv = live_data()
        return make_dist_fig(d[0]["^STOXX"].dropna(), "STOXX 600", TICKER_COLORS[0],
            compute_live_log_return(lv, ["^STOXX"])["^STOXX"] if lv else None,
            _get_selected_fc("^STOXX"))

    @render_widget("dist_STOXX50E")
    def dist_STOXX50E():
        d = market_data(); lv = live_data()
        return make_dist_fig(d[0]["^STOXX50E"].dropna(), "STOXX 50", TICKER_COLORS[1],
            compute_live_log_return(lv, ["^STOXX50E"])["^STOXX50E"] if lv else None,
            _get_selected_fc("^STOXX50E"))

    @render_widget("dist_AEX")
    def dist_AEX():
        d = market_data(); lv = live_data()
        return make_dist_fig(d[0]["^AEX"].dropna(), "AEX", TICKER_COLORS[2],
            compute_live_log_return(lv, ["^AEX"])["^AEX"] if lv else None,
            _get_selected_fc("^AEX"))

    @render_widget("dist_GSPC")
    def dist_GSPC():
        d = market_data(); lv = live_data()
        return make_dist_fig(d[0]["^GSPC"].dropna(), "S&P 500", TICKER_COLORS[3],
            compute_live_log_return(lv, ["^GSPC"])["^GSPC"] if lv else None,
            _get_selected_fc("^GSPC"))

    @render_widget("dist_GDAXI")
    def dist_GDAXI():
        d = market_data(); lv = live_data()
        return make_dist_fig(d[0]["^GDAXI"].dropna(), "DAX", TICKER_COLORS[4],
            compute_live_log_return(lv, ["^GDAXI"])["^GDAXI"] if lv else None,
            _get_selected_fc("^GDAXI"))

    @render_widget("dist_NDX")
    def dist_NDX():
        d = market_data(); lv = live_data()
        return make_dist_fig(d[0]["^NDX"].dropna(), "Nasdaq 100", TICKER_COLORS[5],
            compute_live_log_return(lv, ["^NDX"])["^NDX"] if lv else None,
            _get_selected_fc("^NDX"))

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

            col         = d[1][ticker].dropna()
            ret_aligned = d[0][ticker].dropna().reindex(col.index)
            fig.add_trace(go.Scatter(
                x=col.index.to_pydatetime(),
                y=col.values * 100,
                name=label, mode="lines",
                visible=visible, legendgroup=grp,
                line=dict(color=color, width=2),
                customdata=ret_aligned.values * 100,
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    "Cumulative: <b>%{y:.2f}%</b><br>"
                    "Day return: <b>%{customdata:.2f}%</b>"
                    f"<extra>{label}</extra>"
                ),
            ))

            lc       = live_cum.get(ticker)
            live_ret = llr.get(ticker) if llr else None

            if lc is not None:
                last_date = col.index[-1].to_pydatetime()
                last_val  = float(col.iloc[-1]) * 100
                fig.add_trace(go.Scatter(
                    x=[last_date, today], y=[last_val, lc * 100],
                    mode="lines", visible=visible, legendgroup=grp,
                    showlegend=False,
                    line=dict(color=color, width=1.5, dash="dot"),
                    hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=[today], y=[lc * 100],
                    mode="markers+text", visible=visible, legendgroup=grp,
                    showlegend=False,
                    marker=dict(symbol="circle", size=10, color=color,
                                line=dict(color="white", width=1.5)),
                    text=[f"  {label}"], textposition="middle right",
                    textfont=dict(size=9, color=color),
                    customdata=[[live_ret * 100 if live_ret is not None else float("nan")]],
                    hovertemplate=(
                        "%{x|%d %b %Y %H:%M}<br>"
                        "Cumulative: <b>%{y:.2f}%</b><br>"
                        "Day return: <b>%{customdata[0]:.2f}%</b>"
                        f"<extra>{label} — live</extra>"
                    )))

            if fc_cum:
                fc = fc_cum.get(ticker)
                if fc is not None:
                    conn_x = [today, fc["fc_date"]] if lc is not None \
                             else [fc["anchor_date"], fc["fc_date"]]
                    conn_y = [lc * 100, fc["fc_cum"]] if lc is not None \
                             else [fc["anchor_cum"], fc["fc_cum"]]
                    fig.add_trace(go.Scatter(
                        x=conn_x, y=conn_y, mode="lines",
                        visible=visible, legendgroup=grp, showlegend=False,
                        line=dict(color=color, width=1, dash="dash"),
                        hoverinfo="skip"))
                    fig.add_trace(go.Scatter(
                        x=[fc["fc_date"]], y=[fc["fc_cum"]],
                        mode="markers", visible=visible, legendgroup=grp,
                        showlegend=False,
                        marker=dict(symbol="diamond", size=11, color=color,
                                    line=dict(color="white", width=1.5)),
                        hovertemplate=(
                            "%{x|%d %b %Y}<br>"
                            f"Forecast cumulative: <b>%{{y:.2f}}%</b><br>"
                            f"Forecast return: <b>{fc['fc_return']:+.3f}%</b><br>"
                            f"68% CI: [{fc['fc_68_low']:+.3f}%, "
                            f"{fc['fc_68_high']:+.3f}%]"
                            f"<extra>{label} — GJR-GARCH forecast</extra>"
                        )))

        prev_snap = prev_snapshot.get()
        if prev_snap:
            for i, ticker in enumerate(TICKERS):
                ps = prev_snap.get(ticker)
                if ps is None:
                    continue
                color   = TICKER_COLORS[i]
                visible = "legendonly" if ticker in CUM_HIDDEN else True
                grp     = f"grp_{ticker.replace('^','')}"
                correct = ps["error"] >= 0
                fig.add_trace(go.Scatter(
                    x=[ps["fc_date"], ps["fc_date"]],
                    y=[ps["fc_cum"],  ps["actual_cum"]],
                    mode="lines", visible=visible, legendgroup=grp,
                    showlegend=False,
                    line=dict(color=color, width=1.5, dash="dot"),
                    hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=[ps["fc_date"]], y=[ps["fc_cum"]],
                    mode="markers", visible=visible, legendgroup=grp,
                    showlegend=False,
                    marker=dict(symbol="circle-open", size=10, color=color,
                                line=dict(color=color, width=2)),
                    hovertemplate=(
                        f"%{{x|%d %b %Y}}<br>"
                        f"Predicted: <b>{ps['fc_cum']:+.2f}%</b><br>"
                        f"Pred. return: <b>{ps['fc_return']:+.3f}%</b><br>"
                        f"68% CI: [{ps['fc_68_low']:+.3f}%, {ps['fc_68_high']:+.3f}%]"
                        f"<extra>{TICKER_LABELS[ticker]} — yesterday's forecast</extra>"
                    )))
                dot_color = "#4ADE80" if correct else "#F87171"
                fig.add_trace(go.Scatter(
                    x=[ps["fc_date"]], y=[ps["actual_cum"]],
                    mode="markers", visible=visible, legendgroup=grp,
                    showlegend=False,
                    marker=dict(symbol="circle", size=10, color=dot_color,
                                line=dict(color="white", width=1.5)),
                    hovertemplate=(
                        f"%{{x|%d %b %Y}}<br>"
                        f"Actual: <b>{ps['actual_cum']:+.2f}%</b><br>"
                        f"Actual return: <b>{ps['actual_return']:+.3f}%</b><br>"
                        f"Error (actual−pred): <b>{ps['error']:+.3f}%</b>"
                        f"<extra>{TICKER_LABELS[ticker]} — actual close</extra>"
                    )))

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