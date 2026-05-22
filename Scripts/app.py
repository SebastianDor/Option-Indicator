from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import yfinance as yf
import numpy as np
import psutil
import time


# ── Data ──────────────────────────────────────────────────────────────────────
def get_index_returns(
    tickers: list[str] = ["^STOXX", "^STOXX50E", "^AEX"],
    years_back: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_date   = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date.replace(year=end_date.year - years_back)

    df_multi = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    returns     = pd.DataFrame()
    cum_returns = pd.DataFrame()

    for ticker in tickers:
        prices  = df_multi[ticker]["Close"]
        log_ret = np.log(prices / prices.shift(1))
        cum_ret = np.exp(log_ret.cumsum()) - 1
        returns[ticker]     = log_ret
        cum_returns[ticker] = cum_ret

    return returns, cum_returns

def get_live_prices(tickers):
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
            prev_close = float(closes.iloc[-2])
            live_ret[ticker] = float(np.log(live_px / prev_close))
        except Exception:
            live_ret[ticker] = None
    return live_ret


def compute_live_cum_return(cum_returns: pd.DataFrame, live_prices: dict, tickers: list[str]) -> dict:
    live_cum = {}
    for ticker in tickers:
        try:
            live_px = live_prices.get(ticker)
            if live_px is None:
                live_cum[ticker] = None
                continue
            last_cum   = cum_returns[ticker].dropna().iloc[-1]
            hist_daily = yf.Ticker(ticker).history(period="5d", interval="1d")
            last_close = float(hist_daily["Close"].dropna().iloc[-1])
            first_close = last_close / (1 + last_cum)
            live_cum[ticker] = live_px / first_close - 1
        except Exception:
            live_cum[ticker] = None
    return live_cum


# ── Config ────────────────────────────────────────────────────────────────────
TICKERS       = ["^STOXX", "^STOXX50E", "^AEX"]
TICKER_LABELS = {"^STOXX": "STOXX 600", "^STOXX50E": "STOXX 50", "^AEX": "AEX"}
TICKER_COLORS = ["#4A90D9", "#E8734A", "#4CAF82"]

REFRESH_CHOICES = {"0": "Off", "5": "5s", "10": "10s", "30": "30s", "60": "60s"}

# ── Colour schemes ────────────────────────────────────────────────────────────
THEMES = {
    "light": {
        "primary":        "#4A90D9",
        "primary_dark":   "#2C5F8A",
        "sidebar_bg":     "#1E2A38",
        "sidebar_text":   "#CBD5E1",
        "sidebar_hover":  "#2E3D50",
        "sidebar_active": "#4A90D9",
        "page_bg":        "#F4F6F9",
        "card_bg":        "#FFFFFF",
        "border":         "#E2E8F0",
        "text_main":      "#1E293B",
        "text_muted":     "#64748B",
        "settings_bg":    "#FFFFFF",
        "settings_border":"#E2E8F0",
        "divider":        "#E2E8F0",
        "plot_bg":        "#FFFFFF",
        "plot_paper":     "#F4F6F9",
        "plot_grid":      "#E2E8F0",
        "plot_font":      "#1E293B",
    },
    "dark": {
        "primary":        "#4A90D9",
        "primary_dark":   "#2C5F8A",
        "sidebar_bg":     "#0F1720",
        "sidebar_text":   "#CBD5E1",
        "sidebar_hover":  "#1A2535",
        "sidebar_active": "#4A90D9",
        "page_bg":        "#1A1F2E",
        "card_bg":        "#242B3D",
        "border":         "#2E3A4E",
        "text_main":      "#E8EDF5",
        "text_muted":     "#8A9BB5",
        "settings_bg":    "#242B3D",
        "settings_border":"#2E3A4E",
        "divider":        "#2E3A4E",
        "plot_bg":        "#242B3D",
        "plot_paper":     "#1A1F2E",
        "plot_grid":      "#2E3A4E",
        "plot_font":      "#E8EDF5",
    },
}

# ── CSS ───────────────────────────────────────────────────────────────────────
def make_css(c):
    return f"""
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: 'Segoe UI', sans-serif;
        background: {c['page_bg']};
        color: {c['text_main']};
        height: 100vh;
        overflow: hidden;
        transition: background 0.3s, color 0.3s;
    }}
    .app-shell {{ display: flex; height: 100vh; }}
    #active_theme, label[for=active_theme],
    #proc_page, label[for=proc_page] {{ display: none; }}

    /* ── Sidebar ── */
    .sidebar {{
        width: 220px; min-width: 220px;
        background: {c['sidebar_bg']}; color: {c['sidebar_text']};
        display: flex; flex-direction: column;
        padding: 1rem 0; overflow: hidden;
        transition: width 0.3s ease, min-width 0.3s ease, padding 0.3s ease;
        white-space: nowrap;
    }}
    .sidebar.collapsed {{ width: 0; min-width: 0; padding: 0; }}
    .sidebar .app-title {{
        font-size: 1.1rem; font-weight: 700; color: #fff;
        padding: 0.5rem 1.25rem 1.25rem;
        border-bottom: 1px solid {c['primary_dark']};
        margin-bottom: 0.5rem; overflow: hidden;
    }}
    .sidebar .sidebar-label {{
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {c['text_muted']};
        padding: 0.75rem 1.25rem 0.25rem; overflow: hidden;
    }}
    .sidebar .sidebar-item {{
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.6rem 1.25rem; cursor: pointer;
        font-size: 0.9rem; color: {c['sidebar_text']};
        border-left: 3px solid transparent;
        transition: background 0.15s, border-color 0.15s;
        text-decoration: none; background: none;
        border-right: none; border-top: none; border-bottom: none;
        width: 100%; text-align: left; overflow: hidden;
    }}
    .sidebar .sidebar-item:hover  {{ background: {c['sidebar_hover']}; }}
    .sidebar .sidebar-item.active {{
        background: {c['sidebar_hover']};
        border-left-color: {c['sidebar_active']}; color: #fff;
    }}
    .sidebar-trigger {{
        position: fixed; left: 0; top: 0;
        width: 18px; height: 100vh; z-index: 100;
    }}
    .sidebar-toggle {{
        position: fixed; top: 50%; transform: translateY(-50%);
        left: 220px; transition: left 0.3s ease;
        z-index: 200; background: {c['primary']};
        border: none; color: #fff; width: 18px; height: 48px;
        border-radius: 0 6px 6px 0; cursor: pointer; font-size: 0.65rem;
        display: flex; align-items: center; justify-content: center; opacity: 0.75;
    }}
    .sidebar-toggle:hover   {{ opacity: 1; }}
    .sidebar-toggle.collapsed {{ left: 0; }}

    /* ── Refresh control in sidebar ── */
    .refresh-control {{
        padding: 0.5rem 1.25rem 0.75rem;
        overflow: hidden;
    }}
    .refresh-control label {{
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {c['text_muted']};
        display: block; margin-bottom: 0.4rem;
    }}
    .refresh-control select {{
        width: 100%; background: {c['sidebar_hover']};
        color: {c['sidebar_text']}; border: 1px solid {c['primary_dark']};
        border-radius: 5px; padding: 0.3rem 0.5rem;
        font-size: 0.82rem; cursor: pointer; outline: none;
    }}

    /* ── Main content ── */
    .main-content {{
        flex: 1; overflow-y: auto; padding: 2rem;
        background: {c['page_bg']}; transition: background 0.3s;
    }}
    .page {{ display: none; }}
    .page.active {{ display: block; }}
    .page-header {{
        margin-bottom: 1.5rem; padding-bottom: 1rem;
        border-bottom: 1px solid {c['border']};
    }}
    .page-header h1 {{ font-size: 1.5rem; font-weight: 600; color: {c['text_main']}; }}
    .page-header p  {{ font-size: 0.9rem; color: {c['text_muted']}; margin-top: 0.25rem; }}
    .placeholder-card {{
        background: {c['card_bg']}; border-radius: 8px; padding: 2rem;
        margin-bottom: 1.5rem; min-height: 180px;
        display: flex; align-items: center; justify-content: center;
        color: {c['text_muted']}; font-size: 0.95rem;
        border: 2px dashed {c['border']};
    }}
    .plot-card {{
        background: {c['card_bg']}; border: 1px solid {c['border']};
        border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
        min-width: 0; overflow: hidden;
    }}
    .plot-card h3 {{
        font-size: 0.95rem; font-weight: 600;
        color: {c['text_main']}; margin-bottom: 0.25rem;
    }}
    .plot-card p {{ font-size: 0.8rem; color: {c['text_muted']}; margin-bottom: 1rem; }}
    .plot-row {{
        display: grid; grid-template-columns: 1fr 1fr 1fr;
        gap: 1.25rem; margin-bottom: 1.5rem;
    }}
    .plot-row > * {{ min-width: 0; }}

    /* ── Server stat cards ── */
    .stat-row {{
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem; margin-bottom: 1.5rem;
    }}
    .stat-card {{
        background: {c['card_bg']}; border: 1px solid {c['border']};
        border-radius: 10px; padding: 1.25rem 1.5rem;
    }}
    .stat-card .stat-label {{
        font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.06em; color: {c['text_muted']}; margin-bottom: 0.4rem;
    }}
    .stat-card .stat-value {{
        font-size: 1.6rem; font-weight: 700; color: {c['text_main']};
    }}
    .stat-card .stat-sub {{
        font-size: 0.78rem; color: {c['text_muted']}; margin-top: 0.2rem;
    }}

    /* ── Process table ── */
    .proc-table {{
        width: 100%; border-collapse: collapse; font-size: 0.85rem;
    }}
    .proc-table th {{
        text-align: left; padding: 0.5rem 0.75rem;
        border-bottom: 2px solid {c['border']};
        color: {c['text_muted']}; font-weight: 600;
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .proc-table td {{
        padding: 0.45rem 0.75rem; border-bottom: 1px solid {c['border']};
        color: {c['text_main']};
    }}
    .proc-table tr:last-child td {{ border-bottom: none; }}

    /* ── Settings ── */
    .settings-page {{ max-width: 620px; }}
    .settings-section {{
        background: {c['settings_bg']}; border: 1px solid {c['settings_border']};
        border-radius: 10px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    }}
    .settings-section h2 {{
        font-size: 1rem; font-weight: 600; color: {c['text_main']}; margin-bottom: 0.25rem;
    }}
    .settings-section .section-desc {{
        font-size: 0.82rem; color: {c['text_muted']}; margin-bottom: 1.25rem;
    }}
    .settings-divider {{ border: none; border-top: 1px solid {c['divider']}; margin: 1.25rem 0; }}
    .settings-placeholder {{
        font-size: 0.88rem; color: {c['text_muted']}; font-style: italic; padding: 0.5rem 0;
    }}
    .theme-options {{ display: flex; gap: 1rem; margin-top: 0.25rem; }}
    .theme-option {{
        display: flex; align-items: center; gap: 0.5rem;
        cursor: pointer; padding: 0.6rem 1.1rem;
        border-radius: 8px; border: 2px solid {c['border']};
        background: {c['page_bg']}; color: {c['text_main']};
        font-size: 0.9rem; transition: border-color 0.2s, background 0.2s; user-select: none;
    }}
    .theme-option input[type=radio] {{ accent-color: {c['primary']}; }}
    .theme-option.selected {{ border-color: {c['primary']}; background: {c['card_bg']}; }}
    """

# ── Plotly layout helper ──────────────────────────────────────────────────────
def plotly_layout(theme: dict, height: int = 300) -> dict:
    return dict(
        paper_bgcolor=theme["plot_paper"],
        plot_bgcolor=theme["plot_bg"],
        font=dict(color=theme["plot_font"], family="Segoe UI", size=12),
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(gridcolor=theme["plot_grid"], zeroline=False),
        yaxis=dict(gridcolor=theme["plot_grid"], zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=height,
    )

# ── Generic page helper ───────────────────────────────────────────────────────
def make_page(page_id, title, subtitle, active=False):
    return ui.tags.div(
        {"class": f"page {'active' if active else ''}", "id": f"page-{page_id}"},
        ui.tags.div(
            {"class": "page-header"},
            ui.tags.h1(title),
            ui.tags.p(subtitle),
        ),
        ui.tags.div("Placeholder", **{"class": "placeholder-card"}),
        ui.tags.div("Placeholder", **{"class": "placeholder-card"}),
    )

# ── Analysis page ─────────────────────────────────────────────────────────────
def make_analysis_page():
    return ui.tags.div(
        {"class": "page", "id": "page-analysis"},
        ui.tags.div(
            {"class": "page-header"},
            ui.tags.h1("Analysis"),
            ui.tags.p("Index return distributions and cumulative performance."),
        ),
        ui.tags.div(
            {"class": "plot-row"},
            *[
                ui.tags.div(
                    {"class": "plot-card"},
                    ui.tags.h3(f"{TICKER_LABELS[t]} — Return Distribution"),
                    ui.tags.p("Daily log returns"),
                    output_widget(f"box_{t.replace('^', '')}"),
                )
                for t in TICKERS
            ],
        ),
        ui.tags.div(
            {"class": "plot-card"},
            ui.tags.h3("Cumulative Returns"),
            ui.tags.p("Log-cumulative returns across all three indexes"),
            output_widget("line_cum"),
        ),
    )

# ── Server page ───────────────────────────────────────────────────────────────
def make_server_page():
    return ui.tags.div(
        {"class": "page", "id": "page-server"},
        ui.tags.div(
            {"class": "page-header"},
            ui.tags.h1("Server"),
            ui.tags.p("Live system resource usage and running processes."),
        ),
        # Stat cards
        ui.tags.div(
            {"class": "stat-row"},
            ui.output_ui("stat_cpu"),
            ui.output_ui("stat_ram"),
            ui.output_ui("stat_disk"),
            ui.output_ui("stat_uptime"),
        ),
        # Process table
        ui.tags.div(
            {"class": "plot-card"},
            ui.tags.h3("Top Processes"),
            ui.tags.p("Top 15 processes by CPU usage"),
            ui.output_ui("proc_table"),
        ),
    )

# ── Settings page ─────────────────────────────────────────────────────────────
def make_settings_page():
    return ui.tags.div(
        {"class": "page", "id": "page-settings"},
        ui.tags.div(
            {"class": "page-header"},
            ui.tags.h1("Settings"),
            ui.tags.p("Manage your application preferences."),
        ),
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
                        "☀️  Light mode",
                    ),
                    ui.tags.label(
                        {"class": "theme-option", "id": "opt-dark", "onclick": "setTheme('dark')"},
                        ui.tags.input(type="radio", name="theme", value="dark"),
                        "🌙  Dark mode",
                    ),
                ),
            ),
            ui.tags.div(
                {"class": "settings-section"},
                ui.tags.h2("Account"),
                ui.tags.p("Manage your account details.", **{"class": "section-desc"}),
                ui.tags.hr(**{"class": "settings-divider"}),
                ui.tags.p("— Placeholder for future account settings —", **{"class": "settings-placeholder"}),
            ),
            ui.tags.div(
                {"class": "settings-section"},
                ui.tags.h2("Notifications"),
                ui.tags.p("Configure notification preferences.", **{"class": "section-desc"}),
                ui.tags.hr(**{"class": "settings-divider"}),
                ui.tags.p("— Placeholder for future notification settings —", **{"class": "settings-placeholder"}),
            ),
            ui.tags.div(
                {"class": "settings-section"},
                ui.tags.h2("Data & Privacy"),
                ui.tags.p("Control data usage and privacy options.", **{"class": "section-desc"}),
                ui.tags.hr(**{"class": "settings-divider"}),
                ui.tags.p("— Placeholder for future privacy settings —", **{"class": "settings-placeholder"}),
            ),
        ),
    )

# ── UI ────────────────────────────────────────────────────────────────────────
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
            ui.tags.button("🏠  Dashboard", **{"class": "sidebar-item active", "onclick": "setPage('dashboard')"}),
            ui.tags.button("📊  Analysis",  **{"class": "sidebar-item",        "onclick": "setPage('analysis')"}),
            ui.tags.button("🖥️  Server",    **{"class": "sidebar-item",        "onclick": "setPage('server')"}),
            ui.tags.button("👤  Users",     **{"class": "sidebar-item",        "onclick": "setPage('users')"}),
            ui.tags.span("Other", **{"class": "sidebar-label"}),
            ui.tags.button("⚙️  Settings",  **{"class": "sidebar-item",        "onclick": "setPage('settings')"}),
            ui.tags.button("❓  Help",      **{"class": "sidebar-item",        "onclick": "setPage('help')"}),
            # Refresh control — shown only on Analysis and Server pages
            ui.tags.span("Refresh", **{"class": "sidebar-label", "id": "refresh-label",
                                       "style": "display:none"}),
            ui.tags.div(
                {"class": "refresh-control", "id": "refresh-control", "style": "display:none"},
                ui.input_select(
                    "refresh_interval",
                    label="Auto-refresh",
                    choices=REFRESH_CHOICES,
                    selected="0",
                ),
            ),
        ),

        ui.tags.div(
            {"class": "main-content"},
            make_page("dashboard", "Dashboard", "Welcome to your dashboard.", active=True),
            make_analysis_page(),
            make_server_page(),
            make_page("users",     "Users",     "Manage users and permissions."),
            make_settings_page(),
            make_page("help",      "Help",      "Documentation and support."),
        ),
    ),

    ui.tags.script("""
        const lightCSS = `""" + make_css(THEMES["light"]).replace("`", "\\`") + """`;
        const darkCSS  = `""" + make_css(THEMES["dark"]).replace("`", "\\`") + """`;

        function setTheme(theme) {
            document.getElementById('themeStyle').textContent = theme === 'dark' ? darkCSS : lightCSS;
            document.getElementById('opt-light').classList.toggle('selected', theme === 'light');
            document.getElementById('opt-dark').classList.toggle('selected',  theme === 'dark');
            const el = document.getElementById('active_theme');
            el.value = theme;
            el.dispatchEvent(new Event('change'));
        }

        const sidebar = document.getElementById('sidebar');
        const toggle  = document.getElementById('sidebarToggle');
        const trigger = document.getElementById('sidebarTrigger');
        let pinned = true, hideTimer = null;

        function showSidebar() {
            clearTimeout(hideTimer);
            sidebar.classList.remove('collapsed');
            toggle.classList.remove('collapsed');
            toggle.textContent = '◀';
        }
        function hideSidebar() {
            sidebar.classList.add('collapsed');
            toggle.classList.add('collapsed');
            toggle.textContent = '▶';
        }
        toggle.addEventListener('click', () => { pinned = !pinned; pinned ? showSidebar() : hideSidebar(); });
        trigger.addEventListener('mouseenter', () => { if (!pinned) showSidebar(); });
        sidebar.addEventListener('mouseleave', () => { if (!pinned) hideTimer = setTimeout(hideSidebar, 300); });
        sidebar.addEventListener('mouseenter', () => clearTimeout(hideTimer));

        const REFRESH_PAGES = ['analysis', 'server'];

        function setPage(name) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById('page-' + name).classList.add('active');
            document.querySelectorAll('.sidebar-item').forEach(b => b.classList.remove('active'));
            event.currentTarget.classList.add('active');
            const show = REFRESH_PAGES.includes(name);
            document.getElementById('refresh-control').style.display = show ? 'block' : 'none';
            document.getElementById('refresh-label').style.display   = show ? 'block' : 'none';
        }

        function updatePage(delta) {
            const el = document.getElementById('proc_page');
            el.value = Math.max(1, parseInt(el.value || 1) + delta);
            el.dispatchEvent(new Event('change'));
        }
    """),
)

# ── Server ────────────────────────────────────────────────────────────────────
def server(input, output, session):

    market_data    = reactive.Value(None)
    live_data      = reactive.Value(None)
    last_load_date = None  # plain Python variable, not reactive

    def load_data():
        nonlocal last_load_date
        today = datetime.now().date()
        # Only reload historical data if it's a new day or we have nothing yet
        if last_load_date != today or market_data.get() is None:
            r, c = get_index_returns(TICKERS)
            r = r.dropna()
            c = c.loc[r.index]
            market_data.set((r, c))
            last_load_date = today

    def load_live():
        live_data.set(get_live_prices(TICKERS))

    # initial load
    load_data()
    load_live()

    @reactive.Effect
    def _auto_refresh():
        interval = int(input.refresh_interval())
        if interval > 0:
            reactive.invalidate_later(interval)
            load_data()   # will only reload historical if date changed
            load_live()   # always refresh live prices

    def current_theme() -> dict:
        return THEMES.get(input.active_theme(), THEMES["light"])

    def make_box_fig(col: pd.Series, label: str, color: str, live_log_ret=None) -> go.Figure:
        theme    = current_theme()
        hist_pct = col.values * 100
        all_pct  = np.append(hist_pct, live_log_ret * 100) if live_log_ret is not None else hist_pct

        mn   = float(np.min(all_pct))
        q1   = float(np.percentile(all_pct, 25))
        med  = float(np.median(all_pct))
        mean = float(np.mean(all_pct))
        q3   = float(np.percentile(all_pct, 75))
        mx   = float(np.max(all_pct))

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=all_pct, x0=0, name=label,
            marker_color=color, marker=dict(color=color, size=4),
            boxmean="sd", boxpoints="outliers", hoverinfo="none", width=0.4,
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[med], mode="markers",
            marker=dict(size=12, opacity=0, color=color),
            showlegend=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Max:    {mx:.2f}%<br>"
                f"Q3:     {q3:.2f}%<br>"
                f"Mean:   {mean:.2f}%<br>"
                f"Median: {med:.2f}%<br>"
                f"Q1:     {q1:.2f}%<br>"
                f"Min:    {mn:.2f}%"
                "<extra></extra>"
            ),
        ))
        if live_log_ret is not None:
            live_pct = live_log_ret * 100
            fig.add_trace(go.Scatter(
                x=[0], y=[live_pct],
                mode="markers+text",
                marker=dict(
                    symbol="x",
                    size=12,
                    color="#FFD700",
                    line=dict(color="#FFD700", width=2),
                ),
                text=["  today"],
                textposition="middle right",
                textfont=dict(size=9, color="#FFD700"),
                showlegend=False,
                hovertemplate=f"<b>Today's return</b><br>{live_pct:.2f}%<extra></extra>",
            ))

        layout = plotly_layout(theme)
        layout["showlegend"] = False
        layout["margin"]     = dict(l=40, r=10, t=10, b=30)
        layout["autosize"]   = True
        layout["xaxis"]      = dict(visible=False, range=[-1, 1], gridcolor=theme["plot_grid"])
        layout["yaxis"]      = dict(title="Daily return (%)", ticksuffix="%",
                                    gridcolor=theme["plot_grid"],
                                    zeroline=True, zerolinecolor=theme["plot_grid"])
        fig.update_layout(**layout)
        return fig

    @render_widget("box_STOXX")
    def box_STOXX():
        d  = market_data()
        lv = live_data()
        if d is None:
            return go.Figure()
        live_ret = compute_live_log_return(lv, ["^STOXX"])["^STOXX"] if lv else None
        return make_box_fig(d[0]["^STOXX"].dropna(), "STOXX 600", TICKER_COLORS[0], live_ret)

    @render_widget("box_STOXX50E")
    def box_STOXX50E():
        d  = market_data()
        lv = live_data()
        if d is None:
            return go.Figure()
        live_ret = compute_live_log_return(lv, ["^STOXX50E"])["^STOXX50E"] if lv else None
        return make_box_fig(d[0]["^STOXX50E"].dropna(), "STOXX 50", TICKER_COLORS[1], live_ret)

    @render_widget("box_AEX")
    def box_AEX():
        d  = market_data()
        lv = live_data()
        if d is None:
            return go.Figure()
        live_ret = compute_live_log_return(lv, ["^AEX"])["^AEX"] if lv else None
        return make_box_fig(d[0]["^AEX"].dropna(), "AEX", TICKER_COLORS[2], live_ret)

    @render_widget("line_cum")
    def line_cum():
        d  = market_data()
        lv = live_data()
        if d is None:
            return go.Figure()
        theme = current_theme()
        fig   = go.Figure()

        for i, ticker in enumerate(TICKERS):
            col      = d[1][ticker].dropna()
            ret_col  = d[0][ticker].dropna()
            dates    = col.index.to_pydatetime()

            # Align daily returns to cumulative index
            ret_aligned = ret_col.reindex(col.index)

            fig.add_trace(go.Scatter(
                x=dates,
                y=col.values * 100,
                name=TICKER_LABELS[ticker],
                mode="lines",
                line=dict(color=TICKER_COLORS[i], width=2),
                customdata=ret_aligned.values * 100,
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    "Cumulative: <b>%{y:.2f}%</b><br>"
                    "Day return: <b>%{customdata:.2f}%</b>"
                    "<extra>" + TICKER_LABELS[ticker] + "</extra>"
                ),
            ))

        if lv:
            live_cum = compute_live_cum_return(d[1], lv, TICKERS)
            today    = datetime.now()
            for i, ticker in enumerate(TICKERS):
                lc = live_cum.get(ticker)
                if lc is None:
                    continue
                last_date = d[1][ticker].dropna().index[-1].to_pydatetime()
                last_val  = float(d[1][ticker].dropna().iloc[-1]) * 100
                fig.add_trace(go.Scatter(
                    x=[last_date, today], y=[last_val, lc * 100],
                    mode="lines",
                    line=dict(color=TICKER_COLORS[i], width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                live_ret = compute_live_log_return(lv, [ticker])[ticker]
                fig.add_trace(go.Scatter(
                    x=[today], y=[lc * 100],
                    mode="markers",
                    marker=dict(symbol="circle", size=10, color=TICKER_COLORS[i],
                                line=dict(color="white", width=1.5)),
                    name=f"{TICKER_LABELS[ticker]} (live)",
                    customdata=[[live_ret * 100 if live_ret is not None else float("nan")]],
                    hovertemplate=(
                        "%{x|%d %b %Y %H:%M}<br>"
                        "Cumulative: <b>%{y:.2f}%</b><br>"
                        "Day return: <b>%{customdata[0]:.2f}%</b>"
                        "<extra>" + TICKER_LABELS[ticker] + " — live</extra>"
                    ),
                ))

        layout = plotly_layout(theme, height=380)
        layout["xaxis"]     = dict(type="date", gridcolor=theme["plot_grid"], zeroline=False)
        layout["yaxis"]     = dict(title="Cumulative return (%)", ticksuffix="%",
                                   gridcolor=theme["plot_grid"], zeroline=True,
                                   zerolinecolor=theme["plot_grid"])
        layout["hovermode"] = "x unified"
        layout["legend"]    = dict(orientation="h", yanchor="bottom", y=1.02,
                                   xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
        fig.update_layout(**layout)
        return fig

    def _stat_card(label: str, value: str, sub: str = "") -> ui.Tag:
        return ui.tags.div(
            {"class": "stat-card"},
            ui.tags.div(label, **{"class": "stat-label"}),
            ui.tags.div(value, **{"class": "stat-value"}),
            ui.tags.div(sub,   **{"class": "stat-sub"}) if sub else ui.tags.span(),
        )

    @render.ui
    def stat_cpu():
        interval = int(input.refresh_interval())
        if interval > 0:
            reactive.invalidate_later(interval)
        pct   = psutil.cpu_percent(interval=0.2)
        cores = psutil.cpu_count(logical=True)
        return _stat_card("CPU Usage", f"{pct:.1f}%", f"{cores} logical cores")

    @render.ui
    def stat_ram():
        interval = int(input.refresh_interval())
        if interval > 0:
            reactive.invalidate_later(interval)
        vm    = psutil.virtual_memory()
        used  = vm.used  / 1024**3
        total = vm.total / 1024**3
        return _stat_card("RAM Usage", f"{vm.percent:.1f}%", f"{used:.1f} / {total:.1f} GB")

    @render.ui
    def stat_disk():
        interval = int(input.refresh_interval())
        if interval > 0:
            reactive.invalidate_later(interval)
        dk    = psutil.disk_usage("/")
        used  = dk.used  / 1024**3
        total = dk.total / 1024**3
        return _stat_card("Disk Usage", f"{dk.percent:.1f}%", f"{used:.1f} / {total:.1f} GB")

    @render.ui
    def stat_uptime():
        interval = int(input.refresh_interval())
        if interval > 0:
            reactive.invalidate_later(interval)
        boot   = psutil.boot_time()
        uptime = time.time() - boot
        h, rem = divmod(int(uptime), 3600)
        m, s   = divmod(rem, 60)
        return _stat_card("Uptime", f"{h}h {m}m",
                          f"Boot: {datetime.fromtimestamp(boot).strftime('%d %b %H:%M')}")

    @render.ui
    def proc_table():
        interval = int(input.refresh_interval())
        if interval > 0:
            reactive.invalidate_later(interval)

        SKIP      = {"system idle process", "idle"}
        proc_objs = list(psutil.process_iter(["pid", "name", "memory_percent", "status"]))
        for p in proc_objs:
            try:
                p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(0.3)

        procs_info = []
        cpu_count  = psutil.cpu_count(logical=True) or 1
        for p in proc_objs:
            try:
                name = p.info["name"] or "—"
                if name.lower() in SKIP:
                    continue
                cpu = p.cpu_percent(interval=None) / cpu_count
                procs_info.append({
                    "pid":    p.info["pid"],
                    "name":   name,
                    "cpu":    cpu,
                    "mem":    p.info["memory_percent"] or 0.0,
                    "status": p.info["status"] or "—",
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        procs_info.sort(key=lambda x: (x["cpu"], x["mem"]), reverse=True)

        page_size    = 15
        total        = len(procs_info)
        n_pages      = max(1, (total + page_size - 1) // page_size)
        try:
            current_page = int(input.proc_page())
        except Exception:
            current_page = 1
        current_page = max(1, min(current_page, n_pages))

        start = (current_page - 1) * page_size
        chunk = procs_info[start : start + page_size]

        rows = [
            ui.tags.tr(
                ui.tags.td(str(p["pid"])),
                ui.tags.td(p["name"]),
                ui.tags.td(f'{p["cpu"]:.1f}%'),
                ui.tags.td(f'{p["mem"]:.1f}%'),
                ui.tags.td(p["status"]),
            )
            for p in chunk
        ]

        pagination = ui.tags.div(
            {"style": "display:flex; gap:0.5rem; align-items:center; margin-top:1rem; font-size:0.85rem;"},
            ui.tags.button("← Prev", id="proc_prev", onclick="updatePage(-1)",
                           disabled=current_page <= 1,
                           style="padding:0.3rem 0.75rem; cursor:pointer; border-radius:5px; border:1px solid #ccc;"),
            ui.tags.span(f"Page {current_page} of {n_pages}"),
            ui.tags.button("Next →", id="proc_next", onclick="updatePage(1)",
                           disabled=current_page >= n_pages,
                           style="padding:0.3rem 0.75rem; cursor:pointer; border-radius:5px; border:1px solid #ccc;"),
            ui.tags.span(f"({total} processes)", style="margin-left:0.5rem; color:#888;"),
        )

        return ui.tags.div(
            ui.tags.table(
                {"class": "proc-table"},
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("PID"), ui.tags.th("Name"),
                    ui.tags.th("CPU %"), ui.tags.th("MEM %"), ui.tags.th("Status"),
                )),
                ui.tags.tbody(*rows),
            ),
            pagination,
        )


app = App(app_ui, server)