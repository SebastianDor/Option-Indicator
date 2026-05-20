from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import yfinance as yf
import numpy as np


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


# ── Config ────────────────────────────────────────────────────────────────────
TICKERS       = ["^STOXX", "^STOXX50E", "^AEX"]
TICKER_LABELS = {"^STOXX": "STOXX", "^STOXX50E": "STOXX 50", "^AEX": "AEX"}
TICKER_COLORS = ["#4A90D9", "#E8734A", "#4CAF82"]

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
        ui.tags.div("Placeholder — add your content here", **{"class": "placeholder-card"}),
        ui.tags.div("Placeholder — add your content here", **{"class": "placeholder-card"}),
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
    {"class": "app-shell"},
    ui.tags.style(make_css(THEMES["light"]), id="themeStyle"),
    ui.tags.div({"class": "sidebar-trigger", "id": "sidebarTrigger"}),
    ui.tags.button("◀", id="sidebarToggle", **{"class": "sidebar-toggle"}),

    ui.tags.div(
        {"class": "sidebar", "id": "sidebar"},
        ui.tags.div("MyApp", **{"class": "app-title"}),
        ui.tags.span("Menu", **{"class": "sidebar-label"}),
        ui.tags.button("🏠  Dashboard", **{"class": "sidebar-item active", "onclick": "setPage('dashboard')"}),
        ui.tags.button("📊  Analysis",  **{"class": "sidebar-item",        "onclick": "setPage('analysis')"}),
        ui.tags.button("📁  Projects",  **{"class": "sidebar-item",        "onclick": "setPage('projects')"}),
        ui.tags.button("👤  Users",     **{"class": "sidebar-item",        "onclick": "setPage('users')"}),
        ui.tags.span("Other", **{"class": "sidebar-label"}),
        ui.tags.button("⚙️  Settings",  **{"class": "sidebar-item",        "onclick": "setPage('settings')"}),
        ui.tags.button("❓  Help",      **{"class": "sidebar-item",        "onclick": "setPage('help')"}),
    ),

    ui.tags.div(
        {"class": "main-content"},
        make_page("dashboard", "Dashboard", "Welcome to your dashboard.", active=True),
        make_analysis_page(),
        make_page("projects",  "Projects",  "Browse and organise your projects."),
        make_page("users",     "Users",     "Manage users and permissions."),
        make_settings_page(),
        make_page("help",      "Help",      "Documentation and support."),
    ),

    ui.tags.script("""
        const lightCSS = `""" + make_css(THEMES["light"]).replace("`", "\\`") + """`;
        const darkCSS  = `""" + make_css(THEMES["dark"]).replace("`", "\\`") + """`;

        function setTheme(theme) {
            document.getElementById('themeStyle').textContent = theme === 'dark' ? darkCSS : lightCSS;
            document.getElementById('opt-light').classList.toggle('selected', theme === 'light');
            document.getElementById('opt-dark').classList.toggle('selected',  theme === 'dark');
            // Notify Shiny so plots re-render with the new theme
            if (window.Shiny) Shiny.setInputValue('active_theme', theme, {priority: 'event'});
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

        function setPage(name) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById('page-' + name).classList.add('active');
            document.querySelectorAll('.sidebar-item').forEach(b => b.classList.remove('active'));
            event.currentTarget.classList.add('active');
        }
    """),
)

# ── Server ────────────────────────────────────────────────────────────────────
def server(input, output, session):

    # Data fetched once — never re-fetched on theme change
    returns, cum_returns = get_index_returns(TICKERS)
    returns     = returns.dropna()
    cum_returns = cum_returns.loc[returns.index]

    # Reactive theme — reads input set by JS, defaults to "light"
    def current_theme() -> dict:
        t = input.active_theme() if "active_theme" in input else "light"
        return THEMES.get(t, THEMES["light"])

    # ── Box plot builder ──────────────────────────────────────────────────────
    def make_box_fig(col: pd.Series, label: str, color: str) -> go.Figure:
        theme = current_theme()
        pct   = col.values * 100
        mn    = float(np.min(pct))
        q1    = float(np.percentile(pct, 25))
        med   = float(np.median(pct))
        mean  = float(np.mean(pct))
        q3    = float(np.percentile(pct, 75))
        mx    = float(np.max(pct))

        fig = go.Figure()

        fig.add_trace(go.Box(
            y=pct,
            x0=0,
            name=label,
            marker_color=color,
            marker=dict(color=color, size=4),
            boxmean="sd",
            boxpoints="outliers",
            hoverinfo="none",
            width=0.4,
        ))

        fig.add_trace(go.Scatter(
            x=[0],
            y=[med],
            mode="markers",
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

        layout = plotly_layout(theme)
        layout["showlegend"] = False
        layout["margin"] = dict(l=40, r=10, t=10, b=30)
        layout["autosize"] = True
        layout["xaxis"] = dict(
            visible=False,
            range=[-1, 1],
            gridcolor=theme["plot_grid"],
        )
        layout["yaxis"] = dict(
            title="Daily return (%)",
            ticksuffix="%",
            gridcolor=theme["plot_grid"],
            zeroline=True,
            zerolinecolor=theme["plot_grid"],
        )
        fig.update_layout(**layout)
        return fig

    @render_widget("box_STOXX")
    def box_STOXX():
        input.active_theme()   # declare reactive dependency
        return make_box_fig(returns["^STOXX"].dropna(), "STOXX", TICKER_COLORS[0])

    @render_widget("box_STOXX50E")
    def box_STOXX50E():
        input.active_theme()   # declare reactive dependency
        return make_box_fig(returns["^STOXX50E"].dropna(), "STOXX 50", TICKER_COLORS[1])

    @render_widget("box_AEX")
    def box_AEX():
        input.active_theme()   # declare reactive dependency
        return make_box_fig(returns["^AEX"].dropna(), "AEX", TICKER_COLORS[2])

    # ── Cumulative returns line chart ─────────────────────────────────────────
    @render_widget("line_cum")
    def line_cum():
        input.active_theme()   # declare reactive dependency
        theme = current_theme()
        fig   = go.Figure()
        for i, ticker in enumerate(TICKERS):
            col   = cum_returns[ticker].dropna()
            dates = col.index.to_pydatetime()
            fig.add_trace(go.Scatter(
                x=dates,
                y=col.values * 100,
                name=TICKER_LABELS[ticker],
                mode="lines",
                line=dict(color=TICKER_COLORS[i], width=2),
                hovertemplate="%{x|%d %b %Y}:  <b>%{y:.2f}%</b><extra>"
                              + TICKER_LABELS[ticker] + "</extra>",
            ))
        layout = plotly_layout(theme, height=380)
        layout["xaxis"] = dict(
            type="date",
            gridcolor=theme["plot_grid"],
            zeroline=False,
        )
        layout["yaxis"] = dict(
            title="Cumulative return (%)",
            ticksuffix="%",
            gridcolor=theme["plot_grid"],
            zeroline=True,
            zerolinecolor=theme["plot_grid"],
        )
        layout["hovermode"] = "x unified"
        layout["legend"] = dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(**layout)
        return fig


app = App(app_ui, server)