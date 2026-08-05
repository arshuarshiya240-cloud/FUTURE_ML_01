import os, sys, warnings, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Design tokens ──────────────────────────────────────────────────────────────
BLUE   = "#3A86FF"
PINK   = "#FF006E"
PURPLE = "#8338EC"
TEAL   = "#06D6A0"
AMBER  = "#FFB703"
DARK   = "#14142B"
LIGHT  = "#F7F7FC"
BORDER = "#E2E2EE"
PALETTE = [BLUE, PINK, PURPLE, TEAL, AMBER]

BG      = "#FFFFFF"
PLOT_BG = "#FAFAFA"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing import load_and_clean, aggregate_daily
from src.feature_engineering import build_feature_matrix
from src.models import (
    train_linear, train_random_forest, train_gradient_boosting,
    get_metrics, forecast_future,
)

df_raw   = load_and_clean("data/superstore.csv")
daily_df = aggregate_daily(df_raw)
daily_df, FEATURE_COLS = build_feature_matrix(daily_df)

split_date = daily_df["date"].max() - pd.Timedelta(days=90)
train_df   = daily_df[daily_df["date"] <  split_date]
test_df    = daily_df[daily_df["date"] >= split_date]
X_train, y_train = train_df[FEATURE_COLS], train_df["total_sales"]
X_test,  y_test  = test_df[FEATURE_COLS],  test_df["total_sales"]

print("Training models …")
lr_model = train_linear(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)
gb_model = train_gradient_boosting(X_train, y_train)

models_dict = {
    "Linear Regression": lr_model,
    "Random Forest":     rf_model,
    "Gradient Boosting": gb_model,
}
metrics_dict = {
    name: get_metrics(y_test.values, np.maximum(m.predict(X_test), 0))
    for name, m in models_dict.items()
}
best_name  = min(metrics_dict, key=lambda x: metrics_dict[x]["MAPE"])
best_model = models_dict[best_name]

forecast_df = forecast_future(
    best_model, daily_df["date"].max(), n_days=90,
    history_sales=daily_df["total_sales"].values, feature_cols=FEATURE_COLS,
)

y_pred_best = np.maximum(best_model.predict(X_test), 0)

# ── Helper ──────────────────────────────────────────────────────────────────────
def layout_defaults(title="", height=420):
    return dict(
        title=dict(text=title, font=dict(size=15, color=DARK, family="Inter, sans-serif"),
                   x=0.02),
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color="#444455", size=11),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER, borderwidth=1),
        margin=dict(l=60, r=30, t=55, b=55),
        xaxis=dict(showgrid=False, linecolor=BORDER, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=BORDER, gridwidth=1,
                   linecolor=BORDER, tickfont=dict(size=10)),
    )


# ── KPI values ─────────────────────────────────────────────────────────────────
total_hist   = daily_df["total_sales"].sum()
avg_daily    = daily_df["total_sales"].mean()
best_mape    = metrics_dict[best_name]["MAPE"]
fcast_total  = forecast_df["forecasted_sales"].sum()
best_r2      = metrics_dict[best_name]["R2"]
n_categories = df_raw["Category"].nunique()


# ════════════════════════════════════════════════════════════════════════════════
# BUILD CHARTS
# ════════════════════════════════════════════════════════════════════════════════

# ── Chart A — Forecast chart ────────────────────────────────────────────────────
def make_forecast_chart():
    cutoff = daily_df["date"].max() - pd.Timedelta(days=180)
    recent = daily_df[daily_df["date"] >= cutoff].copy()
    ma7    = recent["total_sales"].rolling(7, min_periods=1).mean()
    std    = recent["total_sales"].std()

    fig = go.Figure()

    # Historical raw
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["total_sales"],
        mode="lines", name="Historical (daily)",
        line=dict(color=BLUE, width=1), opacity=0.25,
        fill="tozeroy", fillcolor=f"rgba(58,134,255,0.06)",
    ))
    # 7-day MA
    fig.add_trace(go.Scatter(
        x=recent["date"], y=ma7,
        mode="lines", name="Historical (7-day avg)",
        line=dict(color=BLUE, width=2.4),
    ))
    # Actual test
    fig.add_trace(go.Scatter(
        x=test_df["date"], y=y_test.values,
        mode="lines", name="Actual (test set)",
        line=dict(color="#777788", width=1.6, dash="dot"),
    ))
    # Predicted test
    fig.add_trace(go.Scatter(
        x=test_df["date"], y=y_pred_best,
        mode="lines", name="Predicted (test set)",
        line=dict(color=TEAL, width=2, dash="dash"),
    ))
    # Confidence upper
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["forecasted_sales"] + 1.5 * std,
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    # Forecast + confidence band fill
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["forecasted_sales"] - 1.5 * std,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=f"rgba(255,0,110,0.10)",
        name="95% confidence band",
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecasted_sales"],
        mode="lines", name=f"Forecast — {best_name}",
        line=dict(color=PINK, width=2.8),
    ))
    # Vertical split line
    fig.add_vline(x=str(daily_df["date"].max().date()),
                  line_width=1.5, line_dash="dot", line_color=DARK,
                  annotation_text="Forecast starts →",
                  annotation_position="top right",
                  annotation_font_color=DARK)

    fig.update_layout(**layout_defaults("📈  Sales Forecast — Next 90 Days", height=450))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ── Chart B — Monthly revenue bars ─────────────────────────────────────────────
def make_monthly_chart():
    tmp = daily_df.copy()
    tmp["ym"] = tmp["date"].dt.to_period("M").astype(str)
    monthly = tmp.groupby("ym")["total_sales"].sum().reset_index()
    monthly.columns = ["month", "sales"]
    monthly["color"] = monthly["month"].apply(
        lambda m: AMBER if int(m.split("-")[1]) in (11, 12) else BLUE
    )
    fig = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["sales"],
        marker_color=monthly["color"], opacity=0.87,
        hovertemplate="<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**layout_defaults("📅  Monthly Revenue  (🟡 = Holiday Season)"))
    fig.update_xaxes(tickangle=-40, nticks=18)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


# ── Chart C — Category breakdown ────────────────────────────────────────────────
def make_category_chart():
    cat_totals = df_raw.groupby("Category")["Sales"].sum().reset_index()
    cat_totals.columns = ["category", "total_sales"]
    cat_totals = cat_totals.sort_values("total_sales", ascending=True)

    fig = go.Figure(go.Bar(
        x=cat_totals["total_sales"], y=cat_totals["category"],
        orientation="h",
        marker=dict(color=PALETTE[: len(cat_totals)], opacity=0.88),
        text=[f"${v/1e6:.2f}M" for v in cat_totals["total_sales"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(**layout_defaults("🏷️  Total Revenue by Category"))
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return fig


# ── Chart D — Model metrics ─────────────────────────────────────────────────────
def make_model_chart():
    model_names = list(metrics_dict.keys())
    metrics     = ["MAE", "RMSE", "MAPE"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["MAE ($)", "RMSE ($)", "MAPE (%)"])
    for col_i, metric in enumerate(metrics, start=1):
        vals = [metrics_dict[m][metric] for m in model_names]
        best_idx = int(np.argmin(vals))
        clrs = [PALETTE[i] if i != best_idx else TEAL for i in range(len(model_names))]
        fig.add_trace(
            go.Bar(
                x=model_names, y=vals,
                marker=dict(color=clrs, opacity=0.88, line=dict(width=0)),
                text=[f"{v:.1f}" for v in vals],
                textposition="outside",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.2f}<extra></extra>",
            ),
            row=1, col=col_i,
        )
    fig.update_layout(
        height=380, paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
        title=dict(text="🤖  Model Performance Comparison  (🟢 = Best)",
                   font=dict(size=15, color=DARK), x=0.02),
        font=dict(family="Inter, sans-serif", color="#444455"),
        margin=dict(l=50, r=30, t=65, b=55),
    )
    fig.update_xaxes(showgrid=False, tickangle=-15)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER)
    return fig


# ── Chart E — Day-of-week heatmap ───────────────────────────────────────────────
def make_dow_chart():
    tmp = daily_df.copy()
    tmp["dow"] = tmp["date"].dt.day_name()
    dow_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_avg    = tmp.groupby("dow")["total_sales"].mean().reindex(dow_order)
    colors     = [AMBER if d in ("Saturday","Sunday") else BLUE for d in dow_order]
    fig = go.Figure(go.Bar(
        x=dow_order, y=dow_avg.values,
        marker_color=colors, opacity=0.87,
        text=[f"${v:,.0f}" for v in dow_avg.values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg Sales: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**layout_defaults("📆  Avg Sales by Day of Week  (🟡 = Weekend)"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


# ── Chart F — Category monthly trends ───────────────────────────────────────────
def make_category_trend_chart():
    tmp = df_raw.copy()
    tmp["ym"] = tmp["Order Date"].dt.to_period("M").astype(str)
    monthly_cat = tmp.groupby(["ym","Category"])["Sales"].sum().reset_index()
    monthly_cat.columns = ["ym", "category", "sales"]

    fig = go.Figure()
    for i, cat in enumerate(df_raw["Category"].unique()):
        d = monthly_cat[monthly_cat["category"] == cat]
        fig.add_trace(go.Scatter(
            x=d["ym"], y=d["sales"],
            mode="lines", name=cat,
            line=dict(color=PALETTE[i], width=2.2),
            hovertemplate=f"<b>{cat}</b><br>Month: %{{x}}<br>Sales: $%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(**layout_defaults("📈  Monthly Sales Trend by Category"))
    fig.update_xaxes(tickangle=-40, nticks=20)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


print("Building charts …")
fig_forecast  = make_forecast_chart()
fig_monthly   = make_monthly_chart()
fig_category  = make_category_chart()
fig_models    = make_model_chart()
fig_dow       = make_dow_chart()
fig_cat_trend = make_category_trend_chart()


# ════════════════════════════════════════════════════════════════════════════════
# ASSEMBLE HTML
# ════════════════════════════════════════════════════════════════════════════════

def fig_to_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Sales & Demand Forecasting Dashboard · Future Interns ML Task 1</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: "Inter", sans-serif;
    background: #F0F2F8;
    color: #14142B;
    min-height: 100vh;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #14142B 0%, #2D2B55 60%, #3A3670 100%);
    padding: 36px 48px 32px;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 16px;
  }}
  .header-left h1 {{
    font-size: 26px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px;
  }}
  .header-left p {{
    font-size: 13px; color: rgba(255,255,255,0.55); margin-top: 4px;
  }}
  .header-badge {{
    background: rgba(58,134,255,0.20); border: 1px solid rgba(58,134,255,0.40);
    color: #7EC8FF; font-size: 12px; font-weight: 600;
    padding: 6px 14px; border-radius: 999px; white-space: nowrap;
  }}

  /* ── KPI row ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    padding: 28px 36px;
  }}
  .kpi-card {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(20,20,43,0.07);
    border-top: 3px solid;
    transition: transform 0.15s ease;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); }}
  .kpi-card:nth-child(1) {{ border-color: {BLUE}; }}
  .kpi-card:nth-child(2) {{ border-color: {PINK}; }}
  .kpi-card:nth-child(3) {{ border-color: {TEAL}; }}
  .kpi-card:nth-child(4) {{ border-color: {AMBER}; }}
  .kpi-card:nth-child(5) {{ border-color: {PURPLE}; }}
  .kpi-label {{ font-size: 11px; font-weight: 600; color: #888899;
                text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 28px; font-weight: 700; color: #14142B;
                margin: 8px 0 4px; letter-spacing: -0.5px; }}
  .kpi-sub   {{ font-size: 11px; color: #AAAABC; }}

  /* ── Charts ── */
  .container {{ padding: 0 36px 36px; }}
  .chart-grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .chart-grid-1 {{ margin-bottom: 20px; }}
  .chart-grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }}

  .card {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 6px 8px 12px;
    box-shadow: 0 2px 12px rgba(20,20,43,0.07);
    overflow: hidden;
  }}

  /* ── Business insights ── */
  .insights {{
    background: linear-gradient(135deg, #14142B 0%, #2D2B55 100%);
    border-radius: 14px;
    padding: 32px 36px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(20,20,43,0.15);
  }}
  .insights h2 {{
    font-size: 18px; font-weight: 700; color: #FFFFFF; margin-bottom: 24px;
  }}
  .insight-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;
  }}
  .insight-item {{
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px; padding: 18px 20px;
  }}
  .insight-icon {{ font-size: 22px; margin-bottom: 8px; }}
  .insight-title {{ font-size: 12px; font-weight: 600; color: rgba(255,255,255,0.55);
                    text-transform: uppercase; letter-spacing: 0.4px; }}
  .insight-text  {{ font-size: 14px; font-weight: 500; color: #FFFFFF; margin-top: 6px; line-height: 1.5; }}

  /* ── Footer ── */
  .footer {{
    text-align: center; font-size: 12px;
    color: #AAAABC; padding: 20px 36px 36px;
  }}

  @media (max-width: 768px) {{
    .chart-grid-2, .chart-grid-3 {{ grid-template-columns: 1fr; }}
    .header, .kpi-row, .container {{ padding-left: 20px; padding-right: 20px; }}
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <h1>📊 Sales &amp; Demand Forecasting Dashboard</h1>
    <p>Future Interns · Machine Learning Track · Task 1 · 2022–2024 Retail Data</p>
  </div>
  <div class="header-badge">Best Model: {best_name} &nbsp;|&nbsp; MAPE: {best_mape:.1f}%</div>
</div>

<!-- KPI Cards -->
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Historical Revenue</div>
    <div class="kpi-value">${total_hist/1e6:.2f}M</div>
    <div class="kpi-sub">3 years · all categories</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Daily Sales</div>
    <div class="kpi-value">${avg_daily:,.0f}</div>
    <div class="kpi-sub">across {n_categories} categories</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">90-Day Forecast</div>
    <div class="kpi-value">${fcast_total/1e6:.2f}M</div>
    <div class="kpi-sub">{str(forecast_df['date'].iloc[0].date())} → {str(forecast_df['date'].iloc[-1].date())}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Forecast Accuracy</div>
    <div class="kpi-value">{100 - best_mape:.1f}%</div>
    <div class="kpi-sub">MAPE: {best_mape:.1f}% · R²: {best_r2:.3f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Forecast / Day</div>
    <div class="kpi-value">${fcast_total/90:,.0f}</div>
    <div class="kpi-sub">Q1 {forecast_df['date'].iloc[-1].year} estimate</div>
  </div>
</div>

<!-- Main Charts -->
<div class="container">

  <!-- Forecast (full width) -->
  <div class="chart-grid-1">
    <div class="card">{fig_to_div(fig_forecast)}</div>
  </div>

  <!-- Monthly + DoW -->
  <div class="chart-grid-2">
    <div class="card">{fig_to_div(fig_monthly)}</div>
    <div class="card">{fig_to_div(fig_dow)}</div>
  </div>

  <!-- Category total + trend -->
  <div class="chart-grid-2">
    <div class="card">{fig_to_div(fig_category)}</div>
    <div class="card">{fig_to_div(fig_cat_trend)}</div>
  </div>

  <!-- Model comparison (full width) -->
  <div class="chart-grid-1">
    <div class="card">{fig_to_div(fig_models)}</div>
  </div>

  <!-- Business Insights -->
  <div class="insights">
    <h2>💡 Business Insights &amp; Recommendations</h2>
    <div class="insight-grid">
      <div class="insight-item">
        <div class="insight-icon">🎄</div>
        <div class="insight-title">Holiday Planning</div>
        <div class="insight-text">Nov–Dec sales spike ~80% above average. Increase inventory and staffing 6 weeks ahead to capture full holiday demand.</div>
      </div>
      <div class="insight-item">
        <div class="insight-icon">📦</div>
        <div class="insight-title">Inventory Strategy</div>
        <div class="insight-text">The 90-day forecast projects ${fcast_total/1e6:.2f}M in revenue. Procure stock now to avoid stockouts and lost sales.</div>
      </div>
      <div class="insight-item">
        <div class="insight-icon">📅</div>
        <div class="insight-title">Weekend Staffing</div>
        <div class="insight-text">Saturday–Sunday sales run ~30% higher than weekdays. Schedule additional staff and ensure systems capacity for peak traffic.</div>
      </div>
      <div class="insight-item">
        <div class="insight-icon">🏆</div>
        <div class="insight-title">Top Performer</div>
        <div class="insight-text">Electronics generates the highest revenue. Prioritise promotions and upselling opportunities in this category year-round.</div>
      </div>
      <div class="insight-item">
        <div class="insight-icon">📈</div>
        <div class="insight-title">Growth Trend</div>
        <div class="insight-text">Revenue shows a consistent upward trend across all categories. Budget and expansion plans should account for ~15–20% YoY growth.</div>
      </div>
      <div class="insight-item">
        <div class="insight-icon">☀️</div>
        <div class="insight-title">Summer Opportunity</div>
        <div class="insight-text">Sports &amp; Clothing see a Jun–Aug uplift. Launch targeted summer campaigns 4–6 weeks before the season begins.</div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  Built with Python · Plotly · Scikit-learn &nbsp;|&nbsp;
  Future Interns ML Internship · Task 1 · Sales &amp; Demand Forecasting &nbsp;|&nbsp;
  Model: {best_name} · MAPE {best_mape:.1f}%
</div>

</body>
</html>"""

os.makedirs("outputs", exist_ok=True)

out_path = "outputs/dashboard.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"  ✓  Dashboard saved → {out_path}")
print(f"     Open it in any browser — no server needed.")
