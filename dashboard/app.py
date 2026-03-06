import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

from src.data_loader import load_data, clean_data
from src.model import predict, FEATURE_COLS

# ─── CONFIG ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

PALETTE = ["#1F4E79", "#2E75B6", "#4BACC6", "#A9CCE3", "#D6E4F0"]
TEMPLATE = "plotly_white"

INSIGHTS = [
    "💡 Revenue is growing but heavily seasonal — every year shows a sharp Q4 spike followed by a steep January drop. Consider mid-year promotions in Q2 to smooth revenue distribution.",
    "💡 The Canon imageCLASS 2200 Copier generates ~$60k in revenue, nearly double the second-ranked product. Monitor its stock levels closely — it's a single-SKU risk.",
    "💡 West leads in both sales and profit. Central has ~$500k in sales but only ~$30k in profit — the worst profit-to-sales ratio of all regions. Audit discount rates in Central.",
    "💡 Tables is the only sub-category operating at a loss despite ~$207k in revenue. Every Tables sale is actively destroying profit. Review pricing and discount strategy urgently.",
    "💡 Discounts above 40% almost always produce negative profit. The discount strategy is not driving enough volume to offset margin loss. Consider capping discounts at 20%.",
    "💡 Technology spikes hardest in Q4 (~$50k in Nov 2017). Ensure Technology inventory is fully stocked by October and plan targeted November campaigns for that category.",
]

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    df_raw = load_data("data/superstore.csv")
    df = clean_data(df_raw)
    return df


@st.cache_data
def get_monthly(df):
    monthly = (
        df.groupby(["year", "month"])
        .agg(
            sales=("sales", "sum"),
            orders=("order_id", "nunique"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum")
        )
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly = monthly.sort_values("date").reset_index(drop=True)
    return monthly


@st.cache_resource
def get_model():
    return joblib.load("data/model.pkl")


@st.cache_data
def get_forecasts():
    return pd.read_csv("data/forecasts.csv", parse_dates=["date"])


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

def build_sidebar(df):
    st.sidebar.image("https://img.icons8.com/color/96/combo-chart--v2.png", width=60)
    st.sidebar.title("Sales Analytics")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Sales Trends", "Product & Category", "Regional Analysis", "Demand Forecast"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    # Date range
    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Region
    all_regions = sorted(df["region"].unique().tolist())
    selected_regions = st.sidebar.multiselect(
        "Region",
        options=all_regions,
        default=all_regions
    )

    # Category
    all_categories = sorted(df["category"].unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Category",
        options=all_categories,
        default=all_categories
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Superstore Dataset | 2014–2017")

    return page, date_range, selected_regions, selected_categories


def apply_filters(df, date_range, regions, categories):
    filtered = df.copy()

    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[
            (filtered["order_date"] >= start) &
            (filtered["order_date"] <= end)
        ]

    if regions:
        filtered = filtered[filtered["region"].isin(regions)]

    if categories:
        filtered = filtered[filtered["category"].isin(categories)]

    return filtered


# ─── PAGE 1: OVERVIEW ─────────────────────────────────────────────────────────

def show_overview(df):
    st.title("📊 Sales Overview")
    st.markdown("High-level KPIs and key business insights from the Superstore dataset.")
    st.markdown("---")

    # KPI cards
    total_revenue = df["sales"].sum()
    total_orders = df["order_id"].nunique()
    total_profit = df["profit"].sum()
    avg_margin = df["profit_margin"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label="Total Revenue",
        value=f"${total_revenue:,.0f}"
    )
    c2.metric(
        label="Total Orders",
        value=f"{total_orders:,}"
    )
    c3.metric(
        label="Total Profit",
        value=f"${total_profit:,.0f}"
    )
    c4.metric(
        label="Avg Profit Margin",
        value=f"{avg_margin:.1f}%"
    )

    st.markdown("---")

    # Revenue over time mini chart
    monthly = (
        df.groupby(["year", "month"])["sales"]
        .sum()
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly = monthly.sort_values("date")

    fig = px.area(
        monthly,
        x="date",
        y="sales",
        title="Revenue Over Time",
        labels={"date": "Month", "sales": "Total Sales ($)"},
        template=TEMPLATE,
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(line_color="#1F4E79", fillcolor="rgba(31, 78, 121, 0.15)")
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Business insights
    st.markdown("---")
    with st.expander("💼 Key Business Insights", expanded=True):
        for insight in INSIGHTS:
            st.markdown(f"- {insight}")


# ─── PAGE 2: SALES TRENDS ─────────────────────────────────────────────────────

def show_trends(df):
    st.title("📈 Sales Trends")
    st.markdown("Monthly and quarterly revenue patterns from 2014 to 2017.")
    st.markdown("---")

    # Monthly trend
    monthly = (
        df.groupby(["year", "month"])["sales"]
        .sum()
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly = monthly.sort_values("date")

    fig1 = px.line(
        monthly,
        x="date",
        y="sales",
        title="Monthly Sales Revenue",
        labels={"date": "Month", "sales": "Total Sales ($)"},
        template=TEMPLATE,
        color_discrete_sequence=PALETTE,
        markers=True
    )
    fig1.update_layout(hovermode="x unified", height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # Quarterly breakdown
    quarterly = (
        df.groupby(["year", "quarter"])["sales"]
        .sum()
        .reset_index()
    )
    quarterly["period"] = "Q" + quarterly["quarter"].astype(str) + " " + quarterly["year"].astype(str)

    fig2 = px.bar(
        quarterly,
        x="period",
        y="sales",
        color="year",
        title="Quarterly Revenue by Year",
        labels={"sales": "Total Sales ($)", "period": "Quarter", "year": "Year"},
        template=TEMPLATE,
        color_discrete_sequence=PALETTE,
        barmode="group"
    )
    fig2.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)


# ─── PAGE 3: PRODUCT & CATEGORY ───────────────────────────────────────────────

def show_products(df):
    st.title("🛍️ Product & Category Analysis")
    st.markdown("Top performing products and profitability breakdown by sub-category.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        top_products = (
            df.groupby("product_name")["sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig1 = px.bar(
            top_products,
            x="sales",
            y="product_name",
            orientation="h",
            title="Top 10 Products by Revenue",
            labels={"sales": "Total Sales ($)", "product_name": "Product"},
            template=TEMPLATE,
            color="sales",
            color_continuous_scale="Blues"
        )
        fig1.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            height=450
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        subcat = (
            df.groupby(["category", "sub_category"])[["sales", "profit"]]
            .sum()
            .reset_index()
        )

        fig2 = px.treemap(
            subcat,
            path=["category", "sub_category"],
            values="sales",
            color="profit",
            color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
            color_continuous_midpoint=0,
            title="Revenue by Sub-Category (colour = profit)",
            template=TEMPLATE
        )
        fig2.update_traces(textinfo="label+value")
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Discount vs profit scatter
    st.subheader("Discount Effect on Profit")
    fig3 = px.scatter(
        df,
        x="sales",
        y="profit",
        color="discount",
        color_continuous_scale="RdBu_r",
        title="Sales vs Profit — Coloured by Discount Level",
        labels={"sales": "Sales ($)", "profit": "Profit ($)", "discount": "Discount"},
        template=TEMPLATE,
        opacity=0.6,
        hover_data=["product_name", "sub_category", "region"]
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)


# ─── PAGE 4: REGIONAL ANALYSIS ────────────────────────────────────────────────

def show_regional(df):
    st.title("🗺️ Regional Analysis")
    st.markdown("Sales and profit performance broken down by region.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        region = (
            df.groupby("region")[["sales", "profit"]]
            .sum()
            .reset_index()
            .melt(id_vars="region", var_name="metric", value_name="value")
        )

        fig1 = px.bar(
            region,
            x="region",
            y="value",
            color="metric",
            barmode="group",
            title="Sales vs Profit by Region",
            labels={"value": "Amount ($)", "region": "Region", "metric": "Metric"},
            template=TEMPLATE,
            color_discrete_sequence=["#1F4E79", "#4BACC6"]
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        margin_by_region = (
            df.groupby("region")["profit_margin"]
            .mean()
            .reset_index()
            .sort_values("profit_margin", ascending=False)
        )

        fig2 = px.bar(
            margin_by_region,
            x="region",
            y="profit_margin",
            title="Average Profit Margin by Region (%)",
            labels={"profit_margin": "Avg Profit Margin (%)", "region": "Region"},
            template=TEMPLATE,
            color="profit_margin",
            color_continuous_scale="Blues"
        )
        fig2.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Category performance by region
    st.subheader("Sales by Region and Category")
    region_cat = (
        df.groupby(["region", "category"])["sales"]
        .sum()
        .reset_index()
    )

    fig3 = px.bar(
        region_cat,
        x="region",
        y="sales",
        color="category",
        title="Sales by Region and Category",
        labels={"sales": "Total Sales ($)", "region": "Region", "category": "Category"},
        template=TEMPLATE,
        color_discrete_sequence=PALETTE,
        barmode="stack"
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)


# ─── PAGE 5: DEMAND FORECAST ──────────────────────────────────────────────────

def show_forecast(df):
    st.title("🔮 Demand Forecast")
    st.markdown("XGBoost model trained on 2014–2016, evaluated on 2017, forecasting Jan–Mar 2018.")
    st.markdown("---")

    forecasts = get_forecasts()

    # Rebuild monthly for context
    monthly = get_monthly(df)

    # Rebuild features for actual vs predicted on test set
    from src.features import build_features
    featured = build_features(monthly)
    test = featured[featured["date"] >= "2017-01-01"]
    model = get_model()
    preds = predict(model, test[FEATURE_COLS])

    # Chart 1 — actual vs predicted
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=test["date"], y=test["sales"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#1F4E79", width=2),
        marker=dict(size=6)
    ))

    fig1.add_trace(go.Scatter(
        x=test["date"], y=preds,
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#4BACC6", width=2, dash="dash"),
        marker=dict(size=6)
    ))

    fig1.update_layout(
        title="Actual vs Predicted Monthly Sales — 2017 Test Period (R²=0.63)",
        xaxis_title="Month",
        yaxis_title="Total Sales ($)",
        template=TEMPLATE,
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # Chart 2 — forward forecast
    historical = monthly.tail(12)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=historical["date"], y=historical["sales"],
        mode="lines+markers",
        name="Historical Sales",
        line=dict(color="#1F4E79", width=2),
        marker=dict(size=5)
    ))

    fig2.add_trace(go.Scatter(
        x=forecasts["date"], y=forecasts["predicted_sales"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#4BACC6", width=2, dash="dash"),
        marker=dict(size=8, symbol="star")
    ))

    fig2.add_trace(go.Scatter(
        x=pd.concat([forecasts["date"], forecasts["date"][::-1]]),
        y=pd.concat([forecasts["upper_bound"], forecasts["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(75, 172, 198, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Band"
    ))

    fig2.update_layout(
        title="Sales Forecast — Jan to Mar 2018 (±15% confidence band)",
        xaxis_title="Month",
        yaxis_title="Total Sales ($)",
        template=TEMPLATE,
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Forecast table
    st.subheader("Forecast Values")
    display_forecasts = forecasts.copy()
    display_forecasts["date"] = display_forecasts["date"].dt.strftime("%B %Y")
    display_forecasts.columns = ["Month", "Predicted Sales ($)", "Lower Bound ($)", "Upper Bound ($)"]
    display_forecasts = display_forecasts.set_index("Month")

    for col in display_forecasts.columns:
        display_forecasts[col] = display_forecasts[col].apply(lambda x: f"${x:,.0f}")

    st.dataframe(display_forecasts, use_container_width=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    df = get_data()
    page, date_range, selected_regions, selected_categories = build_sidebar(df)
    filtered_df = apply_filters(df, date_range, selected_regions, selected_categories)

    if len(filtered_df) == 0:
        st.warning("No data matches the current filters. Please adjust the sidebar filters.")
        return

    if page == "Overview":
        show_overview(filtered_df)
    elif page == "Sales Trends":
        show_trends(filtered_df)
    elif page == "Product & Category":
        show_products(filtered_df)
    elif page == "Regional Analysis":
        show_regional(filtered_df)
    elif page == "Demand Forecast":
        show_forecast(filtered_df)


if __name__ == "__main__":
    main()