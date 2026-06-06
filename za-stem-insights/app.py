"""
app.py  —  Learner 5 module
-----------------------------
Multi-page Streamlit dashboard for ZA-STEM Insights.
Visualises South African matric performance trends and ML predictions.

Contributor: SKYLearn-Innovation Learner 5
Mentor: Dingaan Mahlatse Machethe

Run:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from preprocess import load_clean
from analyse import (
    national_trend, province_summary,
    stem_vs_nonstem, subject_ranking, flag_at_risk,
)

st.set_page_config(
    page_title="ZA-STEM Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1F4E79"
ACCENT  = "#117A65"
WARN    = "#E74C3C"


@st.cache_data
def get_data() -> pd.DataFrame:
    try:
        return load_clean()
    except FileNotFoundError:
        st.error(
            "Data file not found. Run `python data/generate_sample_data.py` first.",
            icon="🚨",
        )
        st.stop()


def sidebar(df: pd.DataFrame) -> tuple:
    st.sidebar.image(
        "https://img.shields.io/badge/SKYLearn--Innovation-NPO-1F4E79?style=for-the-badge",
        use_container_width=True,
    )
    st.sidebar.title("ZA-STEM Insights")
    st.sidebar.caption("South African Matric Performance Analyser")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "📈 Trends", "🗺️ Provinces", "📚 Subjects", "⚠️ At-Risk Predictor"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    years = sorted(df["year"].unique())
    selected_year = st.sidebar.select_slider("Reference Year", options=years, value=years[-1])
    selected_type = st.sidebar.radio(
        "Subject Category", ["All", "STEM", "Non-STEM"], horizontal=True
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Built by** SKYLearn-Innovation learners  \n"
        "**Mentor:** [Dingaan Machethe](https://www.mahlontebe.org.za/portfolio)  \n"
        "**Org:** SKYLearn-Innovation NPO"
    )
    return page, selected_year, selected_type


def kpi_row(df: pd.DataFrame, year: int) -> None:
    yr = df[df["year"] == year]
    prev_yr = df[df["year"] == year - 1]

    national_rate = yr["passed"].sum() / yr["wrote"].sum() * 100
    prev_rate = prev_yr["passed"].sum() / prev_yr["wrote"].sum() * 100 if len(prev_yr) > 0 else national_rate
    stem_rate = yr[yr["subject_type"] == "STEM"]["pass_rate"].mean()
    at_risk_count = (yr["pass_rate"] < 60).sum()
    distinctions = yr["distinctions"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("National Pass Rate", f"{national_rate:.1f}%", f"{national_rate - prev_rate:+.1f}pp vs {year-1}")
    c2.metric("STEM Average Pass Rate", f"{stem_rate:.1f}%")
    c3.metric("At-Risk Combinations", int(at_risk_count), help="Province-subject pairs below 60%")
    c4.metric("Total Distinctions", f"{distinctions:,}")


def page_overview(df: pd.DataFrame, year: int) -> None:
    st.title("📊 ZA-STEM Insights")
    st.subheader(f"South African Matric Performance — {year} Snapshot")
    st.divider()
    kpi_row(df, year)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        trend = national_trend(df)
        fig = px.line(
            trend, x="year", y="national_pass_rate",
            title="National Pass Rate Trend (All Subjects)",
            markers=True, color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(yaxis_range=[40, 100], xaxis_title="Year", yaxis_title="Pass Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sv = stem_vs_nonstem(df)
        fig2 = px.line(
            sv, x="year", y="avg_pass_rate", color="subject_type",
            title="STEM vs Non-STEM Pass Rate Gap",
            markers=True, color_discrete_map={"STEM": PRIMARY, "Non-STEM": ACCENT},
        )
        fig2.update_layout(yaxis_range=[40, 100], xaxis_title="Year", yaxis_title="Avg Pass Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("At-Risk Province-Subject Pairs (Latest Year)")
    at_risk = flag_at_risk(df, threshold=60)
    if at_risk.empty:
        st.success("No combinations below 60% pass rate in the latest year.")
    else:
        st.dataframe(
            at_risk.style.background_gradient(subset=["pass_rate"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )


def page_trends(df: pd.DataFrame, year: int, subject_type: str) -> None:
    st.title("📈 National Trends")
    st.divider()

    stype = None if subject_type == "All" else subject_type
    trend = national_trend(df, subject_type=stype)

    fig = px.area(
        trend, x="year", y="national_pass_rate",
        title=f"National Pass Rate Over Time ({subject_type})",
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enrolment Over Time")
        fig2 = px.bar(
            trend, x="year", y="total_registered",
            color_discrete_sequence=[ACCENT],
            title="Total Learners Registered",
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        sv = stem_vs_nonstem(df)
        pivot = sv.pivot(index="year", columns="subject_type", values="avg_pass_rate")
        pivot["gap"] = pivot.get("Non-STEM", 0) - pivot.get("STEM", 0)
        pivot = pivot.reset_index()
        fig3 = px.bar(
            pivot, x="year", y="gap",
            title="Non-STEM minus STEM Pass Rate Gap (pp)",
            color_discrete_sequence=[WARN],
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Year-by-Year Data Table")
    st.dataframe(trend, use_container_width=True, hide_index=True)


def page_provinces(df: pd.DataFrame, year: int) -> None:
    st.title("🗺️ Provincial Performance")
    st.divider()

    prov = province_summary(df, year=year)

    fig = px.bar(
        prov.sort_values("avg_pass_rate"),
        x="avg_pass_rate", y="province",
        orientation="h",
        title=f"Average Pass Rate by Province — {year}",
        color="avg_pass_rate",
        color_continuous_scale="RdYlGn",
        range_color=[40, 100],
        labels={"avg_pass_rate": "Avg Pass Rate (%)", "province": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    all_prov = province_summary(df)

    prov_options = sorted(df["province"].unique())
    selected_provs = st.multiselect("Compare Provinces", prov_options, default=prov_options[:3])

    if selected_provs:
        filtered = df[df["province"].isin(selected_provs)]
        trend_p = (
            filtered.groupby(["year", "province"])
            .agg(avg_pass_rate=("pass_rate", "mean"))
            .reset_index()
        )
        fig2 = px.line(
            trend_p, x="year", y="avg_pass_rate", color="province",
            title="Province Pass Rate Trends", markers=True,
        )
        fig2.update_layout(yaxis_range=[40, 100])
        st.plotly_chart(fig2, use_container_width=True)


def page_subjects(df: pd.DataFrame, year: int, subject_type: str) -> None:
    st.title("📚 Subject Performance")
    st.divider()

    ranking = subject_ranking(df, year=year)
    stype = None if subject_type == "All" else subject_type
    if stype:
        ranking = ranking[ranking["subject_type"] == stype]

    fig = px.bar(
        ranking.sort_values("avg_pass_rate"),
        x="avg_pass_rate", y="subject",
        orientation="h",
        color="subject_type",
        title=f"Subject Pass Rate Ranking — {year}",
        color_discrete_map={"STEM": PRIMARY, "Non-STEM": ACCENT},
        labels={"avg_pass_rate": "Avg Pass Rate (%)", "subject": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    subject_list = sorted(df["subject"].unique())
    chosen = st.selectbox("Deep-dive: select a subject", subject_list)
    subj_df = df[df["subject"] == chosen]
    trend_s = (
        subj_df.groupby("year")
        .agg(avg_pass_rate=("pass_rate", "mean"))
        .reset_index()
    )
    fig2 = px.line(
        trend_s, x="year", y="avg_pass_rate",
        title=f"{chosen} — National Average Pass Rate Over Time",
        markers=True, color_discrete_sequence=[PRIMARY],
    )
    fig2.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader(f"{chosen} — Province Breakdown ({year})")
    subj_year = subj_df[subj_df["year"] == year][
        ["province", "pass_rate", "registered", "passed", "distinctions"]
    ].sort_values("pass_rate", ascending=False)
    st.dataframe(
        subj_year.style.background_gradient(subset=["pass_rate"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )


def page_predictor(df: pd.DataFrame) -> None:
    st.title("⚠️ At-Risk Predictor")
    st.caption("ML model (RandomForest) trained on 10 years of matric data")
    st.divider()

    try:
        from model import load_model, predict_next_year, feature_importance, train, build_features
        model_loaded = True
    except Exception:
        model_loaded = False

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Predict Next Year")
        province_options = sorted(df["province_code"].unique())
        subject_options  = sorted(df["subject"].unique())

        pcode   = st.selectbox("Province", province_options)
        subject = st.selectbox("Subject", subject_options)
        s_type  = df[df["subject"] == subject]["subject_type"].iloc[0]
        st.caption(f"Subject type: **{s_type}**")

        next_year  = int(df["year"].max()) + 1
        registered = st.number_input("Estimated Registered Learners", 500, 50_000, 5_000, step=100)
        absent_pct = st.slider("Expected Absentee Rate (%)", 0.0, 15.0, 3.0, step=0.5)

        predict_btn = st.button("Predict Risk", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            if not model_loaded:
                st.info("Training model now (first run may take a few seconds)...")
                result = train(df)
                clf, encoders = result["model"], result["encoders"]
                st.success(
                    f"Model trained — Accuracy: {result['metrics']['accuracy']:.1%} | "
                    f"ROC-AUC: {result['metrics']['roc_auc']:.3f}"
                )
            else:
                clf, encoders = load_model()

            try:
                pred = predict_next_year(
                    clf, encoders, pcode, subject, s_type, int(registered), float(absent_pct), next_year
                )
                risk_pct = pred["risk_probability"]
                color = WARN if pred["at_risk"] else ACCENT
                st.markdown(f"### Prediction for {next_year}")
                st.markdown(
                    f"<h2 style='color:{color}'>{pred['label']}</h2>",
                    unsafe_allow_html=True,
                )
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    title={"text": "Risk Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 40],  "color": "#d5f5e3"},
                            {"range": [40, 60], "color": "#fdebd0"},
                            {"range": [60, 100],"color": "#fadbd8"},
                        ],
                    },
                ))
                gauge.update_layout(height=300)
                st.plotly_chart(gauge, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    st.divider()
    st.subheader("Current At-Risk Province-Subject Pairs")
    at_risk_df = flag_at_risk(df, threshold=60)
    if at_risk_df.empty:
        st.success("No combinations below 60% in the latest year.")
    else:
        fig = px.scatter(
            at_risk_df,
            x="pass_rate", y="province",
            color="subject_type",
            size="registered",
            hover_data=["subject", "passed"],
            title="At-Risk Combinations (pass rate < 60%)",
            color_discrete_map={"STEM": PRIMARY, "Non-STEM": ACCENT},
        )
        fig.add_vline(x=60, line_dash="dash", line_color=WARN)
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    df = get_data()
    page, selected_year, selected_type = sidebar(df)

    if page == "🏠 Overview":
        page_overview(df, selected_year)
    elif page == "📈 Trends":
        page_trends(df, selected_year, selected_type)
    elif page == "🗺️ Provinces":
        page_provinces(df, selected_year)
    elif page == "📚 Subjects":
        page_subjects(df, selected_year, selected_type)
    elif page == "⚠️ At-Risk Predictor":
        page_predictor(df)


if __name__ == "__main__":
    main()
