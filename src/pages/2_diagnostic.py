"""Diagnostic view: why did patients die? (analysis by Anupom Das).

The analysis itself lives in `src/diagnostic/analysis.py`; this page only
loads the selected cohort, runs the analysis once (cached) and displays it.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diagnostic.analysis import run_diagnostic_analysis  # noqa: E402
from ui_common import COHORT_FILES  # noqa: E402

# The regression needs enough deaths in every episode category, so the small
# South Korean validation cohort (137 admissions) is not offered here.
DIAGNOSTIC_COHORTS = {
    name: path for name, path in COHORT_FILES.items() if "Validation" not in name
}
EXPORT_TABLES = [
    "quality_report",
    "descriptive_summary",
    "age_summary",
    "age_tests",
    "sex_table",
    "sex_percent",
    "sex_tests",
    "episode_table",
    "episode_rates",
    "episode_tests",
    "odds_ratios",
]


@st.cache_resource(show_spinner="Running diagnostic analysis…")
def analyse(cohort_name: str) -> dict:
    """Load a cohort and run the diagnostic analysis (cached per cohort)."""
    data = pd.read_csv(DIAGNOSTIC_COHORTS[cohort_name])
    return run_diagnostic_analysis(data)


def results_workbook(results: dict) -> bytes:
    """Write all result tables and metrics to an Excel workbook in memory."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name in EXPORT_TABLES:
            results[name].to_excel(writer, sheet_name=name[:31])
        pd.Series(results["metrics"], name="value").to_frame().to_excel(
            writer, sheet_name="metrics"
        )
    return buffer.getvalue()


def render() -> None:
    """Render the diagnostic page."""
    st.title("Diagnostic analysis")
    st.markdown(
        "Which patient characteristics are associated with in-hospital death? "
        "Data quality checks, bivariate tests and a multivariable logistic "
        "regression (outcome: death = 1)."
    )

    cohort_name = st.sidebar.selectbox("Dataset", list(DIAGNOSTIC_COHORTS))
    try:
        results = analyse(cohort_name)
    except (OSError, ValueError) as error:
        st.error(f"Analysis could not be completed: {error}")
        return

    metrics = results["metrics"]
    st.sidebar.caption(f"{len(results['cleaned_data']):,} records loaded")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Admissions in model", f"{metrics['n_observations']:,}")
    summary_columns[1].metric("Deaths", f"{metrics['deaths']:,}")
    summary_columns[2].metric(
        "Observed mortality", f"{metrics['observed_mortality']:.1%}"
    )
    summary_columns[3].metric(
        "In-sample ROC-AUC", f"{metrics['roc_auc_in_sample']:.3f}"
    )

    tabs = st.tabs(
        ["Data quality", "Descriptive", "Bivariate", "Regression", "Evaluation"]
    )
    figures = results["figures"]

    with tabs[0]:
        st.subheader("Quality report")
        st.dataframe(results["quality_report"], hide_index=True, use_container_width=True)
        cleaned = results["cleaned_data"]
        missing = pd.DataFrame(
            {
                "Missing count": cleaned.isna().sum(),
                "Missing (%)": cleaned.isna().mean() * 100,
            }
        ).sort_values("Missing (%)", ascending=False)
        st.subheader("Missing values after validation")
        st.dataframe(missing.round(2), use_container_width=True)

    with tabs[1]:
        st.subheader("Numeric summary")
        st.dataframe(results["descriptive_summary"], use_container_width=True)
        left, right = st.columns(2)
        left.pyplot(figures["age_histogram"], clear_figure=False)
        right.pyplot(figures["age_boxplot"], clear_figure=False)
        st.pyplot(figures["categorical_counts"], clear_figure=False)

    with tabs[2]:
        st.subheader("Age versus hospital outcome")
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(results["age_summary"].round(2), use_container_width=True)
            st.caption("Welch's t-test and Mann–Whitney U test")
            st.dataframe(results["age_tests"].round(6), hide_index=True, use_container_width=True)
        right.pyplot(figures["age_by_outcome"], clear_figure=False)

        st.divider()
        st.subheader("Sex versus hospital outcome")
        left, right = st.columns([3, 2])
        with left:
            st.caption("Observed counts and row percentages (%)")
            st.dataframe(results["sex_table"], use_container_width=True)
            st.dataframe(results["sex_percent"].round(2), use_container_width=True)
            st.caption("Chi-square test")
            st.dataframe(results["sex_tests"].round(6), hide_index=True, use_container_width=True)
        right.pyplot(figures["sex_outcome"], clear_figure=False)

        st.divider()
        st.subheader("Episode number versus hospital outcome")
        left, right = st.columns([3, 2])
        with left:
            st.caption("Mortality by episode")
            st.dataframe(results["episode_rates"].round(2), use_container_width=True)
            st.caption("Chi-square test")
            st.dataframe(results["episode_tests"].round(6), hide_index=True, use_container_width=True)
        right.pyplot(figures["episode_mortality"], clear_figure=False)

    with tabs[3]:
        st.caption(
            "Outcome: death (1 = dead, 0 = alive). Episode 1 is the reference "
            "category. The model adjusts for age, sex and episode number."
        )
        st.subheader("Adjusted odds ratios (95% CI)")
        st.dataframe(results["odds_ratios"].round(4), use_container_width=True)
        st.pyplot(figures["forest_plot"], clear_figure=False)
        with st.expander("Full logistic regression summary"):
            st.text(results["model"].summary().as_text())

    with tabs[4]:
        left, right = st.columns(2)
        left.metric("In-sample ROC-AUC", f"{metrics['roc_auc_in_sample']:.3f}")
        right.metric("In-sample Brier score", f"{metrics['brier_score_in_sample']:.3f}")
        left.pyplot(figures["roc_curve"], clear_figure=False)
        right.pyplot(figures["calibration_plot"], clear_figure=False)
        st.warning(
            "These metrics are computed on the observations used to fit the "
            "model (in-sample) and may be optimistic. Held-out and external "
            "performance is reported on the Predictive page."
        )

    st.divider()
    st.download_button(
        "Download all diagnostic tables (Excel)",
        data=results_workbook(results),
        file_name="sepsis_diagnostic_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
    )


render()
