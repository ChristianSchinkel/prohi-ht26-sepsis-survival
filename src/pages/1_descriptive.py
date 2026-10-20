"""Descriptive view of the sepsis survival cohorts."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "resources" / "sepsis_survival_dataset"
DATASETS = {
    "Primary cohort": DATASET_DIR / "1_primary_cohort.csv",
    "Validation cohort": DATASET_DIR / "2_validation_cohort.csv",
    "Study cohort": DATASET_DIR / "3_study_cohort.csv",
}

COLUMN_LABELS = {
    "age_years": "Age (years)",
    "sex_0male_1female": "Sex",
    "episode_number": "Episode number",
    "hospital_outcome_1alive_0dead": "Hospital outcome",
}


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    """Load a cohort from disk and provide human-readable categorical labels."""
    data = pd.read_csv(path)
    data["sex"] = data["sex_0male_1female"].map({0: "Male", 1: "Female"})
    data["hospital_outcome"] = data["hospital_outcome_1alive_0dead"].map(
        {0: "Died", 1: "Alive"}
    )
    return data


def _format_percentage(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    """Render the descriptive decision-making page."""
    st.set_page_config(
        page_title="Descriptive analysis | Sepsis Survival",
        page_icon="📊",
        layout="wide",
    )
    st.title("Descriptive analysis")
    st.markdown(
        "Explore the three sepsis survival cohorts using descriptive statistics. "
        "This page supports the **descriptive** family of decision making by "
        "summarising what is present in the observed data."
    )

    selected_cohort = st.sidebar.selectbox("Dataset", list(DATASETS))
    data_path = DATASETS[selected_cohort]
    if not data_path.exists():
        st.error(f"Dataset not found: `{data_path}`")
        return

    data = load_dataset(str(data_path))
    st.sidebar.caption(f"{len(data):,} records loaded")

    st.subheader(selected_cohort)
    st.caption(
        "Source variables use the original dataset encoding: sex (0 = male, "
        "1 = female) and hospital outcome (0 = died, 1 = alive)."
    )

    age_min = int(data["age_years"].min())
    age_max = int(data["age_years"].max())
    age_range = st.sidebar.slider(
        "Age range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )
    selected_sexes = st.sidebar.multiselect(
        "Sex",
        options=["Male", "Female"],
        default=["Male", "Female"],
    )
    selected_outcomes = st.sidebar.multiselect(
        "Hospital outcome",
        options=["Alive", "Died"],
        default=["Alive", "Died"],
    )

    filtered = data[
        data["age_years"].between(*age_range)
        & data["sex"].isin(selected_sexes)
        & data["hospital_outcome"].isin(selected_outcomes)
    ]

    total_records = len(filtered)
    survival_rate = (
        filtered["hospital_outcome_1alive_0dead"].mean()
        if total_records
        else 0.0
    )
    median_age = filtered["age_years"].median() if total_records else 0.0
    repeat_episode_rate = (
        (filtered["episode_number"] > 1).mean() if total_records else 0.0
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Records", f"{total_records:,}")
    metric_columns[1].metric("Survived", _format_percentage(survival_rate))
    metric_columns[2].metric(
        "Median age",
        f"{median_age:.1f} years" if total_records else "—",
    )
    metric_columns[3].metric(
        "Repeat episodes",
        _format_percentage(repeat_episode_rate),
    )

    if not total_records:
        st.warning("No records match the selected filters.")
        return

    st.divider()
    distribution_column, outcome_column = st.columns(2)
    with distribution_column:
        st.subheader("Age distribution")
        st.bar_chart(
            filtered["age_years"].value_counts().sort_index(),
            x_label="Age (years)",
            y_label="Records",
        )
    with outcome_column:
        st.subheader("Hospital outcome by sex")
        outcome_by_sex = pd.crosstab(filtered["sex"], filtered["hospital_outcome"])
        st.bar_chart(outcome_by_sex, y_label="Records")

    st.subheader("Numerical value distributions")
    numerical_columns = [
        "age_years",
        "episode_number",
        "hospital_outcome_1alive_0dead",
    ]
    boxplot_data = filtered[numerical_columns].rename(columns=COLUMN_LABELS).melt(
        var_name="Variable",
        value_name="Value",
    )
    boxplot = (
        alt.Chart(boxplot_data)
        .mark_boxplot(size=55)
        .encode(
            x=alt.X("Variable:N", title=None, sort=list(COLUMN_LABELS.values())),
            y=alt.Y("Value:Q", title="Value"),
            tooltip=[
                alt.Tooltip("Variable:N", title="Variable"),
                alt.Tooltip("Value:Q", title="Value"),
            ],
        )
        .properties(height=350)
    )
    st.altair_chart(boxplot, use_container_width=True)
    st.caption(
        "The hospital outcome is shown using its original numeric encoding "
        "(0 = died, 1 = alive)."
    )

    st.subheader("Numerical summary")
    summary = filtered[
        ["age_years", "episode_number", "hospital_outcome_1alive_0dead"]
    ].describe().T
    summary = summary.rename(index=COLUMN_LABELS)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Filtered records")
    display_columns = [
        "age_years",
        "sex",
        "episode_number",
        "hospital_outcome",
    ]
    display_data = filtered[display_columns].rename(columns=COLUMN_LABELS)
    st.dataframe(display_data, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data (CSV)",
        data=display_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_cohort.lower().replace(' ', '_')}_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
