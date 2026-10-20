"""Home page: project overview and entry points to the four analytics pages."""
import pandas as pd
import streamlit as st

from ui_common import COHORT_FILES, DATASET_CITATION, DATASET_URL, DISCLAIMER

ANALYTICS = [
    {
        "page": "pages/1_descriptive.py",
        "label": "Descriptive",
        "icon": ":material/bar_chart:",
        "question": "What happened?",
        "summary": "Explore the three cohorts: age, sex, episode number and "
        "survival, with interactive filters and downloadable tables.",
    },
    {
        "page": "pages/2_diagnostic.py",
        "label": "Diagnostic",
        "icon": ":material/troubleshoot:",
        "question": "Why did it happen?",
        "summary": "Data quality checks, bivariate tests and a multivariable "
        "logistic regression of in-hospital death.",
    },
    {
        "page": "pages/3_predictive.py",
        "label": "Predictive",
        "icon": ":material/query_stats:",
        "question": "What is likely to happen?",
        "summary": "Frozen survival models trained on the Norwegian cohorts, "
        "with an untouched external test on the South Korean cohort.",
    },
    {
        "page": "pages/4_prescriptive.py",
        "label": "Prescriptive",
        "icon": ":material/bed:",
        "question": "What should we do?",
        "summary": "Decision-curve thresholds and a simulated allocation of "
        "scarce high-dependency (HDU) beds for the bed manager.",
    },
]

TEAM = [
    ("Christian Schinkel", "Product Owner", "Descriptive analysis"),
    ("Zoltan Kurhejja", "Scrum Master", "Report and presentation"),
    ("Anupom Das", "Developer", "Diagnostic analysis"),
    ("Tauska (Xuehai Wang)", "Developer", "Predictive analysis"),
    ("Kevin (Qizheng Zhao)", "Developer", "Prescriptive analysis"),
    ("Kai (Kaiwen Huang)", "Developer", "User interface and integration"),
    ("Evangelos Balanikas", "Developer", ""),
    ("Yuri (Yuanrui Zou)", "Developer", ""),
]


@st.cache_data
def cohort_overview() -> pd.DataFrame:
    """Return size and observed in-hospital mortality of each cohort."""
    rows = []
    for name, path in COHORT_FILES.items():
        data = pd.read_csv(path)
        rows.append(
            {
                "Cohort": name,
                "Admissions": len(data),
                "In-hospital deaths": int(
                    (data["hospital_outcome_1alive_0dead"] == 0).sum()
                ),
                "Mortality": (data["hospital_outcome_1alive_0dead"] == 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    """Render the home page."""
    st.title("Sepsis survival decision support")
    st.markdown(
        "A four-step analytics app built on the **Sepsis Survival Minimal "
        "Clinical Records** dataset: hospital admissions for sepsis in Norway "
        "(2011–2012) with an external cohort from South Korea. Each record has "
        "only age, sex, septic episode number and in-hospital outcome."
    )

    overview = cohort_overview()
    metric_columns = st.columns(len(overview))
    for column, row in zip(metric_columns, overview.itertuples(index=False)):
        column.metric(
            row.Cohort,
            f"{row.Admissions:,} admissions",
            f"{row.Mortality:.1%} in-hospital mortality",
            delta_color="off",
        )

    st.subheader("Analytics")
    card_columns = st.columns(2)
    for index, item in enumerate(ANALYTICS):
        with card_columns[index % 2].container(border=True):
            st.markdown(f"**{index + 1} · {item['label']}** — *{item['question']}*")
            st.write(item["summary"])
            st.page_link(
                item["page"],
                label=f"Open {item['label'].lower()} analysis",
                icon=item["icon"],
            )

    with st.expander("How to read this app"):
        st.markdown(
            "- Use the sidebar to move between pages; filters on each page "
            "appear in the sidebar below the navigation.\n"
            "- Sex is encoded 0 = male, 1 = female; hospital outcome is "
            "encoded 1 = alive, 0 = died, as in the source dataset.\n"
            "- The Norwegian primary and study cohorts overlap. Patient "
            "identifiers are not available, so repeated admissions of the same "
            "patient cannot be linked."
        )

    with st.expander("Team"):
        team = pd.DataFrame(TEAM, columns=["Member", "Role", "Contribution"])
        st.dataframe(team, hide_index=True, use_container_width=True)

    st.caption(f"{DATASET_CITATION} · [UCI page]({DATASET_URL})")
    st.caption(DISCLAIMER)


render()
