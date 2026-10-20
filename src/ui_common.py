"""Shared paths, text and sidebar elements used by every page of the app."""
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "resources" / "sepsis_survival_dataset"

COHORT_FILES = {
    "Primary cohort": DATASET_DIR / "1_primary_cohort.csv",
    "Study cohort (Sepsis-3)": DATASET_DIR / "3_study_cohort.csv",
    "Validation cohort (South Korea)": DATASET_DIR / "2_validation_cohort.csv",
}

DATASET_CITATION = (
    "Chicco, D. & Jurman, G. (2020). Sepsis Survival Minimal Clinical Records "
    "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53C8N"
)
DATASET_URL = (
    "https://archive.ics.uci.edu/dataset/827/"
    "sepsis+survival+minimal+clinical+records"
)
DISCLAIMER = (
    "Coursework prototype (PROHI HT26, Group 2). Not a validated clinical tool."
)


def sidebar_footer() -> None:
    """Render the dataset citation and disclaimer at the bottom of the sidebar."""
    with st.sidebar:
        st.divider()
        st.caption(f"**Data:** UCI dataset 827 · [source]({DATASET_URL})")
        st.caption(DISCLAIMER)
