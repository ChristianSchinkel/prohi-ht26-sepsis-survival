"""Entry point and router for the Streamlit application.

`src/main.py` launches this file. Navigation is declared here with
`st.navigation`, so the sidebar order, titles and icons are controlled in one
place instead of being derived from the file names in `src/pages/`.
"""
import streamlit as st

from ui_common import sidebar_footer

PAGES = {
    "Overview": [
        st.Page("home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Analytics": [
        st.Page(
            "pages/1_descriptive.py",
            title="1 · Descriptive",
            icon=":material/bar_chart:",
            url_path="descriptive",
        ),
        st.Page(
            "pages/2_diagnostic.py",
            title="2 · Diagnostic",
            icon=":material/troubleshoot:",
            url_path="diagnostic",
        ),
        st.Page(
            "pages/3_predictive.py",
            title="3 · Predictive",
            icon=":material/query_stats:",
            url_path="predictive",
        ),
        st.Page(
            "pages/4_prescriptive.py",
            title="4 · Prescriptive",
            icon=":material/bed:",
            url_path="prescriptive",
        ),
    ],
}


def main() -> None:
    """Configure the app, render shared sidebar elements and run the selected page."""
    st.set_page_config(
        page_title="Sepsis Survival · PROHI Group 2",
        page_icon=":material/monitor_heart:",
        layout="wide",
    )
    navigation = st.navigation(PAGES, position="sidebar", expanded=True)
    sidebar_footer()
    navigation.run()


if __name__ == "__main__":
    main()
