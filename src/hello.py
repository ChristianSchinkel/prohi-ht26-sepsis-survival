"""This is the entry point for the Streamlit application."""
import streamlit as st

BODY = """
    # Sepsis Survival Dashboard
        Dataset: Sepsis Survival Minimal Clinical Records,
        UCI Machine Learning Repository dataset 827.
        It contains 110,204 hospital admissions
        from 84,811 patients in Norway during 2011-2012.
    **👈 Select a card from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Chicco, D. & Jurman, G. (2020). Sepsis Survival Minimal Clinical Records [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53C8N. [dataset](https://doi.org/10.24432/C53C8N)
    """


def main() -> None:
    """Render the Streamlit application entry point."""
    st.set_page_config(page_title="Sepsis Survival", page_icon="🩺")
    st.title("# Sepsis Survival")
    st.sidebar.success("Select a page above.")
    st.write("## Welcome to the sepsis survival prediction app.")
    st.markdown(BODY)


if __name__ == "__main__":
    main()
