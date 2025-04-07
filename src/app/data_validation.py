import streamlit as st


def validate_data(paper_name: str, paper_abstract: str) -> None:
    if paper_name == "" or paper_abstract == "":
        st.error("Paper name or abstract are required")
        return
    if paper_abstract == "":
        st.warning(
            "Without abstract, the performance of the model will be significantly worse"
        )
        return
