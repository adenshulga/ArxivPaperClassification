import streamlit as st


def validate_data(paper_name: str, paper_abstract: str) -> bool:
    if paper_name == "" and paper_abstract == "":
        st.error("Paper name or abstract are required")
        return False
    if paper_abstract == "":
        st.warning(
            "Without abstract, the performance of the model will be significantly worse"
        )
        return True
    return True
