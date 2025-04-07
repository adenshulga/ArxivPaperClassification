import streamlit as st
from src.app.setup_model import setup_pipeline, get_top_label_names, LabelScore
from src.app.tags_mapping import tags2full_name
from src.app.visualization import visualize_predicted_categories
from config.inference_config import InferenceConfig
from src.app.data_validation import validate_data

st.title("arXiv Paper Classifier")
st.markdown("Enter paper details to predict arXiv categories")

st.text_input("Enter paper name", key="paper_name")
st.text_area("Enter paper abstract", key="paper_abstract", height=250)

if st.button("Predict Categories", type="primary"):
    validate_data(st.session_state["paper_name"], st.session_state["paper_abstract"])
    with st.spinner("Analyzing paper..."):
        pipeline = setup_pipeline(InferenceConfig())
        scores: list[LabelScore] = pipeline(
            st.session_state["paper_name"] + " " + st.session_state["paper_abstract"],
            output_scores=True,
        )  # type: ignore

        top_labels = get_top_label_names(scores, tags2full_name, 0.95)

    visualize_predicted_categories(top_labels, scores, tags2full_name)
else:
    st.info("Enter paper details and click 'Predict Categories' to get predictions.")
