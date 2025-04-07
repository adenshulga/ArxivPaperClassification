import streamlit as st
from src.app.setup_model import LabelScore
from typing import Dict


def visualize_predicted_categories(
    top_labels: list[LabelScore],
    scores: list[LabelScore],
    label_to_name_mapping: Dict[str, str],
):
    """
    Visualize the predicted categories in a streamlit app

    Args:
        top_labels: List of top labels to display
        scores: All scores from the model
        label_to_name_mapping: Mapping from label codes to full names
    """
    st.subheader("Predicted Categories")

    for i, label in enumerate(top_labels):
        score = next((s["score"] for s in scores if s["label"] == label["label"]), 0)

        # Color gradient based on confidence
        color_intensity = min(int(score * 255), 255)

        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                # Access full_name from the mapping if available
                full_name = label_to_name_mapping.get(label["label"], label["label"])
                st.markdown(f"**{full_name}**")
                st.caption(f"Tag: {label['label']}")
            with cols[1]:
                st.markdown(
                    f"<h3 style='text-align: right; color: rgb(0, {color_intensity}, {255 - color_intensity});'>{score:.2f}</h3>",
                    unsafe_allow_html=True,
                )

    display_all_scores(scores, label_to_name_mapping)


def display_all_scores(scores: list[LabelScore], label_to_name_mapping: Dict[str, str]):
    """
    Display all scores in an expandable section

    Args:
        scores: All scores from the model
        label_to_name_mapping: Mapping from label codes to full names
    """
    with st.expander("View all category scores"):
        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        for score_item in sorted_scores[:20]:  # Show top 20
            label_name = label_to_name_mapping.get(
                score_item["label"], score_item["label"]
            )
            st.text(f"{label_name} ({score_item['label']}): {score_item['score']:.4f}")
