from comet_ml import Experiment
import os


def setup_logging():
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="arxiv-papers-classification",
        auto_output_logging="simple",
    )

    return experiment
