# arXiv Paper Classification

A machine learning application that predicts arXiv categories for academic papers based on their title and abstract. This tool uses a fine-tuned SciBERT model to classify papers into arXiv subject categories. This task is completed as homework for YSDA ML 2 course

I personally hate jupyter-notebooks, so as a proof that i conducted experiments i made Comet ML logger project public.

Latest training logs, configs and other details can be found here https://www.comet.com/adenshulga/arxiv-papers-classification/ef1256f1d4eb4b588da881366eb27578?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step

## Installation

There are two relatively close dockerfile configurations. container_setup folder contains scripts and dockerfile to setup interactive developmpent environment. Dockerfile in the root is for deploying a StreamlitApp.

### Streamlit App Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/adenshulga/arxiv-paper-classification.git
   cd arxiv-paper-classification
   ```

2. Give permissions for executable scripts:
    ```
    chmod +x scripts/pipeline.sh scripts/launch_app.sh
    ```

3. Build and launch docker container:
    ```
    docker build -t arxiv-paper-clf .
    docker run -p 9001:9001 arxiv-paper-clf
    ```


### Configuration

You can modify the inference settings in `config/inference_config.py`:

- `model_name`: Base model name from Hugging Face
- `checkpoint_path`: Path to fine-tuned model checkpoint
- `top_percent`: Cumulative score threshold for showing predictions
- `minimal_score`: Minimum confidence score to display

## Development and model Training

To enter development environment
1. Fill container_setup/credentials file

2. Give executable permissions to build and launch scripts:
    ```
    chmod +x container_setup/build.sh container_setup/launch_script.sh
    ```

3. Specify resources constrains in ./container_setup/launch_container.sh
 

4. Build and launch docker container
    ```
    ./container_setup/build.sh
    ./container_setup/launch_container.sh
    ```

5. Attach to running container
    ```
    docker attach <container-id>
    ```

6. Install the dependencies
    ```
    uv venv
    uv sync
    ```

To train the model:

1. Load and unzip the arxiv dataset in the `data` folder(https://www.kaggle.com/datasets/neelshah18/arxivdataset)
2. Configure the process in config/pipeline_config.py

Run the training script:
   ```
   scripts/pipeline.sh    
   ```