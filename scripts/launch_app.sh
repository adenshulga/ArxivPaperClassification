#!/bin/bash

export PYTHONPATH='.'

# source .venv/bin/activate

CUDA_VISIBLE_DEVICES="" uv run -m streamlit run entrypoints/app.py --server.address=0.0.0.0 --server.port=9001
