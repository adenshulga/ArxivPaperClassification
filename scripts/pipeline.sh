#!/bin/bash

export PYTHONPATH='.'

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 uv run entrypoints/pipeline.py 