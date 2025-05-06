#!/bin/bash

uv sync
uv run src/load_datasets.py
uv run src/main.py
