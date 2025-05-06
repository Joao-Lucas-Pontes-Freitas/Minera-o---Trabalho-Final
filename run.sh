#!/bin/bash

uv sync
uv run src/load_dataset.py
uv run src/main.py
