#!/bin/bash
# bash script to activate the virtual environment and run the main.py

# CD to the script's directory
cd "$(dirname "$0")"

source .venv/bin/activate && python main.py