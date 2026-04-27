#!/bin/bash

# bash script to dump or load pip requirements
# usage: ./dump_req.sh (--load)?
if [ "$1" == "--load" ]; then
    echo "Loading requirements from requirements.txt..."
    pip install -r requirements.txt
elif [ "$1" == "--help" ]; then
    echo "Usage: $0 [--load|--help]"
    echo "  --load   Load requirements from requirements.txt"
    echo "  --help   Show this help message"
else
    echo "Dumping current pip requirements to requirements.txt..."
    pip freeze > requirements.txt
fi