#!/bin/bash

for f in "$1"/*/; do
    if [ -d "$f" ]; then
        echo "$f"
        python3 ./src/scenario_data_collector.py -p "$f" -o "$2"
    fi
done