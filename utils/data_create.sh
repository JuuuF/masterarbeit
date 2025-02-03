#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start> <stop>"
    exit 1
fi

start=$1
stop=$2

# Loop through the range from start to stop (inclusive)
if ((stop >= start)); then
    for ((i = $start; i <= $stop; i++)); do
        echo "Running script with argument: $i"
        python src/ma_darts/generation/data_generation.py --start $i --n_samples 1
    done
else
    for ((i = $start; i >= $stop; i--)); do
        echo "Running script with argument: $i"
        python src/ma_darts/generation/data_generation.py --start $i --n_samples 1
    done
fi
