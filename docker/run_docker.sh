#!/bin/bash

DOCKER_ARGS=()

# General
DOCKER_ARGS+=("--name=jf_masterarbeit")
DOCKER_ARGS+=("-it")
DOCKER_ARGS+=("-d")
DOCKER_ARGS+=("-h=MA")

# Memory management
DOCKER_ARGS+=("--memory=32GB")
DOCKER_ARGS+=("--memory-reservation=32GB")
DOCKER_ARGS+=("--cpus=8")
DOCKER_ARGS+=("--gpus=device=1")

# Volumes
DOCKER_ARGS+=("-v" "$HOME/masterarbeit:/masterarbeit")
DOCKER_ARGS+=("-v" "/datahome/jfuerstenwerth/:/masterarbeit/data/cache/")

docker run "${DOCKER_ARGS[@]}" jf/masterarbeit

# Install repo package
docker exec jf_masterarbeit pip install -e /masterarbeit
