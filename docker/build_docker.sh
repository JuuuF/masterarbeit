#!/bin/bash

DOCKER_ARGS=()

# General
DOCKER_ARGS+=("-t" "jf/masterarbeit")
DOCKER_ARGS+=("-f" "$HOME/masterarbeit/docker/Dockerfile_server")

docker build "${DOCKER_ARGS[@]}" .
