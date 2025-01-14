#!/bin/bash

SSH_ALIAS="ma_kiel"

EXCLUDE_DIRS=(
    "data/darts_references"
    "data/cache"
    "data/generation/out*"
    "data/generation/references"
    "data/paper"
    "data/ai"
    "dump"
    "notes"
    "private"
    "workspaces"
    "writing"
    ".git"
    ".gitignore"
    ".gitattributes"
    "README.md"
)

EXCLUDE_OPTS=""
for dir in "${EXCLUDE_DIRS[@]}"; do
    EXCLUDE_OPTS+="--exclude=$dir "
done

rsync -avz  $EXCLUDE_OPTS $(pwd)/ $SSH_ALIAS:"~/masterarbeit"
