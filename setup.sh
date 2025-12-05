#!/bin/bash

git update-index --skip-worktree config/config_train.yaml
git update-index --skip-worktree config/dataset.ini

echo "Config files have been set to skip-worktree."

