#!/usr/bin/bash
# This will exit at the first failure
# The most probable point of failure would be a non existent venv, to prevent messing with the global python env
set -e

echo "Navigating to project's root..."
cd "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
echo "Navigating to project's root...done"

echo "Activating environment..."
source "$(pwd).13.env/bin/activate"
echo "Activating environment...done"

echo "Updating..."
pip install -U pip
pip install -Ur requirements.txt
pip install -r artifacts/requirements.txt
echo "Updating...done"

echo "Freezing..."
pip freeze > frozen-requirements.txt
echo "Freezing...done"
