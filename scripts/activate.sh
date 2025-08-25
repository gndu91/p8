# This will activate the environment


test "${BASH_SOURCE[0]}" == "$0" && {
  echo WARNING: This script needs to be sourced, not run > /dev/stderr
  exit 1
}

# TODO: Auto create environment

echo "Navigating to project's root..."
cd "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
echo "Navigating to project's root...done"

echo "Activating environment..."
source "$(pwd).13.env/bin/activate"
echo "Activating environment...done"
