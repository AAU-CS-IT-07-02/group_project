#!/bin/bash

# Parse arguments
while getopts "e:" flag; do
  case "${flag}" in
    e) EXP_FILE=${OPTARG};;
    *) echo "Usage: $0 -e <experiment_file>"; exit 1;;
  esac
done

# Check if EXP_FILE is set
if [ -z "$EXP_FILE" ]; then
  echo "Error: Missing required -e argument for experiment file."
  echo "Usage: $0 -e <experiment_file>"
  exit 1
fi

echo "Experiment file provided: $EXP_FILE"

# You can now pass this to sbatch like this:
sbatch --export=ALL,EXP_FILE="$EXP_FILE" sparse_id_JOB_Leader.sh
