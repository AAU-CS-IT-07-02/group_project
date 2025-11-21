#!/bin/bash
#SBATCH --job-name=L-pysindy
#SBATCH --mail-type=ALL  # Type of email notification: BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ebraha25@student.aau.dk
#SBATCH --output=/nfs/home/student.aau.dk/cj32if/group_project/SLURM_scripts_and_results/sparse_id_py-venv-OUTPUT/LEADER-sparse_id-venv-JOB-%j.out  # Redirect the output stream to this file (%j is the jobid)
#SBATCH --error=/nfs/home/student.aau.dk/cj32if/group_project/SLURM_scripts_and_results/sparse_id_py-venv-OUTPUT/LEADER-sparse_id-venv-JOB-%j.err   # Redirect the error stream to this file (%j is the jobid)
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

MAX_CONCURRENT=10     # limit number of concurrent sub jobs
OUTDIR=${PD}/group_project/thermodynamics_modeling/data_fragmentation/out

U=$(whoami)
PD=$(pwd) 

# Create a unique folder for this job execution in your scratch folder.
SCRATCH_DIRECTORY=/scratch/${U}/${SLURM_JOBID}  
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# Copy the taskts to run 
cp ${PD}/group_project/SLURM_scripts_and_results/sparse_id_py-venv-JOB.sh .
cp ${PD}/group_project/AAU-BUILD-sensor.actuator/6roomsOffice/dataset_with_occupancy_delimiter_comma.csv .

# Copy your project
cp -r ${PD}/group_project/thermodynamics_modeling/* .

source ${PD}/group_project/thermodynamics_modeling/venv/bin/activate

# Step 1: Run split script for each experiment path
TMP_OUT_PATHS=()
while IFS= read -r EXP_PATH; do
  python3 ${PD}/group_project/thermodynamics_modeling/data_fragmentation/split_by_rooms_category_timeframe.py \
    --sensors "$EXP_PATH/sensors.txt" \
    --actuators "$EXP_PATH/actuators.txt" \
    --config "$EXP_PATH/configuration.txt" \
    --rooms "$EXP_PATH/rooms.txt" \
    --timeframe "$EXP_PATH/timeframe.txt" \
    --outdir "out"
done < "$EXP_FILE"

TMP_OUT_PATHS=($(ls -d ${OUTDIR}/*/))

# Step 2: Generate hyperparameter combinations
HP_FILE="hp_combinations.txt"
python3 ${PD}/group_project/thermodynamics_modeling/generate_hp_combination.py --output "$HP_FILE"

# Step 3: Submit jobs for each output path and hyperparameter combination
while IFS= read -r HP_COM; do
  for OUT_PATH in "${TMP_OUT_PATHS[@]}"; do
    # Wait for available slot
    while [ $(squeue -u $USER -h -n W-pysindy | wc -l) -ge $MAX_CONCURRENT ]; do
      echo "[$(date)] Waiting for free slot..."
      sleep 60
    done

    PARAMS="--sensors $OUT_PATH/data_sensors.csv --actuators $OUT_PATH/data_actuators.csv --config $OUT_PATH/data_configuration.csv $HP_COM"
    echo "[$(date)] Submitting job with: $PARAMS"
    sbatch --export=ALL,PARAMS="$PARAMS" ${PD}/group_project/SLURM_scripts_and_results/sparse_id_py-venv-JOB.sh
  done
done < "$HP_FILE"

echo "[$(date)] All jobs submitted."

while [ $(squeue -u $(whoami) -h -n W-pysindy | wc -l) -gt 0 ]; do
  echo "[$(date)] Waiting for all worker jobs to finish..."
  sleep 60
done

echo "[$(date)] All worker jobs completed. Cleaning up..."
# Clean up after yourself
cd /scratch/${U}
[ -d "${SLURM_JOBID}" ] && rm -r ${SLURM_JOBID}
