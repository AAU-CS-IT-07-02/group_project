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

source venv/bin/activate
cd sparse_identification



  # Wait until fewer than MAX_CONCURRENT jobs are running
  # while [ $(squeue -u $USER -h -n W-pysindy | wc -l) -ge $MAX_CONCURRENT ]; do
  #   echo "[$(date)] Waiting for free slot..."
  #
  #   JobIDs=$(squeue -u $(whoami) -o "%F" --noheader)
  #   for i in "${JobIDs[@]}"; do
  #     sstat -j $i.batch -o JobID,AveCPU,MaxRSS,AveRSS,MaxDiskRead,MaxDiskWrite
  #   done
  #   sleep 60
  # done

  # echo "[$(date)] Submitting worker for $PARAMS"
  # sbatch --export=ALL,PARAMS="$PARAMS" ../sparse_id_py-venv-JOB.sh

while IFS= read -r PATH_TO_EXP_FOLDERS; do
  
done < $EXP_FILE

echo "[$(date)] All jobs submitted."

while [ $(squeue -u $(whoami) -h -n W-pysindy | wc -l) -gt 0 ]; do
  echo "[$(date)] Waiting for all worker jobs to finish..."
  sleep 60
done

echo "[$(date)] All worker jobs completed. Cleaning up..."
# Clean up after yourself
cd /scratch/${U}
[ -d "${SLURM_JOBID}" ] && rm -r ${SLURM_JOBID}
