#!/bin/bash
#SBATCH --mail-type=ALL  # Type of email notification: BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<email_for_fail_report>
#SBATCH --output=/nfs/home/student.aau.dk/<AAU-ID>/group_project/SLURM_scripts/sparse_id_py-venv-OUTPUT/sparse_id-venv-JOB-%j.out  # Redirect the output stream to this file (%j is the jobid)
#SBATCH --error=/nfs/home/student.aau.dk/<AAU-ID>/group_project/SLURM_scripts/sparse_id_py-venv-OUTPUT/sparse_id-venv-JOB-%j.err   # Redirect the error stream to this file (%j is the jobid)
#SBATCH --partition=naples,dhabi,rome  # Which partitions may your job be scheduled on
#SBATCH --mem=2G  # Memory limit that slurm allocates
#SBATCH --time=1:00:00  # (Optional) time limit in dd:hh:mm:ss format. Make sure to keep an eye on your jobs (using 'squeue -u $(whoami)') anyways.

let "m=1024*1024*$SLURM_MEM_PER_NODE"
ulimit -v $m

U=$(whoami)
PD=$(pwd) 

# Create a unique folder for this job execution in your scratch folder.
SCRATCH_DIRECTORY=/scratch/${U}/${SLURM_JOBID}  
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# Activate python virtual environment
python3 -m venv venv
source venv/bin/activate

# Copy your project, and install dependencies (must be listed in requirements.txt)
cp -r ${PD}/group_project/thermodynamics_modeling/sparse_identification/* .
python -m pip install -r requirements.txt

## NOTE: Installing dependencies for each job invocation may be too expensive in some cases!
## So consider using the same installation folder. And please share an example, if you make it work :)
###

##########################
# Run your python script # 
##########################

python pysindy.py

# Maybe you need to copy a result file back to ${PD}


# Clean up after yourself
cd /scratch/${U}
[ -d "${SLURM_JOBID}" ] && rm -r ${SLURM_JOBID}
