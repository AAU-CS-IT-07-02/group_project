!/bin/bash
#SBATCH --job-name=W-pysindy
#SBATCH --mail-type=ALL  # Type of email notification: BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ebraha25@student.aau.dk
#SBATCH --output=/nfs/home/student.aau.dk/cj32if/group_project/SLURM_scripts_and_results/sparse_id_py-venv-OUTPUT/CHILD-sparse_id-venv-JOB-%j.out  # Redirect the output stream to this file (%j is the jobid)
#SBATCH --error=/nfs/home/student.aau.dk/cj32if/group_project/SLURM_scripts_and_results/sparse_id_py-venv-OUTPUT/CHILD-sparse_id-venv-JOB-%j.err   # Redirect the error stream to this file (%j is the jobid)
#SBATCH --partition=naples,dhabi,rome  # Which partitions may your job be scheduled on
#SBATCH --mem=10G  # Memory limit that slurm allocates
#SBATCH --time=15:00:00  # (Optional) time limit in dd:hh:mm:ss format. Make sure to keep an eye on your jobs (using 'squeue -u $(whoami)') anyways.

python3 dynamic_model_smart_building.py $PARAMS

# Maybe you need to copy a result file back to ${PD}
