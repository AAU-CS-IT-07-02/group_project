#  SLURM Basics

**SLURM** (Simple Linux Utility for Resource Management) is a workload manager used on clusters and HPC systems. It schedules jobs, allocates compute resources, and manages queues.

### Core ideas

* **Job** = your task (script, program, simulation).
* **Node** = a compute server in the cluster.
* **Partition** = a queue or group of nodes with similar properties.
* You submit jobs with `sbatch`, check them with `squeue`, and cancel them with `scancel`.

---

# Writing a SLURM Job Script

A SLURM job script is just a **bash script** with special headers (`#SBATCH`) that tell SLURM what resources you need.

### Example: running a SLURM script from the **mcc3** docs

```bash
#!/bin/bash
#SBATCH --job-name=hello_world  # Give your experiment a name
#SBATCH --mail-type=FAIL  # Type of email notification: BEGIN,END,FAIL,ALL,NONE
#SBATCH --mail-user=<YOUR-EMAIL>
#SBATCH --partition=naples,dhabi,rome  # Which partitions may your job be scheduled on
#SBATCH --time=1:00:00  # (Optional) time limit in dd:hh:mm:ss format. Make sure to keep an eye on your jobs (using 'squeue -u $(whoami)') anyways.
#SBATCH --mem=1G  # Memory limit that slurm allocates

# Memory limit for user program. Equals the SBATCH-directive allocation.
#  (allows graceful handling of out-of-memory errors in your program.)
let "m=1024*$SLURM_MEM_PER_NODE"
ulimit -v $m

# Print info to stdout on job details
echo "Job-name: $SLURM_JOB_NAME; jobid: $SLURM_JOB_ID; Partition: $SLURM_JOB_PARTITION; No. Nodes: $SLURM_JOB_NUM_NODES; Node: $SLURM_JOB_NODELIST; Memory per Node: ${SLURM_MEM_PER_NODE}MB; Start-time: $(date); Hostname: $(hostname)"

#######################
# Your code goes below #
#######################
```

---

# Workflow to run

1. Save this as `hello-world.sh`.
2. Submit to SLURM:

   ```bash
   sbatch ~/experiments/hello-world.sh
   ```
3. Check job status:

   ```bash
   squeue -u $(whoami)
   ```
4. When it finishes, check the output in `slurm-$<jobid>.out`.

---

# Key Notes

* Always check your cluster’s documentation — for [mcc3](https://github.com/DEIS-Tools/DEIS-MCC/blob/main/usage/NODES.md)
* If you need **GPUs**, add:

  ```bash
  #SBATCH --gres=gpu:1
  ```

## Sources
[SLURM official site](https://slurm.schedmd.com/overview.html)
[Standford SLURM basics](https://stanford-rc.github.io/docs-earth/docs/slurm-basics)
[Hello world in SLURM](https://github.com/DEIS-Tools/DEIS-MCC/blob/main/usage/SIMPLE.md)


# SLURM job parameters 
Be sure that you change the `<email_for_fail_report>` with your actual email to get notified for the job state. Also  `<AAU-ID>`  with your actual one(check the shell is written there ex. `12ab4t`)

```
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
```

# Main script body
Here you can change what dirs are copied to  `scratch/`  and what python file is executed.  

```
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
```

# Cleaning after yourself
This is probably the most important part, be sure that is at the end of the script.

```
# Clean up after yourself
cd /scratch/${U}
[ -d "${SLURM_JOBID}" ] && rm -r ${SLURM_JOBID}
```

## sparse_id_py-venv-JOB.sh

```bash
--8<-- "SLURM_scripts_&_results/sparse_id_py-venv-JOB.sh"
```
