#!/bin/bash
#SBATCH --job-name=NODE
#SBATCH --mail-type=ALL  # Type of email notification: BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<email_for_fail_report>
#SBATCH --output=/nfs/home/student.aau.dk/<student-id>/group_project/SLURM_scripts_and_results/node_train-OUTPUT/node_train-JOB-%j.out  # Redirect the output stream to this file (%j is the jobid)
#SBATCH --error=/nfs/home/student.aau.dk/<student-id>/group_project/SLURM_scripts_and_results/node_train-OUTPUT/node_train-JOB-%j.err   # Redirect the error stream to this file (%j is the jobid)
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G

set -euo pipefail

# Who/where
U=$(whoami)
PD=$(pwd)  # submit from repo root

# Create unique scratch folder for this job
SCRATCH_DIRECTORY=/scratch/${U}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
cd "${SCRATCH_DIRECTORY}"

echo "[JOB] Scratch dir: ${SCRATCH_DIRECTORY}"

# Create or copy venv, then activate
SKIP_PIP_INSTALL=0
if [ -d "${PD}/thermodynamics_modeling/venv" ]; then
    echo "[JOB] Copying existing venv from ${PD}/thermodynamics_modeling/venv to scratch..."
    cp -a "${PD}/thermodynamics_modeling/venv" ./venv
    source venv/bin/activate
    SKIP_PIP_INSTALL=1
elif [ -d "${PD}/venv" ]; then
    echo "[JOB] Copying existing venv from ${PD}/venv to scratch..."
    cp -a "${PD}/venv" ./venv
    source venv/bin/activate
    SKIP_PIP_INSTALL=1
else
    echo "[JOB] No venv found in repo; creating a new venv..."
    python3 -m venv venv
    source venv/bin/activate
    SKIP_PIP_INSTALL=0
fi

# Install dependencies only if we didn't copy an existing venv
if [ "$SKIP_PIP_INSTALL" -eq 0 ]; then
    pip install --upgrade pip
    pip install -r "${PD}/thermodynamics_modeling/requirements.txt"
    pip install -r "${PD}/thermodynamics_modeling/pythorch_node/requirements.txt"
else
    echo "[JOB] Using copied venv; skipping pip install."
fi

# Copy the full repository to scratch to preserve relative dataset paths
echo "[JOB] Copying repository to scratch..."
cp -a "${PD}/." .

pwd
ls 

cd "${SCRATCH_DIRECTORY}/thermodynamics_modeling/pythorch_node/"

# Determine OUTDIR from config (falls back to './out')
OUTDIR=$(python3 - <<'PY'
import yaml,sys
try:
    cfg = yaml.safe_load(open('thermodynamics_modeling/pythorch_node/config.yml'))
    print(cfg.get('outdir','out'))
except Exception as e:
    print('out')
PY
)

echo "[JOB] Using OUTDIR=${OUTDIR}"

### 1) Generate dataset split
echo "[JOB] Generating dataset split..."
python3 split_for_test.py

### 2) Train Neural ODE
echo "[JOB] Starting training..."
python3 torchdiffeq_model.py

### 3) Evaluate and plot
echo "[JOB] Running evaluation and plotting..."
# Evaluate script expects --out to point to the folder with scalers & checkpoint
python3 evaluate_and_plot.py --test ./dataset_split/test_data.csv -out ${OUT} --H 500 --mode last --solver rk4 --windows 1

### 4) Copy outputs back to project directory
echo "[JOB] Copying outputs back to project directory..."
DEST_DIR="${PD}/SLURM_outputs/job_${SLURM_JOBID}"
mkdir -p "${DEST_DIR}"

if [ -d "${OUTDIR}" ]; then
    cp "${OUTDIR}/" "${DEST_DIR}/${OUTDIR}/"
    echo "[JOB] Copied ${OUTDIR} -> ${DEST_DIR}/${OUTDIR}"
else
    echo "[WARN] OUTDIR '${OUTDIR}' not found; copying repository log files instead."
fi

# Copy job stdout/stderr if available (Slurm will also write to central location)
cp -v "${SLURM_JOB_ID:-${SLURM_JOBID}}"* "${DEST_DIR}/" 2>/dev/null || true

echo "[JOB] Done. Outputs stored in ${DEST_DIR}"

# Cleanup: remove scratch directory (optional; commented out for safety)
cd /scratch/${U}
[ -d "${SCRATCH_DIRECTORY}" ] && rm -rf "${SCRATCH_DIRECTORY}"

deactivate || true
