#!/bin/bash
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err
#SBATCH -p Quick
#SBATCH --exclude=GPU41,GPU42
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --array=0-3

source $(conda info --base)/etc/profile.d/conda.sh

SHIFTS=("texture" "blur" "gamma" "noise")
SHIFT=${SHIFTS[$SLURM_ARRAY_TASK_ID]}
PORT=$((3100 + SLURM_ARRAY_TASK_ID))

echo "Running shift mode: $SHIFT on reward server port $PORT"

# Start reward server in background on unique port, log to temp file
SERVER_LOG=/tmp/server_${PORT}.log
conda activate monkey-verifier
cd /data/dayneguy/vla/RoboMonkey/monkey-verifier/src
python infer_server.py --port $PORT > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# Wait until model is fully loaded (startup event complete)
until grep -q "Application startup complete" $SERVER_LOG 2>/dev/null; do
    sleep 5
done
echo "Reward server ready on port $PORT"

conda activate openvla
cd /data/dayneguy/vla/openvla

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5 \
  --shift_mode $SHIFT \
  --reward_server_port $PORT \
  --task_id $SLURM_ARRAY_TASK_ID \
  --mode robomonkey

kill $SERVER_PID