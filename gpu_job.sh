# Pick GPU(s) to use
#export CUDA_VISIBLE_DEVICES=2,3

# Optional: limit PyTorch GPU memory growth (if using PyTorch)
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:10240

export PYTORCH_NO_TELEMETRY=1

# Name of log file
LOGDIR="/home/hofmanja/logs"
LOGFILE="$LOGDIR/infered_velocity$(date +%Y%m%d_%H%M%S).log"


# prepare environment
module load python/3.12.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

#git checkout b07aa2b97ae0ba8680425fe39ee199253eb44cce

# Run the Python script in the background, redirecting output to log
nohup env CUDA_VISIBLE_DEVICES=2 python main.py '/data/scratch/hofmanja/Heat-Plume-Data/runs/infered_velocity/config.yaml' > "$LOGFILE" 2>&1 &
#bash vampireman.sh > "$LOGFILE" 2>&1 &


# Print job info
#echo "Job started on GPU $CUDA_VISIBLE_DEVICES"
echo "Logging to $LOGFILE"
echo "Use 'tail -f $LOGFILE' to monitor progress"

echo "New version"

#git checkout main