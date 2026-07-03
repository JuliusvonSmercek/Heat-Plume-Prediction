source .venv/bin/activate
pip install -r requirements.txt

LOGDIR="/home/hofmanja/logs"
LOGFILE="$LOGDIR/test$(date +%Y%m%d_%H%M%S).log"

nohup env CUDA_VISIBLE_DEVICES=1 python code/main.py settings/config_iks.yaml > "$LOGFILE" 2>&1 &
