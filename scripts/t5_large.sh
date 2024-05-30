methods="kd"
#methods="initial vanilla recadam mixreview lora kadapter_2 modular kd"
export CUDA_VISIBLE_DEVICES=1
for method in $methods
do
python run.py --config ~/cf-kilm/configs/t5_large/t5_${method}.json
done


#python nq_eval.py --config /home/yuhao/OCKL/configs/online/wiki/month/t5_initial.json

#WANDB_BASE_URL="https://api.wandb.ai"
# WANDB_MODE="offline"