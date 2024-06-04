methods="initial vanilla recadam mixreview lora kadapter_2 modular kd kilm"
export CUDA_VISIBLE_DEVICES=1
for method in $methods
do
python run.py --config ~/OCKL/configs/t5_large/t5_${method}.json
done