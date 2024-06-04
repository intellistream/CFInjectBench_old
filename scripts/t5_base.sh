methods="initial vanilla recadam mixreview lora kadapter_2 modular kd kilm"
export CUDA_VISIBLE_DEVICES=0
for method in $methods
do
python run.py --config ~/OCKL/configs/t5_base/t5_${method}.json
done