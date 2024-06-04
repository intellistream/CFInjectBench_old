methods="vanilla recadam mixreview lora kadapter_2 modular kd kilm"
export CUDA_VISIBLE_DEVICES=0
for method in $methods
do
python run_faststream.py --config ~/cf-kilm/configs/stream/t5_${method}_stream.json
done