methods="vanilla recadam mixreview lora kadapter_2"
export CUDA_VISIBLE_DEVICES=1
for method in $methods
do
python run.py --config ~/cf-kilm/configs/gpt2/gpt2_${method}.json
done
