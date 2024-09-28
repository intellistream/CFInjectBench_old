methods="initial vanilla mixreview lora kadapter_2"
export CUDA_VISIBLE_DEVICES=0
for method in $methods
do
python run.py --config ~/cf-kilm/configs/llama/llama_${method}.json
done
