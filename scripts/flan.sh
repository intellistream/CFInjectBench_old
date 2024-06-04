methods="vanilla mixreview lora kadapter_2 modular_small kilm"
export CUDA_VISIBLE_DEVICES=1
for method in $methods
do
python run.py --config ~/cf-kilm/configs/flan/flant5_${method}.json
done