methods="kcenter model random"
export CUDA_VISIBLE_DEVICES=0
for method in $methods
do
python run.py --config ~/cf-kilm/configs/coreset/t5_base_${method}.json
done