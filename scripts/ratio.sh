methods="r=0.25 r=0.75"
export CUDA_VISIBLE_DEVICES=0
for method in $methods
do
python run.py --config ~/OCKL/configs/ratio/t5_base_kcenter_${method}.json
done