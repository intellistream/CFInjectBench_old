##  Continuous Fast Knowledge Injection in Language Models: Benchmarking and Rethinking

## Setup
- Install miniconda
- `conda env create -f environment.yml`
- `conda activate cf`
- `python -m spacy download en_core_web_sm`

## Components in each configurations file
- input_length (int) : the input sequence length
- output_length (int) : the output sequence length
- num_train_epochs (int) : number of training epochs 
- output_dir (string) : the directory to save the model checkpoints
- dataset (string) : the dataset to perform continual pretraining
- dataset_version (string) : the version of the dataset ['day', 'month', 'quarter']
- train_batch_size (int) : batch size used for training
- eval_batch_size (int) : batch size used for evaluation
- learning rate (float) : learning rate used for training
- model (string) : model name in huggingface models (https://huggingface.co/models)
- method (string) : method being used ['vannila', 'kadapter', 'lora', 'mixreview', 'modular', 'recadam']
- freeze_level (int) : how much of the model to freeze during traininig (0 for none, 1 for freezing only encoder, 2 for freezing all of the parameters)
- gradient_accumulation_steps (int) : gradient accumulation used to match the global training batch of each method
- ngpu (int) : number of gpus used for the run
- num_workers (int) : number of workers for the Dataloader
- use_deepspeed (bool) : false by default. Currently not extensively tested.
- use_lr_scheduling (bool) : true if using learning rate scheduling
- check_validation (bool) : true for evaluation (no training)
- checkpoint_path (string) : path to the model checkpoint that is used for evaluation
- output_log (string) : directory to log evaluation results to

## Running Experiments
All commands should be run under the project root directory. 
- Running experiments based on T5-base model: `sh scripts/t5_base.sh`
- Running experiments based on T5-large model: `sh scripts/t5_large.sh`
- Running experiments based on Flan-T5-xl model: `sh scripts/flan.sh`
- Running time constrainted setup experiments based on T5-base model: `sh scripts/stream.sh`
- Running experiments based on GPT2 model: `sh scripts/gpt2.sh`
- Running coreset experiments based on T5-base model: `sh scripts/coreset.sh`
- Running K-Center selection with varing ratio: `sh scripts/ratio.sh`

## Computing Metrics
Computing the final performance of experiments: `python compute_metric.py`
