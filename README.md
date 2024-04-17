# Fast Knowledge Injection in Large Language Models

In order to reproduce our results, take the following steps:

### Prerequisites

- Python (=3.8.x)
- Required Python libraries (listed in `requirements.txt`)

### Setup

1. Install necessary Python libraries:
```
pip install -r requirements.txt
```

> NOTE: Also, make sure to install the correct version of pytorch corresponding to the CUDA version and environment:

2. Download dataset for experiments:
```
sh Download_dataset
```

3. Run experiments:

- version = [`day`, `month`, `quarter`]
- method = [`vannila`, `kadapter_2`, `kadapter_3`, `lora`, `mixreview`, `modular`, `recadam`]

```
python3 run.py --config configs/online/wiki/{version}/t5_{method}.json

```

Replace `{version}` and `{method}` with appropriate values from the provided options.

#### Components in each configurations file
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
