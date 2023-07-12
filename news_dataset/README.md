# Data Generation

This is brief about generating dataset of news articles crawled from various domains using the [news-please](https://github.com/fhamborg/news-please) library and [Common Crawl](https://commoncrawl.org/) dataset. 

## Requirements

- Python (>= 3.10)
- news-please
- Other dependencies specified in requirements.txt

## Installation

1. Clone the repository:
```
git clone https://github.com/mozhu621/Online-continual-learning-factual-knowledge-for-LLM.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download the trained Spacy language model for entity extraction:
```
python3 -m spacy download en_core_web_trf
```

## Data Generation Pipeline

### Data Crawling

1. Use the news-please library to crawl news articles from the Common Crawl dataset. This can be done by specifying the desired domains and time range for crawling.

> NOTE: domains used to crawl are provided in `domains.json`.

2. After crawling, use the `data_gen.py` script to read the crawled JSON files and generate a Pandas DataFrame for further processing.

### Data Cleaning

1. Open the `data_clean.py` script to perform data cleaning and preprocessing on the generated DataFrame.

### Entity Extraction

1. Run the `ner_gen.py` script to extract entities from each news article using the trained Spacy language model.

### Training Dataset Generation

1. Use `train_data_gen.py` script to generate training data for Salient Span Masking (SSM) task using the extracted entities and text from news articles.


## Running Training Data Generation Pipeline

To automate the entire training data generation pipeline, you can use the provided shell script `pipeline.sh`. This script will run each step of the pipeline and handle errors gracefully. Follow the steps below to execute the pipeline:

```
sh train_data_pipeline.sh
```

The script will execute each step sequentially, checking for the existence of output files before performing each step. If an output file already exists, the corresponding step will be skipped. Any errors encountered in any step will cause the pipeline to stop executing.