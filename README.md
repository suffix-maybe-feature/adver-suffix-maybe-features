# Adversarial Suffixes: Exploring Embedding-Based Attacks on Language Models

This repository provides tools and datasets for investigating adversarial suffix attacks using embedding techniques. Our project focuses on generating adversarial embeddings and creating harmful datasets for Llama2 and Llama3 models, demonstrating potential vulnerabilities in these language models.

## Features

- Embedding suffix attack implementation
- Adversarial embedding generation tools
- Datasets for harmful fine-tuning
- Scripts for generating harmful data using Llama2 and Llama3 models

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers library
- Datasets library

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/adversarial-suffixes.git
   cd adversarial-suffixes
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Generating Adversarial Embedding Suffixes

1. Prepare your dataset:
   - Use your own benign/harmful dataset or utilize the provided dataset in `code/data`.
   - Update the dataset path in `latent_at/lat_datasets.py` (line 221):
     ```python
     dataset = load_dataset('parquet', data_files="./your_dataset_path.parquet", split='train')
     ```

2. Run the attack:
   - For value attack (default):
     ```
     python test.py
     ```
   - For token attack, you are required to test on different parameter to ensure a successful generation which usually take longer:
     ```
     python test.py --embedding_constraint True
     ```

### Generating Harmful Data

- Using Llama2-uap results:
  ```
  python exper_on_uap_llama2.py
  ```

- Using Llama3-uap results:
  ```
  python exper_on_uap_llama3.py
  ```

## Directory Structure

- `Llama2-uap/`: Results from Llama2 model using adversarial suffixes
- `Llama3-uap/`: Results from Llama3 model using adversarial suffixes
- `datasets/`: Contains benign dataset for harmful fine-tuning
- `code/`: Source code for attacks and data generation


## Disclaimer

This project is for research purposes only. The authors are not responsible for any misuse of the provided tools or datasets.

For questions or support, please open an issue in the GitHub repository.
