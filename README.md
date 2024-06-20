# Python Code Generator

## Project Overview

This personal project aims to train GPT-2 and BERT models on Python code sourced from GitHub repositories. The primary goal is to understand how these transformer models learn and process information. Although the data collection and preprocessing stages have been completed, I did not have sufficient computational power to fully train the models.

The dataset consists of Python files created on GitHub within a span of 6 days. These files are tokenized and fed into the transformer models (GPT-2 and BERT) to generate code. However, the project's scope and accuracy are limited due to the small data volume and insufficient computational resources. Thus far, I have only attempted to train the model using a single NVIDIA GeForce GPU.

## Highlights

- **Data Collection:** Gathered Python files from GitHub repositories.
- **Preprocessing:** Thoroughly processed the collected data for model training.
- **Model Training:** Attempted to train GPT-2 and BERT models, focusing on understanding their learning mechanisms.

## Challenges

- **Data Volume:** The dataset is limited to Python files generated over 6 days.
- **Computational Power:** Training the models is constrained by the available hardware, specifically a single NVIDIA GeForce GPU.

## Files

- `data_collect.py`: Collects Python repositories from GitHub.
- `preprocess.py`: Preprocesses the collected data.
- `gpt2_trainer.py`: Trains a GPT-2 model on the preprocessed data.
- `bert_trainer.py`: Trains a BERT model on the preprocessed data.
- `main.py`: The main script to run the project.
- `requirements.txt`: Lists the dependencies.
- `token.txt`: GitHub access token (not included in the repository).

## Conclusion

While this project has provided valuable insights into the workings of GPT-2 and BERT models, the results are limited due to data and computational constraints. Future improvements could include gathering a larger dataset and leveraging more powerful hardware for training.


## Usage

1. Set up your environment:

    ```bash
    pip install -r requirements.txt
    ```

2. Collect data:

    ```bash
    python data_collect.py
    ```

3. Preprocess data:

    ```bash
    python preprocess.py
    ```

4. Train the model (GPT-2 or BERT)–the code should be modified to choose between the models:

    ```bash
    python main.py
    ```

