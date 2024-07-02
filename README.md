Welcome to the GitHub repository for my thesis, **Cross-Lingual Bias and Multilingual Fidelity in Large Language
Models: A Comparative Analysis**. This repository contains various Jupyter Notebooks, datasets, and Python code files used for the analysis and evaluation of two models, LLama3-8B-Instruct and MISTRAL-7b-Instruct. Below is an overview of the contents of each folder and file in this repository.

## Repository Contents

### Folders

#### Arts & Landmarks Question Forming
This folder contains Jupyter Notebooks that detail the performance of the models (LLama3 and MISTRAL) on the Arts and Landmarks dataset. It includes:
- Methodology of how the Arts and Landmarks datasets were created.
- Examples of questions and their respective prompt types (Multiple Choice, True/False, Open-Ended).
- Results of model performance for each prompt type.

#### Datasets
This folder contains compiled or merged versions of all the datasets used in the experiments. These datasets are provided to allow for re-running the experiments and verifying the results.

#### Historical Events Question Forming
Similar to the Arts & Landmarks folder, this folder includes:
- Methodology of how the Historical Events datasets were created.
- Examples of questions and their respective prompt types.
- Results of model performance for each prompt type.

#### People Question Forming
This folder focuses on the People datasets and includes:
- Methodology of how the People datasets were created.
- Examples of questions and their respective prompt types.
- Results of model performance for each prompt type.
- An additional analysis on whether the cultural or geographical origin of an item affects recall for both models (LLama3 and MISTRAL).

#### Python Model Codes
This folder contains the Python scripts used to run the models on different sections of the datasets. Each script is named to indicate the model, language, question format, and dataset section it handles.

### Notebooks

#### All Merged LLAMA.ipynb
This notebook provides an overview of the LLama3-8B-Instruct model's performance across all datasets and prompt types, along with relevant metrics.

#### All Merged MISTRAL.ipynb
This notebook provides an overview of the MISTRAL-7b-Instruct model's performance across all datasets and prompt types, along with relevant metrics.

#### Exploratory Data Analysis.ipynb
This notebook offers an in-depth overview of the quality of the extracted data, including:
- Number of instances.
- Total number of missing words.
- Imputation of the missing words.
