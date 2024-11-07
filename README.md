MLOPS - Credit Card Fraud Detection
==============================

## Data Source
Data used in this project is sourced from the [Capital One Data Science Challenge GitHub Repository](https://github.com/CapitalOneRecruiting/DS).
<img src="https://github.com/user-attachments/assets/0b8d2663-ef94-42b7-9c9a-1f3ad6eb0bfd" alt="Challenge Image" width="400"/>

This Repo is dedicated to end-to-end Machine Learning Project with MLOps

## DVC Pipeline Execution command:

```bash
python3 -m venv .mlops_venv  # Create a new virtual environment in the .mlops_venv directory
source .mlops_venv/bin/activate  # Activate the virtual environment

pip install -e .  # Install the current package in editable mode

dvc init  # Initialize a new DVC repository

dvc dag  # Display the DVC pipeline as a directed acyclic graph (DAG)

# To execute a machine learning pipeline defined in DVC, you can use the following command
# This will execute Data Preprocessing, Feature Engineering, Model Training, and Evaluation stages
# as defined in the dvc.yaml file, in the correct order and only if there are changes
dvc repro

# Add Google Drive as a remote storage for DVC
# Replace 'myremote' with your preferred remote name
# Replace 'folder_id' with the actual ID of your Google Drive folder
dvc remote add -d myremote gdrive://folder_id/path/to/dvc/storage

python3 src/gdrive_setup/setup_dvc_remote.py  # Run a script to set up the DVC remote configuration with gdrve client secret keys

git push  # Push dvc data changes to the Google drive 
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

# notebook description
--------
- `1_load_data_exploration.ipynb`: Jupyter Notebook for loading and understanding the dataset.
- `2_data_visualization.ipynb`: Jupyter Notebook for data visualization and plotting.
- `3_data_wrangling_modeling.ipynb`: Jupyter Notebook for data wrangling, EDA, data preparation, and building machine learning models.
- `4-model-testing.ipynb`: Jupyter Notebook for model testing.
- `5-model-deployment.ipynb`: Jupyter Notebook for data visualization and plotting (in progress).

- **Notebook 1: 1_load_data_exploration.ipynb**: 
In this initial notebook, I focused on establishing a strong foundation for the project. I meticulously loaded the dataset from github file and extracted the zip file in data folder, ensuring its integrity and consistency. This step was crucial to ensure that subsequent analyses and modeling were built upon reliable data. Afterwards, I worked on basic data exploration of Categorical, Numerical, and Datetime attributes and data structure.

- **Notebook 2: 2_data_visualization.ipynb**: 
With a solid foundation in place, I delved into the world of data visualization. This notebook was dedicated to unraveling the hidden patterns within the dataset. Through an array of plots, charts, and visualizations, I deciphered the distribution of features, uncovered potential correlations, and gained crucial insights into the underlying trends. These visual revelations served as guiding lights for subsequent decision-making.

- **Notebook 3: 3_data_wrangling_modeling.ipynb**: 
In the final phase of my exploration, I undertook comprehensive data wrangling and modeling endeavors. This notebook encapsulated the essence of my project, combining the insights from previous notebooks into actionable steps. Here, I embarked on an intricate journey: 
    - Duplicate Transaction Identification: I delved into the identification and analysis of multi-swipe and reversed duplicate transactions. This endeavor provided a deeper understanding of these transactions' impact on the overall dataset. 
    - Feature Engineering, Cleaning, and Normalization: With an eye for improvement, I engaged in feature engineering to harness the latent potential of the dataset. Additionally, I handled missing values and employed normalization techniques to ensure data consistency and reliability.
    - Effective Imbalanced Data Handling: Recognizing the importance of tackling data imbalance, I implemented an undersampling strategy with n iterations. This method effectively addressed the challenge while retaining the integrity of the dataset.
    - Advanced Modeling with Rigorous Evaluation: Armed with well-preprocessed data, I ventured into modeling armed with cross-validation and hyperparameter tuning. Rigorous evaluation using key metrics helped ascertain the model's performance and suitability for the fraud detection task.


--------
## Future Work:<br>

#### Data Preprocessing
- Implement MICE (Multiple Imputation by Chained Equations) for missing value imputation
- Apply various data transformation techniques on right-skewed attributes
- Utilize PCA (Principal Component Analysis) for dimensionality reduction

#### Statistical Analysis
- Conduct statistical tests such as hypothesis testing, t-tests, and F-statistics among features

#### Advanced Techniques
- **Clustering for Data Segmentation**: Apply algorithms like K-Means or DBSCAN to segment data into meaningful clusters, using cluster labels as additional features
- **Fraud Trend Analysis**: Identify temporal and transaction-related patterns specific to fraudulent activities
- **Iterative Undersampling**: Perform undersampling for each cluster to balance class distribution while maintaining dataset diversity

#### Model Development
- **Model Selection and Tuning**: Explore various classification models (e.g., Random Forest, Gradient Boosting, XGBoost, Support Vector Machines) with hyperparameter tuning for each cluster
- **Ensemble Strategies**: Implement techniques like stacking to combine predictions from different models, weighting them based on performance and cluster association

#### Evaluation and Monitoring
- Regularly evaluate models on validation and holdout sets
- Implement monitoring mechanisms to detect model degradation or concept drift

#### Feature Engineering
- Create time-based features, transaction frequency metrics, and transaction value ratios

#### Continuous Improvement
- Update and refine the model with new data
- Stay informed about new techniques and research in fraud detection
