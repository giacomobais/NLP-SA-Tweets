# NLP-SentimentAnalysis-Tweets (WIP)
 
Simple project that implements a full ML pipeline. The aim of the project is to train an encoder block (i.e. BERT) sentiment analyzer on a tweets dataset. Then, deploy the model using FastAPI. In the project are included different tools used in MLOps, like model tracking using WandB, containeraization using Docker and CI using Github Actions.
This project is a work in progress.

# Training

To train the model, run the train.py script, passing a boolean argument that determines whether to perform hyperparameter tuning through WandB. Hyperparameters for the sweep are contained in the sweep_config.yaml file.
