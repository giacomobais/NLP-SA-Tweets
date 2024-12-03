# NLP-customer-classification (WIP)
 
Simple project that implements a full ML pipeline. This project is a work in progress.
The projects includes usage of Wandb for model tracking and hyperparameter tuning. The trained model is deployed through FastAPI and is containerized using Docker. 

# Training

To train the model, run the train.py script, passing a boolean argument that determines whether to perform hyperparameter tuning through WandB. Hyperparameters for the sweep are contained in the sweep_config.yaml file.
