# Makefile for Pollutant Prediction Project

# Create and activate conda environment, install dependencies
setup:
	conda create -n bbsr-env python=3.12 -y
	conda activate bbsr-env && pip install -r requirements.txt

# Train the pollutant classifier
train:
	PYTHONPATH=src python scripts/train_pollutant_model.py

# Run inference on new data
inference:
	PYTHONPATH=src python scripts/inference_pipeline.py

# Generate plots and export top predictions
visualize:
	PYTHONPATH=src python scripts/visualize_results.py

# Run everything (except setup)
all: train inference visualize