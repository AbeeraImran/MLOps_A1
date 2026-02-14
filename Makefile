.PHONY: setup download-data preprocess feature-engineering train predict evaluate all clean
setup:
	pip install -r requirements.txt

download-data:
	python src/download_data.py

preprocess: download-data
	python src/preprocess.py

feature-engineering: preprocess
	python src/feature_engineering.py

train: feature-engineering
	python src/train.py

predict: train
	python src/predict.py

evaluate: predict
	python src/evaluate.py
all: setup download-data preprocess feature-engineering train predict evaluate
clean:
	rmdir /s /q data\processed data\features models results

clean:
	del /q data\raw\* data\processed\* data\features\* models\* results\*