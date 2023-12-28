<p align="center">
  <a href="https://www.udacity.com/course/data-scientist-nanodegree--nd025">
    <img src="./assets/disaster-response-project1.png" style="cursor: zoom-in; width: 100%; max-width: 800px" alt="Bootstrap logo" >
  </a>
</p>

## Disaster Response Pipeline
This repo contains the implementations of the Disaster Response Pipeline project, which is part of the Udacity Data Scientist Program.

## Table of contents

- [Motivation](#motivation)
- [Installation](#installation)
- [Structure](#structure)
- [Usage](#usage)
- [Limitations](#limitations)

## Motivation
This project is part of the Udacity Data Scientist Program (Data Engineering). I followed the instructions to build ETL, NLP and machine learning pipelines during the course, and used the code and skills learned to complete this project.

## Installation
Please note that to reproduce the results, you need to install the exact versions of the libraries using the pip:
```
pip install -r requirements.txt
```
I recommend to install into a virtual environment like Anaconda.

## Structure
The folder structure of this repo is as follows:
```
|-app
  |-templates				# html templates 
    |-run.py				# script to run web demo
|-data
  |-disaster_categories.csv	# original categories dataset
  |-disaster_messages.csv	# original messages dataset
  |-process_data.py			# ETL pipeline script
  |-ETL Pipeline Preparation.ipynb	# the ETL pipeline preparation notebook
|-models
  |-train_classifier.py		# NLP&ML pipeline script
  |-ML Pipeline Preparation.ipynb	# the NLP&ML pipeline preparation notebook
```

## Usage
1. Run the following commands in the project's root directory to set up your database and model.
- To run ETL pipeline that cleans data and stores in database
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
- To run ML pipeline that trains classifier and saves
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

5. Enter a message and click the green "Classify Message" button to see the results 

## Limitations
This implementation aims to record my learning path and made as a homework and contains several limitations:
- <b>The dataset is heavily unbalanced.</b> Some categories have very small number of positive samples (even 0 positive sample).
For categories with 0 positive samples, the classification results are meaningless.
For categories with small amount of positive samples, other technics like over-sampling need to be introduced during the training process.
- <b>The model parameters could be further tuned.</b> Since Udacity offers limited on-line computing resources, the grid search had to be done
in a small space. More optimized parameters could be retrived with added computing power.
