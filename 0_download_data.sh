#!/usr/bin/env bash

set -e

export KAGGLE_USERNAME=myronzhangweb3
export KAGGLE_KEY=313af1a98e3a4c1beb2331d6b0056105

# You need to install kaggle using pip and then have a valid ~/.kaggle/kaggle.json, that you can 
# generate from "Create new API token" on your account page in kaggle.com
# Alternatively, the dataset can be downloaded manually at 
# https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
rm -rf local_datasets
mkdir -p dataset/local_datasets
cd dataset/local_datasets

kaggle datasets download -d crowdflower/twitter-airline-sentiment

unzip twitter-airline-sentiment.zip -d twitter-airline-sentiment
