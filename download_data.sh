#!/usr/bin/env bash

set -e

# You need to install kaggle using pip and then have a valid ~/.kaggle/kaggle.json, that you can 
# generate from "Create new API token" on your account page in kaggle.com
# Alternatively, the dataset can be downloaded manually at 
# https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
rm -rf local_datasets
mkdir local_datasets
cd local_datasets

kaggle datasets download -d crowdflower/twitter-airline-sentiment

unzip twitter-airline-sentiment.zip -d twitter-airline-sentiment
