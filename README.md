---
title: Sentiment Analysis On Encrypted Data Using Fully Homomorphic Encryption
emoji: 🥷💬
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
tags: [FHE, PPML, privacy, privacy preserving machine learning, homomorphic encryption, security]
python_version: 3.10.11
---

# Sentiment Analysis With FHE

## Set up the app locally

- First, create a virtual env and activate it:

```bash
conda create --name sentiment_analysis_demo python=3.10.11
conda activate sentiment_analysis_demo
```

- Then, install required packages:

```bash
pip3 install pip --upgrade
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```

Check it finish well (with a "Done!"). Please note that the actual model initialization and training 
can be found in the [SentimentClassification notebook](SentimentClassification.ipynb) (see below).

## Compile the FHE algorithm

### Download data

```shell
pip3 install kaggle
./0_download_data.sh
```

### Compile

```bash
python3 1_build/main.py
```


### Launch the app locally

- In a terminal:

```bash
python3 2_run/app.py
```

## Launch the app with docker

TODO

## Interact with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/` in the 
terminal).