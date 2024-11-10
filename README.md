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
python_version: 3.9.6
---

# Sentiment Analysis With FHE

## Set up the app locally

- First, create a virtual env and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- Then, install required packages:

```bash
pip3 install pip --upgrade
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed

# mac z3
brew install z3
pip3 uninstall z3-solver
pip3 install z3-solver
pip3 install more-itertools
```

Check it finish well (with a "Done!"). Please note that the actual model initialization and training 
can be found in the [SentimentClassification notebook](SentimentClassification.ipynb) (see below).

### Launch the app locally

- In a terminal:

```bash
source .venv/bin/activate
python3 app.py
```

## Interact with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/` in the 
terminal).