import pickle
import torch
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from concrete.ml.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import tqdm
from pathlib import Path
from concrete.ml.deployment import FHEModelDev

# Load and prepare the dataset
train = pd.read_csv("../dataset/local_datasets/twitter-airline-sentiment/Tweets.csv", index_col=0)
text_X = train["text"]
y = train["airline_sentiment"].replace(["negative", "neutral", "positive"], [0, 1, 2])

text_X_train, text_X_test, y_train, y_test = train_test_split(
    text_X, y, test_size=0.1, random_state=42
)

# Load the tokenizer and model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
transformer_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
).to(device)


# Function to convert text to tensor
def text_to_tensor(list_text, transformer_model, tokenizer, device):
    tokenized_text = [tokenizer.encode(text, return_tensors="pt") for text in list_text]
    output_hidden_states_list = [None] * len(tokenized_text)

    for i, tokenized_x in enumerate(tqdm.tqdm(tokenized_text)):
        output_hidden_states = transformer_model(tokenized_x.to(device), output_hidden_states=True)[1][-1]
        output_hidden_states = output_hidden_states.mean(dim=1).detach().cpu().numpy()
        output_hidden_states_list[i] = output_hidden_states

    return np.concatenate(output_hidden_states_list, axis=0)


# Vectorize the text
X_train_transformer = text_to_tensor(text_X_train.tolist(), transformer_model, tokenizer, device)
X_test_transformer = text_to_tensor(text_X_test.tolist(), transformer_model, tokenizer, device)

# Train the model
model = XGBClassifier()
parameters = {"n_bits": [2, 3], "max_depth": [1], "n_estimators": [10, 30, 50]}
grid_search = GridSearchCV(model, parameters, cv=5, scoring="accuracy")
grid_search.fit(X_train_transformer, y_train)

# Evaluate the model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_transformer)
matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(matrix).plot()

# FHE Inference
best_model.compile(X_train_transformer)
tested_tweet = ["AirFrance is awesome, almost as much as Zama!"]
X_tested_tweet = text_to_tensor(tested_tweet, transformer_model, tokenizer, device)
decrypted_proba = best_model.predict_proba(X_tested_tweet, fhe="execute")

# Deployment
DEPLOYMENT_DIR = Path("../deployment")
DEPLOYMENT_DIR.mkdir(exist_ok=True)
fhe_api = FHEModelDev(DEPLOYMENT_DIR / "sentiment_fhe_model", best_model)
fhe_api.save(via_mlir=True)
with (DEPLOYMENT_DIR / "serialized_model").open("w") as file:
    best_model.dump(file)

# TODO useless?
with (DEPLOYMENT_DIR / "serialized_model_zkml").open("wb") as file:
    pickle.dump(best_model.dump_dict(), file)