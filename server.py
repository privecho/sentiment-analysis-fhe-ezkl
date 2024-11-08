"""Server that will listen for GET requests from the client."""
import json

from fastapi import FastAPI
from joblib import load
from concrete.ml.deployment import FHEModelServer
from pydantic import BaseModel
import base64
from pathlib import Path

from zkml_ezkl import get_zk_proof

current_dir = Path(__file__).parent

# Load the model
fhe_model = FHEModelServer("deployment/sentiment_fhe_model")

class PredictRequest(BaseModel):
    evaluation_key: str
    encrypted_encoding: str

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FHE Model Server!"}

@app.post("/predict_sentiment")
def predict_sentiment(query: PredictRequest):
    encrypted_encoding = base64.b64decode(query.encrypted_encoding)
    evaluation_key = base64.b64decode(query.evaluation_key)
    print(f"predict_sentiment encrypted_encoding: {encrypted_encoding}")
    # print(f"predict_sentiment evaluation_key: {evaluation_key}")
    prediction = fhe_model.run(encrypted_encoding, evaluation_key)
    print(f"prediction: {encrypted_encoding[:100]}")

    # Encode base64 the prediction
    encoded_prediction = base64.b64encode(prediction).decode()
    return {"encrypted_prediction": encoded_prediction}


# @app.post("/zk_proof")
# def zk_proof(query: PredictRequest):
#     encrypted_encoding = base64.b64decode(query.encrypted_encoding)
#     evaluation_key = base64.b64decode(query.evaluation_key)

#     print(f"encrypted_encoding: {encrypted_encoding}")
#     proof_path = get_zk_proof(encrypted_encoding, evaluation_key)
#     return {"proof_path": proof_path}