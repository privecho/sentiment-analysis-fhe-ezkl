"""A gradio app. that runs locally (analytics=False and share=False) about sentiment analysis on tweets."""

import gradio as gr
from transformer_vectorizer import TransformerVectorizer
from concrete.ml.deployment import FHEModelClient
import numpy
import os
from pathlib import Path
import requests
import json
import base64
import subprocess
import shutil
import time

# This repository's directory
REPO_DIR = Path(__file__).parent

# Download required data files
subprocess.Popen(["bash", "./download_data.sh"], cwd=REPO_DIR)
subprocess.Popen(["uvicorn", "server:app", "--port", "8000"], cwd=REPO_DIR)
subprocess.Popen(["uvicorn", "zkml_non_encrypted:app", "--port", "8001"], cwd=REPO_DIR)
subprocess.Popen(["uvicorn", "zkml_encrypted:app", "--port", "8002"], cwd=REPO_DIR)

# Wait 30 sec for the server to start
time.sleep(30)

# Encrypted data limit for the browser to display
# (encrypted data is too large to display in the browser)
ENCRYPTED_DATA_BROWSER_LIMIT = 500
N_USER_KEY_STORED = 20
FHE_MODEL_PATH = "deployment/sentiment_fhe_model"

print("Loading the transformer model...")

# Initialize the transformer vectorizer
transformer_vectorizer = TransformerVectorizer()


def clean_tmp_directory():
    # Create tmp directory if it doesn't exist
    Path(".fhe_keys/").mkdir(exist_ok=True)

    # Allow 20 user keys to be stored.
    # Once that limitation is reached, deleted the oldest.
    path_sub_directories = sorted([f for f in Path(".fhe_keys/").iterdir() if f.is_dir()], key=os.path.getmtime)

    user_ids = []
    if len(path_sub_directories) > N_USER_KEY_STORED:
        n_files_to_delete = len(path_sub_directories) - N_USER_KEY_STORED
        for p in path_sub_directories[:n_files_to_delete]:
            user_ids.append(p.name)
            shutil.rmtree(p)

    list_files_tmp = Path("tmp/").iterdir()
    # Delete all files related to user_id
    for file in list_files_tmp:
        for user_id in user_ids:
            if file.name.endswith(f"{user_id}.npy"):
                file.unlink()


def keygen():
    # Clean tmp directory if needed
    clean_tmp_directory()

    print("Initializing FHEModelClient...")

    # Create .fhe_keys directory if it doesn't exist
    Path(".fhe_keys/").mkdir(exist_ok=True)

    # Let's create a user_id
    user_id = numpy.random.randint(0, 2 ** 32)
    fhe_api = FHEModelClient(FHE_MODEL_PATH, f".fhe_keys/{user_id}")
    fhe_api.load()

    # Generate a fresh key
    fhe_api.generate_private_and_evaluation_keys(force=True)
    evaluation_key = fhe_api.get_serialized_evaluation_keys()

    # Save evaluation_key in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    numpy.save(f"tmp/tmp_evaluation_key_{user_id}.npy", evaluation_key)

    return [list(evaluation_key)[:ENCRYPTED_DATA_BROWSER_LIMIT], user_id]


def encode_quantize_encrypt(text, user_id):
    if not user_id:
        raise gr.Error("You need to generate FHE keys first.")

    fhe_api = FHEModelClient(FHE_MODEL_PATH, f".fhe_keys/{user_id}")
    fhe_api.load()
    encodings = transformer_vectorizer.transform([text])
    quantized_encodings = fhe_api.model.quantize_input(encodings).astype(numpy.uint8)
    encrypted_quantized_encoding = fhe_api.quantize_encrypt_serialize(encodings)

    # Save encrypted_quantized_encoding in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    numpy.save(f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy", encrypted_quantized_encoding)

    # Compute size
    encrypted_quantized_encoding_shorten = list(encrypted_quantized_encoding)
    encrypted_quantized_encoding_shorten_hex = ''.join(f'{i:02x}' for i in encrypted_quantized_encoding_shorten)
    return (
        encodings[0],
        quantized_encodings[0],
        encrypted_quantized_encoding_shorten_hex,
    )


def run_fhe(user_id):
    encoded_data_path = Path(f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy")
    if not user_id:
        raise gr.Error("You need to generate FHE keys first.")
    if not encoded_data_path.is_file():
        raise gr.Error("No encrypted data was found. Encrypt the data before trying to predict.")

    # Read encrypted_quantized_encoding from the file
    encrypted_quantized_encoding = numpy.load(encoded_data_path)

    # Read evaluation_key from the file
    evaluation_key = numpy.load(f"tmp/tmp_evaluation_key_{user_id}.npy")

    # Use base64 to encode the encodings and evaluation key
    encrypted_quantized_encoding = base64.b64encode(encrypted_quantized_encoding).decode()
    encoded_evaluation_key = base64.b64encode(evaluation_key).decode()

    query = {}
    query["evaluation_key"] = encoded_evaluation_key
    query["encrypted_encoding"] = encrypted_quantized_encoding
    headers = {"Content-type": "application/json"}
    response = requests.post(
        "http://localhost:8000/predict_sentiment", data=json.dumps(query), headers=headers
    )
    encrypted_prediction = base64.b64decode(response.json()["encrypted_prediction"])

    # Save encrypted_prediction in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    numpy.save(f"tmp/tmp_encrypted_prediction_{user_id}.npy", encrypted_prediction)
    encrypted_prediction_shorten = list(encrypted_prediction)
    encrypted_prediction_shorten_hex = ''.join(f'{i:02x}' for i in encrypted_prediction_shorten)
    return encrypted_prediction_shorten_hex


def decrypt_prediction(user_id):
    encoded_data_path = Path(f"tmp/tmp_encrypted_prediction_{user_id}.npy")
    if not user_id:
        raise gr.Error("You need to generate FHE keys first.")
    if not encoded_data_path.is_file():
        raise gr.Error("No encrypted prediction was found. Run the prediction over the encrypted data first.")

    # Read encrypted_prediction from the file
    encrypted_prediction = numpy.load(encoded_data_path).tobytes()

    fhe_api = FHEModelClient(FHE_MODEL_PATH, f".fhe_keys/{user_id}")
    fhe_api.load()

    # We need to retrieve the private key that matches the client specs (see issue #18)
    fhe_api.generate_private_and_evaluation_keys(force=False)

    predictions = fhe_api.deserialize_decrypt_dequantize(encrypted_prediction)
    return {
        "negative": predictions[0][0],
        "neutral": predictions[0][1],
        "positive": predictions[0][2],
    }


def get_zk_proof_non_encrypted(text):
    headers = {"Content-type": "application/json"}
    query = {"text": text}
    response = requests.post(
        "http://localhost:8001/get_zk_proof", data=json.dumps(query), headers=headers
    )
    result = response.json()

    sentiment = ""
    if result["output"][0] > 0.5:
        sentiment = "negative"
    elif result["output"][1] > 0.5:
        sentiment = "neutral"
    else:
        sentiment = "positive"

    return sentiment, result["proof"], result["verify_contract_addr"]


def get_zk_proof_encrypted(user_id):
    encoded_data_path = Path(f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy")
    if not user_id:
        raise gr.Error("You need to generate FHE keys first.")
    if not encoded_data_path.is_file():
        raise gr.Error("No encrypted data was found. Encrypt the data before trying to predict.")

    # Read encrypted_quantized_encoding from the file
    encrypted_quantized_encoding = numpy.load(encoded_data_path)

    # Read evaluation_key from the file
    evaluation_key = numpy.load(f"tmp/tmp_evaluation_key_{user_id}.npy")

    # Use base64 to encode the encodings and evaluation key
    encrypted_quantized_encoding = base64.b64encode(encrypted_quantized_encoding).decode()
    encoded_evaluation_key = base64.b64encode(evaluation_key).decode()

    query = {}
    query["evaluation_key"] = encoded_evaluation_key
    query["encrypted_encoding"] = encrypted_quantized_encoding
    headers = {"Content-type": "application/json"}
    response = requests.post(
        "http://localhost:8002/get_zk_proof", data=json.dumps(query), headers=headers
    )
    result = response.json()
    return result["output"], result["proof"], result["verify_contract_addr"]


demo = gr.Blocks()

print("Starting the demo...")
with demo:
    gr.Markdown(
        """
<p align="center">
    <img width=200 src="https://user-images.githubusercontent.com/5758427/197816413-d9cddad3-ba38-4793-847d-120975e1da11.png">
</p>

<h2 align="center">Sentiment Analysis On Encrypted Data Using Homomorphic Encryption</h2>

<p align="center">
    <a href="https://github.com/zama-ai/concrete-ml"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197972109-faaaff3e-10e2-4ab6-80f5-7531f7cfb08f.png">Concrete-ML</a>
    —
    <a href="https://docs.zama.ai/concrete-ml"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197976802-fddd34c5-f59a-48d0-9bff-7ad1b00cb1fb.png">Documentation</a>
    —
    <a href="https://zama.ai/community"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197977153-8c9c01a7-451a-4993-8e10-5a6ed5343d02.png">Community</a>
    —
    <a href="https://twitter.com/zama_fhe"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197975044-bab9d199-e120-433b-b3be-abd73b211a54.png">@zama_fhe</a>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/56846628/219329304-6868be9e-5ce8-4279-9123-4cb1bc0c2fb5.png" width="60%" height="60%">
</p>
"""
    )

    gr.Markdown(
        """
        <p align="center">
        </p>
        <p align="center">
        </p>
        """
    )

    gr.Markdown("## Notes")
    gr.Markdown(
        """
    - The private key is used to encrypt and decrypt the data and shall never be shared.
    - The evaluation key is a public key that the server needs to process encrypted data.
    """
    )

    gr.Markdown("# Step 1: Generate the keys")

    b_gen_key_and_install = gr.Button("Generate the keys and send public part to server")

    evaluation_key = gr.Textbox(
        label="Evaluation key (truncated):",
        max_lines=4,
        interactive=False,
    )

    user_id = gr.Textbox(
        label="",
        max_lines=4,
        interactive=False,
        visible=False
    )

    gr.Markdown("# Step 2: Provide a message")
    gr.Markdown("## Client side")
    gr.Markdown(
        "Enter a sensitive text message you received and would like to do sentiment analysis on (ideas: the last text message of your boss.... or lover)."
    )
    text = gr.Textbox(label="Enter a message:", value="I really like your work recently")

    gr.Markdown("# Step 3: Encode the message with the private key")
    b_encode_quantize_text = gr.Button(
        "Encode, quantize and encrypt the text with transformer vectorizer, and send to server"
    )

    with gr.Row():
        encoding = gr.Textbox(
            label="Transformer representation:",
            max_lines=4,
            interactive=False,
        )
        quantized_encoding = gr.Textbox(
            label="Quantized transformer representation:", max_lines=4, interactive=False
        )
        encrypted_quantized_encoding = gr.Textbox(
            label="Encrypted quantized transformer representation (truncated):",
            max_lines=4,
            interactive=False,
        )

    gr.Markdown("# Step 4: Run the FHE evaluation")
    gr.Markdown("## Server side")
    gr.Markdown(
        "The encrypted value is received by the server. Thanks to the evaluation key and to FHE, the server can compute the (encrypted) prediction directly over encrypted values. Once the computation is finished, the server returns the encrypted prediction to the client."
    )

    b_run_fhe = gr.Button("Run FHE execution there")
    encrypted_prediction = gr.Textbox(
        label="Encrypted prediction (truncated):",
        max_lines=4,
        interactive=False,
    )

    gr.Markdown("# Step 5: Decrypt the sentiment")
    gr.Markdown("## Client side")
    gr.Markdown(
        "The encrypted sentiment is sent back to client, who can finally decrypt it with its private key. Only the client is aware of the original tweet and the prediction."
    )
    b_decrypt_prediction = gr.Button("Decrypt prediction")

    labels_sentiment = gr.Label(label="Sentiment:")

    gr.Markdown("# Step 6: Get ZK Proof(non-encrypted input)")
    gr.Markdown("## Server side")
    gr.Markdown(
        "Get zero-knowledge proof of the sentiment analysis computation (for non-encrypted input)."
    )
    b_get_zk_proof_non_encrypted = gr.Button("Get ZK Proof(non-encrypted input)")

    with gr.Row():
        zk_sentiment_non_encrypted = gr.Textbox(
            label="Sentiment:",
            max_lines=1,
            interactive=False,
        )
        zk_proof_non_encrypted = gr.Textbox(
            label="ZK Proof:",
            max_lines=4,
            interactive=False,
        )
        zk_contract_non_encrypted = gr.Textbox(
            label="Verify Contract Address:",
            max_lines=1,
            interactive=False,
        )

    gr.Markdown("# Step 6: Get ZK Proof(encrypted input)")
    gr.Markdown("## Server side")
    gr.Markdown(
        "Get zero-knowledge proof of the sentiment analysis computation (for encrypted input)."
    )
    b_get_zk_proof_encrypted = gr.Button("Get ZK Proof(encrypted input)")

    with gr.Row():
        zk_encrypted_prediction = gr.Textbox(
            label="Encrypted Prediction(same as Step 4 output):",
            max_lines=1,
            interactive=False,
        )
        zk_proof_encrypted = gr.Textbox(
            label="ZK Proof:",
            max_lines=4,
            interactive=False,
        )
        zk_contract_encrypted = gr.Textbox(
            label="Verify Contract Address:",
            max_lines=1,
            interactive=False,
        )

    # Button for key generation
    b_gen_key_and_install.click(keygen, inputs=[], outputs=[evaluation_key, user_id])

    # Button to quantize and encrypt
    b_encode_quantize_text.click(
        encode_quantize_encrypt,
        inputs=[text, user_id],
        outputs=[
            encoding,
            quantized_encoding,
            encrypted_quantized_encoding,
        ],
    )

    # Button to send the encodings to the server using post at (localhost:8000/predict_sentiment)
    b_run_fhe.click(run_fhe, inputs=[user_id], outputs=[encrypted_prediction])

    # Button to decrypt the prediction on the client
    b_decrypt_prediction.click(decrypt_prediction, inputs=[user_id], outputs=[labels_sentiment])

    # Button to get ZK proof(non encrypted)
    b_get_zk_proof_non_encrypted.click(get_zk_proof_non_encrypted, inputs=[text],
                                       outputs=[zk_sentiment_non_encrypted, zk_proof_non_encrypted,
                                                zk_contract_non_encrypted])

    # Button to get ZK proof(encrypted)
    b_get_zk_proof_encrypted.click(get_zk_proof_encrypted, inputs=[user_id],
                                   outputs=[zk_encrypted_prediction, zk_proof_encrypted, zk_contract_encrypted])

    gr.Markdown(
        "The app was built with [Concrete-ML](https://github.com/zama-ai/concrete-ml), a Privacy-Preserving Machine Learning (PPML) open-source set of tools by [Zama](https://zama.ai/). Try it yourself and don't forget to star on Github &#11088;."
    )
demo.launch(share=False)
