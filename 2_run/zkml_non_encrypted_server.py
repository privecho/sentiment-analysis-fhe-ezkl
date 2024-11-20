# https://docs.ezkl.xyz/
# https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_all_public.ipynb
import pickle
import struct
import uuid

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from torch import nn
import ezkl
import os
import json
import torch
import base64
from concrete.ml.deployment import FHEModelServer
from concrete.ml.sklearn import XGBClassifier
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

evaluation_key = None


# Defines the model
class AIWordsModel(nn.Module):
    def __init__(self):
        super(AIWordsModel, self).__init__()

        print("init ZK AIWordsModel")

        # Load the model
        self.model = XGBClassifier()
        train = pd.read_csv("../dataset/local_datasets/twitter-airline-sentiment/Tweets.csv", index_col=0)
        text_X = train["text"]
        y = train["airline_sentiment"].replace(["negative", "neutral", "positive"], [0, 1, 2])

        # Load the tokenizer and model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ).to(self.device)

        text_X_train, text_X_test, y_train, y_test = train_test_split(
            text_X, y, test_size=0.1, random_state=42
        )
        X_train_transformer = self.text_to_tensor(text_X_train.tolist(), self.transformer_model, self.tokenizer,
                                                  self.device)

        with open("../deployment/serialized_model_zkml", 'rb') as file:  # Open in binary read mode
            loaded_data = pickle.load(file)
            self.model.load_dict(loaded_data)
            parameters = {"n_bits": [2, 3], "max_depth": [1], "n_estimators": [10, 30, 50]}
            grid_search2 = GridSearchCV(self.model, parameters, cv=5, scoring="accuracy")
            grid_search2.fit(X_train_transformer, y_train)
            self.best_model2 = grid_search2.best_estimator_
            self.best_model2.load_dict(loaded_data)
            self.best_model2.compile(X_train_transformer)

        print(f"loaded_data finished")

    def forward(self, x):
        prediction = self.best_model2.predict_proba(x, fhe="execute")

        prediction_tensor = torch.tensor(prediction, dtype=torch.float32)
        prediction_tensor = prediction_tensor.squeeze()  # Remove extra dimensions if any

        return prediction_tensor

    # Function to convert text to tensor
    def text_to_tensor(self, list_text, transformer_model, tokenizer, device):
        tokenized_text = [tokenizer.encode(text, return_tensors="pt") for text in list_text]
        output_hidden_states_list = [None] * len(tokenized_text)

        for i, tokenized_x in enumerate(tqdm.tqdm(tokenized_text)):
            output_hidden_states = transformer_model(tokenized_x.to(device), output_hidden_states=True)[1][-1]
            output_hidden_states = output_hidden_states.mean(dim=1).detach().cpu().numpy()
            output_hidden_states_list[i] = output_hidden_states

        return np.concatenate(output_hidden_states_list, axis=0)


class ZKProofRequest(BaseModel):
    text: str


circuit = AIWordsModel()


@app.post("/get_zk_proof")
async def get_zk_proof(request: ZKProofRequest):
    folder_path = f"zkml_non_encrypted/{str(uuid.uuid4())}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model_path = os.path.join(f'{folder_path}/network.onnx')
    compiled_model_path = os.path.join(f'{folder_path}/network.compiled')
    pk_path = os.path.join(f'{folder_path}/test.pk')
    vk_path = os.path.join(f'{folder_path}/test.vk')
    settings_path = os.path.join(f'{folder_path}/settings.json')

    witness_path = os.path.join(f'{folder_path}/witness.json')
    input_data_path = os.path.join(f'{folder_path}/input.json')
    srs_path = os.path.join(f'{folder_path}/kzg14.srs')
    output_path = os.path.join(f'{folder_path}/output.json')

    # After training, export to onnx (network.onnx) and create a data file (input.json)
    words = [request.text]
    x_list = circuit.text_to_tensor(words, circuit.transformer_model, circuit.tokenizer, circuit.device)
    x = torch.tensor(x_list, dtype=torch.float32)

    # Flips the neural net into inference mode
    circuit.eval()

    # Get the output of the model
    with torch.no_grad():
        output = circuit(x)
    # Save the output to a file
    output_data = output.detach().numpy().tolist()
    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    # Export the model
    torch.onnx.export(circuit,  # model being 2_run
                      x,  # model input (or a tuple for multiple inputs)
                      model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    data = dict(input_data=x.tolist())

    # Serialize data into file:
    json.dump(data, open(input_data_path, 'w'))

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"  # "fixed" for params means that the committed to params are used for all proofs

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
    assert res is True

    cal_path = os.path.join(f"{folder_path}/calibration.json")

    # Serialize data into file:
    json.dump(data, open(cal_path, 'w'))

    await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res is True

    # srs path
    res = await ezkl.get_srs(settings_path, srs_path=srs_path)
    assert res is True

    # now generate the witness file
    res = await ezkl.gen_witness(input_data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    # HERE WE SETUP THE CIRCUIT PARAMS
    # WE GOT KEYS
    # WE GOT CIRCUIT PARAMETERS
    # EVERYTHING ANYONE HAS EVER NEEDED FOR ZK

    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path
    )

    assert res is True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # GENERATE A PROOF
    proof_path = os.path.join(f'{folder_path}/test.pf')
    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
        srs_path
    )
    assert os.path.isfile(proof_path)

    # VERIFY IT ON LOCAL
    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        srs_path
    )
    assert res is True
    print("verified on local")

    # VERIFY IT ON CHAIN
    verify_sol_code_path = os.path.join(f'{folder_path}/verify.sol')
    verify_sol_abi_path = os.path.join(f'{folder_path}/verify.abi')
    res = await ezkl.create_evm_verifier(
        vk_path,
        settings_path,
        verify_sol_code_path,
        verify_sol_abi_path,
        srs_path
    )
    assert res is True
    verify_contract_addr_file = f"{folder_path}/addr.txt"
    # rpc_url = "http://172.18.38.166:10001"
    rpc_url = "http://103.231.86.33:10219"
    await ezkl.deploy_evm(
        addr_path=verify_contract_addr_file,
        rpc_url=rpc_url,
        sol_code_path=verify_sol_code_path
    )
    if os.path.exists(verify_contract_addr_file):
        with open(verify_contract_addr_file, 'r') as file:
            verify_contract_addr = file.read()
    else:
        print(f"error: File {verify_contract_addr_file} does not exist.")
        return {"error": "Contract address file not found"}
    res = await ezkl.verify_evm(
        addr_verifier=verify_contract_addr,
        proof_path=proof_path,
        rpc_url=rpc_url
    )
    assert res is True
    print("verified on chain")

    # Read proof file content
    with open(proof_path, 'rb') as f:
        proof_content = base64.b64encode(f.read()).decode('utf-8')

    return {"output": output_data, "proof": proof_content, "verify_contract_addr": verify_contract_addr}
