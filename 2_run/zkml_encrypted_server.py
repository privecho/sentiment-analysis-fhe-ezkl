# https://docs.ezkl.xyz/
# https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_all_public.ipynb
import struct
import uuid

import numpy as np
from torch import nn
import ezkl
import os
import json
import torch
import base64
from concrete.ml.deployment import FHEModelServer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

evaluation_key = None


# Defines the model
class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()

        # Load the model
        self.fhe_model = FHEModelServer("../deployment/sentiment_fhe_model")

    def forward(self, x):
        # print(f"forward input: {x}")

        # Convert to bytes
        x = x[0]
        _encrypted_encoding = x.numpy().tobytes()
        prediction = self.fhe_model.run(_encrypted_encoding, evaluation_key)
        # print(f"forward prediction hex: {prediction.hex()}")

        byte_tensor = torch.tensor(list(prediction), dtype=torch.uint8)
        # print(f"tensor_output: {byte_tensor}")

        return byte_tensor


class ZKProofRequest(BaseModel):
    encrypted_encoding: str
    evaluation_key: str


circuit = AIModel()


@app.post("/get_zk_proof")
async def get_zk_proof(request: ZKProofRequest):
    request.encrypted_encoding = base64.b64decode(request.encrypted_encoding)
    request.evaluation_key = base64.b64decode(request.evaluation_key)

    global evaluation_key
    evaluation_key = request.evaluation_key

    folder_path = f"zkml_encrypted/{str(uuid.uuid4())}"
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
    x = torch.tensor(list([request.encrypted_encoding]), dtype=torch.uint8)

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
    rpc_url = "http://172.18.38.166:10001"
    # rpc_url = "http://103.231.86.33:10219"
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
    # TODO verify failed. It may be because the proof is too large.
    # res = await ezkl.verify_evm(
    #     addr_verifier=verify_contract_addr,
    #     proof_path=proof_path,
    #     rpc_url=rpc_url
    # )
    # assert res is True
    # print("verified on chain")

    # Read proof file content
    with open(proof_path, 'rb') as f:
        proof_content = base64.b64encode(f.read()).decode('utf-8')

    return {"output": array_to_hex_string(output_data)[:100],
            "output_path": output_path,
            "proof": proof_content[:100],
            "proof_path": proof_path,
            "verify_contract_addr": verify_contract_addr}


def array_to_hex_string(array):
    hex_string = ''.join(format(num, '02x') for num in array)
    return hex_string
