import pandas as pd
from concrete.ml.deployment import FHEModelDev
from concrete.ml.common.serialization.loaders import load
import shutil
from pathlib import Path


script_dir = Path(__file__).parent

DEPLOYMENT_DIR = script_dir / "deployment"

print("Compiling the model...")

with (DEPLOYMENT_DIR / "serialized_model").open("r") as file:
    model = load(file)

# Load the data from the csv file to be used for compilation
data = pd.read_csv(DEPLOYMENT_DIR / "samples_for_compilation.csv", index_col=0).values

# Compile the model
model.compile(data)

dev_model_path = DEPLOYMENT_DIR / "sentiment_fhe_model"

# Delete the deployment folder if it exist
if dev_model_path.is_dir():
    shutil.rmtree(dev_model_path)

fhe_api = FHEModelDev(
    model=model, path_dir=dev_model_path
)
fhe_api.save(via_mlir=True)


print("Done!")
