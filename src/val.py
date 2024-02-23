from pathlib import Path
from utils import load_model


model_path = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/models/efficientnet-b0_binary=True_2024-02-23T15:40:00.523")
model, config = load_model(model_path)
print(model.head)
