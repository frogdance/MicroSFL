import os
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import requests
from loguru import logger
import time

# ------------------------------
# Setup
# ------------------------------
app = FastAPI()
os.makedirs("client_models", exist_ok=True)

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Federated server identity
FED_SERVER_ID = os.getenv("FED_SERVER_ID", "fed_server")
FED_SERVER_IP = os.getenv("FED_SERVER_IP", "127.0.0.1:8001")

# ------------------------------
# FC2 Model class
# ------------------------------
class FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(x)

fc2 = None  # global instance

# ------------------------------
# Register federated server with backend
# ------------------------------
def register_with_backend():
    time.sleep(10)
    try:
        resp = requests.post(
            f"{BACKEND_URL}/register_fed",
            json={"client_id": FED_SERVER_ID, "ip": FED_SERVER_IP},
            timeout=5
        )
        if resp.status_code == 200:
            logger.success(f"Fed server registered with IP {FED_SERVER_IP}")
        else:
            logger.warning(f"Failed to register fed server: {resp.text}")
    except Exception as e:
        logger.error(f"Error registering fed server: {e}")

@app.on_event("startup")
async def startup_event():
    register_with_backend()

# ------------------------------
# Internal helper: get client info from backend
# ------------------------------
def get_all_clients(role: str | None = "client"):
    try:
        params = {"role": role} if role else None
        resp = requests.get(f"{BACKEND_URL}/client_info", params=params, timeout=15)
        return resp.json().get("clients", [])
    except Exception as e:
        logger.error(f"Failed to get client info: {e}")
        return []
    
clients = get_all_clients()

# ------------------------------
# Delivery helpers (internal, not API)
# ------------------------------
def delivery_single(client, file_path):
    url = f"http://{client['ip']}/load_fc2"  # client API nhận file
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        try:
            resp = requests.post(url, files=files)
            logger.info(f"Delivered model to client {client['client_id']}: {resp.json()}")
        except Exception as e:
            logger.error(f"Failed to deliver to {client['client_id']}: {e}")

def delivery_all(file_path):
    for client in clients:
        delivery_single(client, file_path)

    time.sleep(2)

# ------------------------------
# API receive model from client
# ------------------------------
@app.post("/upload_model")
async def upload_model(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    filename: str = Form(...)
):
    unique_filename = f"{client_id}_{filename}"
    file_path = os.path.join("client_models", unique_filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"Model received from client {client_id}, saved as {unique_filename}")
    return {"status": "saved", "filename": unique_filename}

# ------------------------------
# Aggregate models
# ------------------------------
@app.get("/aggregate")
def aggregate():
    files = os.listdir("client_models")
    if not files:
        logger.warning("No models to aggregate")
        return {"status": "no models"}

    weighted_sum = None
    total_data = 0

    # Weighted average
    for f in files:
        path = os.path.join("client_models", f)
        model_info = torch.load(path)  # {"state_dict": ..., "num_data": ...}
        state_dict = model_info["state_dict"]
        num_data = model_info["num_data"]

        if weighted_sum is None:
            weighted_sum = {k: v.clone() * num_data for k, v in state_dict.items()}
        else:
            for k in weighted_sum:
                weighted_sum[k] += state_dict[k] * num_data
        total_data += num_data

    avg_state = {k: v / total_data for k, v in weighted_sum.items()}

    # Save aggregate model
    aggregate_file = "fc2.pt"
    torch.save(avg_state, aggregate_file)
    logger.success(f"Aggregated {len(files)} models")

    # Clear client_models folder
    # shutil.rmtree("client_models")
    os.makedirs("client_models", exist_ok=True)

    # Auto-deliver to all clients
    delivery_all(aggregate_file)

    return {"status": "aggregated", "aggregate_file": aggregate_file}

# ------------------------------
# Initialize model (reset)
# ------------------------------
def initialize_model():
    global fc2
    fc2 = FC2()
    # Save to file
    init_file = "fc2.pt"
    torch.save(fc2.state_dict(), init_file)
    logger.info("FC2 model initialized")
    # Auto-deliver to clients
    delivery_all(init_file)

@app.post("/initialize")
async def initialize_fc2():
    try:
        initialize_model()
        return JSONResponse(content={"status": "FC2 initialized and delivered"})
    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)