import os
import uuid
import asyncio
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from loguru import logger
import time
# -----------------------
# Config & Env
# -----------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SPLIT_SERVER_ID = os.getenv("SPLIT_SERVER_ID", "split_server")
SPLIT_SERVER_IP = os.getenv("SPLIT_SERVER_IP", "127.0.0.1:8002")

# -----------------------
# Model fc1
# -----------------------
class FC1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)

    def forward(self, x):
        return torch.relu(self.fc1(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fc1 = FC1().to(device)
optimizer_fc1 = optim.SGD(fc1.parameters(), lr=0.0001)

# -----------------------
# Setup FastAPI
# -----------------------
app = FastAPI()
semaphore = asyncio.Semaphore(4)  # tối đa 4 forward song song
os.makedirs("fc1_grad", exist_ok=True)

# -----------------------
# Data models
# -----------------------
class ForwardInput(BaseModel):
    data: List[List[float]]  # batch z (N,264)


class GradInput(BaseModel):
    grad: List[List[float]]  # grad wrt fc1_out (N,128)
    forward_id: str


# -----------------------
# State lưu intermediate
# -----------------------
pending = {}  # forward_id -> fc1_out tensor

# -----------------------
# Register with backend
# -----------------------
def register_with_backend():
    time.sleep(10)
    try:
        resp = requests.post(
            f"{BACKEND_URL}/register_split",
            json={"client_id": SPLIT_SERVER_ID, "ip": SPLIT_SERVER_IP},
            timeout=5,
        )
        if resp.status_code == 200:
            logger.success(f"Split server registered with IP {SPLIT_SERVER_IP}")
        else:
            logger.warning(f"Failed to register split server: {resp.text}")
    except Exception as e:
        logger.error(f"Error registering split server: {e}")


@app.on_event("startup")
async def startup_event():
    register_with_backend()

# -----------------------
# API forward
# -----------------------
@app.post("/forward")
async def forward(inp: ForwardInput):
    async with semaphore:
        forward_id = f"{len(pending)}_{uuid.uuid4()}"
        x = torch.tensor(inp.data, dtype=torch.float32, device=device)
        out = fc1(x)
        out.retain_grad()
        pending[forward_id] = out
        return {"forward_id": forward_id, "fc1_out": out.detach().cpu().tolist()}

# -----------------------
# API backward
# -----------------------
@app.post("/backward")
async def backward(grad_inp: GradInput):
    forward_id = grad_inp.forward_id
    if forward_id not in pending:
        raise HTTPException(status_code=404, detail="Forward ID not found")

    out = pending.pop(forward_id)
    grad_tensor = torch.tensor(grad_inp.grad, dtype=torch.float32, device=device)
    out.backward(grad_tensor)

    # lưu grad fc1
    grads_fc1 = [p.grad.clone().cpu() for p in fc1.parameters()]
    torch.save(grads_fc1, f"fc1_grad/{forward_id}.pt")

    # zero grad
    fc1.zero_grad()

    return {"status": "backward_done"}

@app.post("/clear_pending")
def clear_pending():
    # ---- Reset pending ----
    pending.clear()

# -----------------------
# API accumulate
# -----------------------
@app.post("/accumulate")
def accumulate():
    # ---- Reset pending ----
    pending.clear()

    files = os.listdir("fc1_grad")
    if not files:
        return {"status": "no grads"}

    accum_fc1 = [torch.zeros_like(p) for p in fc1.parameters()]
    for f in files:
        grads = torch.load(os.path.join("fc1_grad", f))
        for i, g in enumerate(grads):
            accum_fc1[i] += g.to(device)
    for p, g in zip(fc1.parameters(), accum_fc1):
        p.grad = g.to(device)

    optimizer_fc1.step()
    optimizer_fc1.zero_grad()

    # clear folder
    for f in files:
        os.remove(os.path.join("fc1_grad", f))

    return {"status": "accumulated", "num_grads": len(files)}
