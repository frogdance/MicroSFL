import os
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, File, UploadFile, Query
from pydantic import BaseModel
import requests
from tqdm import tqdm
from loguru import logger
import time 

# -----------------------
# URLs từ env
# -----------------------
BACKEND_URL      = os.getenv("BACKEND_URL", "http://localhost:8000")

CLIENT_ID = os.getenv("CLIENT_ID", "1")
CLIENT_IP = os.getenv("CLIENT_IP", "127.0.0.1:8005")
# DATA_PATH = f"../volumes/client{CLIENT_ID}"
DATA_PATH = "data/"
BATCH_SIZE = 64

# -----------------------
# Đăng ký client với backend
# -----------------------
try:
    time.sleep(5)
    resp = requests.post(
        f"{BACKEND_URL}/client_info",
        json={"client_id": CLIENT_ID, 'ip': CLIENT_IP},
        timeout=5
    )
    if resp.status_code == 200:
        logger.success(f"Client {CLIENT_ID} registered with IP {CLIENT_IP}")
    else:
        logger.warning(f"Failed to register client: {resp.text}")
except Exception as e:
    logger.error(f"Error registering client: {e}")

# -----------------------
# FC2 Model
# -----------------------
class FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc2(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim=64, noise_std=0.1):
        super().__init__()
        self.encoder = nn.Linear(28*28, latent_dim)
        self.noise_std = noise_std
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        # Thêm Gaussian noise
        noise = torch.randn_like(z) * self.noise_std
        z_noisy = z + noise
        return z_noisy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fc2 = FC2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(fc2.parameters(), lr=0.001)

encoder = Encoder(latent_dim=64).to(device)
encoder.load_state_dict(torch.load("./encoder.pth"))
encoder.to(device)
encoder.eval()

# ===== Freeze encoder =====
for param in encoder.parameters():
    param.requires_grad = False

# -----------------------
# Load client dataset
# -----------------------
train_data = torch.load(os.path.join(DATA_PATH, "train.pt"))
train_x = torch.stack([d[0] for d in train_data]).float() / 255.0
train_y = torch.tensor([d[1] for d in train_data], dtype=torch.long)

test_data = torch.load(os.path.join(DATA_PATH, "test.pt"))
test_x = torch.stack([d[0] for d in test_data]).float() / 255.0
test_y = torch.tensor([d[1] for d in test_data], dtype=torch.long)

# -----------------------
# FastAPI client
# -----------------------
app = FastAPI()

class LoadFC2In(BaseModel):
    state_dict: dict

def get_split_server_url():
    resp = requests.get(f"{BACKEND_URL}/client_info", params={"role": "split"})
    clients = resp.json().get("clients", [])
    if not clients:
        raise RuntimeError("No split server found")
    split_ip = clients[0]["ip"]
    return f"http://{split_ip}"

def get_fed_server_url():
    resp = requests.get(f"{BACKEND_URL}/client_info", params={"role": "fed"})
    clients = resp.json().get("clients", [])
    if not clients:
        raise RuntimeError("No fed server found")
    fed_ip = clients[0]["ip"]
    return f"http://{fed_ip}"

@app.post("/train")
def train_client(global_round: int = Query(...)):
    SPLIT_SERVER_URL = get_split_server_url()
    FED_SERVER_URL   = get_fed_server_url()

    fc2.train()
    num_batches = (len(train_x) + BATCH_SIZE - 1) // BATCH_SIZE
    os.makedirs("fc2_grad", exist_ok=True)

    total_loss = 0.0

    pbar = tqdm(range(num_batches), desc="Training FC2", unit="batch")
    for i in pbar:
        batch_x = train_x[i*BATCH_SIZE : min((i+1)*BATCH_SIZE, len(train_x))].to(device)
        batch_y = train_y[i*BATCH_SIZE : min((i+1)*BATCH_SIZE, len(train_y))].to(device)

        with torch.no_grad():
            batch_z = encoder(batch_x)
        # -------- Send to Split server forward --------
        payload = {"data": batch_z.cpu().numpy().tolist()}
        resp = requests.post(f"{SPLIT_SERVER_URL}/forward", json=payload).json()
        forward_id = resp["forward_id"]
        fc1_out = torch.tensor(resp["fc1_out"], dtype=torch.float32, device=device, requires_grad=True)

        # -------- Local FC2 forward --------
        output = fc2(fc1_out)
        loss = criterion(output, batch_y)
        loss.backward()

        # -------- Send grad wrt fc1_out back --------
        grad_to_server = fc1_out.grad.detach().cpu().tolist()
        requests.post(f"{SPLIT_SERVER_URL}/backward", json={"grad": grad_to_server, "forward_id": forward_id})

        # -------- Save FC2 grads để accumulate sau --------
        grads_fc2 = [p.grad.clone().cpu() for p in fc2.parameters()]
        torch.save(grads_fc2, f"fc2_grad/grad_batch_{i}.pt")

        # Zero local grad
        fc1_out.grad = None
        fc2.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        logger.info(f"Batch {i} done, loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    logger.info(f"Round average loss: {avg_loss:.4f}")

    # -------- Gửi lên backend
    try:
        resp_metric = requests.post(
            f"{BACKEND_URL}/metric",
            json={"client_id": CLIENT_ID, "loss": avg_loss, "global_round": global_round},
            timeout=5
        )
        logger.success(f"Average loss sent to backend: {resp_metric.json()}")
    except Exception as e:
        logger.error(f"Failed to send metric: {e}")

    # -------- Accumulate FC2 grads từ tất cả batch --------
    grad_files = os.listdir("fc2_grad")
    accum_fc2 = [torch.zeros_like(p) for p in fc2.parameters()]
    for f in grad_files:
        grads = torch.load(os.path.join("fc2_grad", f))
        for idx, g in enumerate(grads):
            accum_fc2[idx] += g.to(device)
    for p, g in zip(fc2.parameters(), accum_fc2):
        p.grad = g.to(device)

    optimizer.step()
    optimizer.zero_grad()

    logger.info(f"done accumulate {len(grad_files)} files")

    # Clear folder
    for f in grad_files:
        os.remove(os.path.join("fc2_grad", f))

    # Sau khi train xong
    fc2_state = fc2.state_dict()
    num_data = len(train_x)

    # Gói thông tin gửi lên federated server
    payload = {"state_dict": fc2_state, "num_data": num_data}
    file_path = f"fc2.pt"

    # Lưu bằng torch.save
    torch.save(payload, file_path)

    # Gửi file kèm form data
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {
            "client_id": CLIENT_ID,
            "filename": os.path.basename(file_path)
        }
        resp = requests.post(f"{FED_SERVER_URL}/upload_model", files=files, data=data)

    logger.success(f"FC2 sent to federated server: {resp.json()}")

    return {"status": f"client {CLIENT_ID} train done"}

@app.post("/eval")
def eval_client(global_round: int = Query(...)):
    SPLIT_SERVER_URL = get_split_server_url()

    fc2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_x), BATCH_SIZE):
            batch_x = test_x[i:i+BATCH_SIZE].to(device)
            batch_y = test_y[i:i+BATCH_SIZE].to(device)

            with torch.no_grad():
                batch_z = encoder(batch_x)

            payload = {"data": batch_z.cpu().numpy().tolist()}
            resp = requests.post(f"{SPLIT_SERVER_URL}/forward", json=payload).json()
            fc1_out = torch.tensor(resp["fc1_out"], dtype=torch.float32, device=device)
            output = fc2(fc1_out)
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    requests.post(f"{BACKEND_URL}/accuracy", json={"client_id": CLIENT_ID, "accuracy": float(accuracy), "global_round": global_round})
    logger.info(f"Evaluation done. Accuracy: {accuracy:.4f}")
    requests.post(f"{SPLIT_SERVER_URL}/clear_pending")
    return {f"global round {global_round}accuracy": accuracy}

@app.post("/load_fc2")
def load_fc2_file(file: UploadFile = File(...)):
    file_path = f"{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    # Load state_dict từ file
    state_dict = torch.load(file_path)
    fc2.load_state_dict(state_dict)
    logger.info(f"FC2 loaded from {file.filename}")
    return {"status": "fc2 loaded"}
