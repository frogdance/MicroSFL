import os
import datetime
import asyncio
import requests
from fastapi import FastAPI, Depends, Query, BackgroundTasks
from pydantic import BaseModel, condecimal
from sqlalchemy import Column, String, Float, Integer, DateTime, select, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from loguru import logger
import time

# -----------------------
# Database config
# -----------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "postgres")

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

time.sleep(5)
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# -----------------------
# DB Models
# -----------------------
class Client(Base):
    __tablename__ = "clients"
    client_id = Column(String, primary_key=True, index=True)
    ip = Column(String)
    role = Column(String, default="client")  # client / split / fed

class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    client_id = Column(String, index=True)
    loss = Column(Float)
    global_round = Column(Integer, default=0)  # thêm cột global_round
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class Accuracy(Base):
    __tablename__ = "accuracies"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    client_id = Column(String, index=True)
    accuracy = Column(Float)
    global_round = Column(Integer, default=0)  # thêm cột global_round
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# -----------------------
# Schemas
# -----------------------
class ClientInfo(BaseModel):
    client_id: str
    ip: str

class MetricIn(BaseModel):
    client_id: str
    loss: float
    global_round: int  # thêm field

class AccuracyIn(BaseModel):
    client_id: str
    accuracy: condecimal(ge=0, le=1)  # accuracy ∈ [0,1]
    global_round: int  # thêm field

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()

async def get_session():
    async with AsyncSessionLocal() as session:
        yield session

# Create tables at startup
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# -----------------------
# Reset DB API
# -----------------------
@app.post("/reset_db")
async def reset_db(background_tasks: BackgroundTasks):
    """
    Drop all tables và tạo lại từ đầu.
    Sử dụng BackgroundTasks để không block request.
    """
    async def reset_task():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    background_tasks.add_task(reset_task)
    return {"status": "reset started"}

@app.post("/clear_metrics_accuracies")
async def clear_metrics_accuracies(db: AsyncSession = Depends(get_session)):
    await db.execute(delete(Metric))
    await db.execute(delete(Accuracy))
    await db.commit()
    return {"status": "metrics and accuracies cleared"}

# -----------------------
# Client & Server APIs
# -----------------------
@app.post("/client_info")
async def post_client_info(info: ClientInfo, db: AsyncSession = Depends(get_session)):
    client = await db.get(Client, info.client_id)
    if not client:
        client = Client(client_id=info.client_id, ip=info.ip, role="client")
        db.add(client)
    else:
        client.ip = info.ip
    await db.commit()
    return {"status": "ok"}

@app.post("/register_split")
async def register_split(info: ClientInfo, db: AsyncSession = Depends(get_session)):
    client = await db.get(Client, info.client_id)
    if not client:
        client = Client(client_id=info.client_id, ip=info.ip, role="split")
        db.add(client)
    else:
        client.ip = info.ip
        client.role = "split"
    await db.commit()
    return {"status": "split registered"}

@app.post("/register_fed")
async def register_fed(info: ClientInfo, db: AsyncSession = Depends(get_session)):
    client = await db.get(Client, info.client_id)
    if not client:
        client = Client(client_id=info.client_id, ip=info.ip, role="fed")
        db.add(client)
    else:
        client.ip = info.ip
        client.role = "fed"
    await db.commit()
    return {"status": "fed registered"}

@app.get("/client_info")
async def get_client_info(role: str | None = Query(None), db: AsyncSession = Depends(get_session)):
    stmt = select(Client.client_id, Client.ip)
    if role:
        stmt = stmt.where(Client.role == role)
    result = await db.execute(stmt)
    return {"clients": [dict(r) for r in result.mappings().all()]}

# -----------------------
# Metric APIs
# -----------------------
@app.post("/metric")
async def post_metric(m: MetricIn, db: AsyncSession = Depends(get_session)):
    metric = Metric(client_id=m.client_id, loss=m.loss, global_round=m.global_round)
    db.add(metric)
    await db.commit()
    return {"status": "ok"}

@app.get("/metric")
async def get_metric(client_id: str = None, db: AsyncSession = Depends(get_session)):
    if client_id:
        stmt = select(Metric.loss, Metric.timestamp).where(Metric.client_id == client_id)
    else:
        stmt = select(Metric.client_id, Metric.loss, Metric.timestamp)
    result = await db.execute(stmt)
    return {"metrics": [dict(r) for r in result.mappings().all()]}

# -----------------------
# Accuracy APIs
# -----------------------
@app.post("/accuracy")
async def post_accuracy(a: AccuracyIn, db: AsyncSession = Depends(get_session)):
    acc = Accuracy(client_id=a.client_id, accuracy=float(a.accuracy), global_round=float(a.global_round))
    db.add(acc)
    await db.commit()
    return {"status": "ok"}

@app.get("/accuracy")
async def get_accuracy(client_id: str = None, db: AsyncSession = Depends(get_session)):
    if client_id:
        stmt = select(Accuracy.accuracy, Accuracy.global_round, Accuracy.timestamp).where(Accuracy.client_id == client_id)
    else:
        stmt = select(Accuracy.client_id, Accuracy.accuracy, Accuracy.global_round, Accuracy.timestamp)
    result = await db.execute(stmt)
    return {"accuracies": [dict(r) for r in result.mappings().all()]}

# -----------------------
# Federated learning
# -----------------------
async def get_fed_split_clients(db: AsyncSession):
    """Lấy IP fed/split và danh sách client từ DB async"""
    result = await db.execute(select(Client.client_id, Client.ip, Client.role))
    clients = result.mappings().all()
    fed_ip = next((c["ip"] for c in clients if c["role"] == "fed"), None)
    split_ip = next((c["ip"] for c in clients if c["role"] == "split"), None)
    client_list = [c for c in clients if c["role"] == "client"]
    return fed_ip, split_ip, client_list

def trigger_client_train(client, global_round):
    try:
        resp = requests.post(f"http://{client['ip']}/train", params={"global_round": global_round}, timeout=60)
        return f"Client {client['client_id']} train done: {resp.status_code}"
    except Exception as e:
        return f"Client {client['client_id']} train failed: {e}"

def trigger_client_eval(client, global_round):
    try:
        resp = requests.post(f"http://{client['ip']}/eval", params={"global_round": global_round}, timeout=60)
        return f"Client {client['client_id']} eval done: {resp.status_code}"
    except Exception as e:
        return f"Client {client['client_id']} eval failed: {e}"

@app.post("/federated_train_async")
async def federated_train_async(global_round: int = Query(..., ge=1), db: AsyncSession = Depends(get_session)):
    fed_ip, split_ip, clients = await get_fed_split_clients(db)

    # Trigger initialize fed server
    if fed_ip:
        try:
            requests.post(f"http://{fed_ip}/initialize", timeout=30)
        except Exception as e:
            logger.error(f"Fed initialize failed: {e}")

    for r in range(global_round):
        print(f"=== Global Round {r+1} ===")

        # 1. Train all clients
        client_train_tasks = [asyncio.to_thread(trigger_client_train, c, r+1) for c in clients]
        train_results = await asyncio.gather(*client_train_tasks)
        for res in train_results:
            print(res)
        print(f"Round {r+1}: All clients finished training")

        # 2. Fed aggregate + Split accumulate (chạy tuần tự, không song song với train)
        requests.get(f"http://{fed_ip}/aggregate", timeout=60)
        requests.post(f"http://{split_ip}/accumulate", timeout=60)

        # 3. Eval all clients
        client_eval_tasks = [asyncio.to_thread(trigger_client_eval, c, r+1) for c in clients]
        eval_results = await asyncio.gather(*client_eval_tasks)
        for res in eval_results:
            print(res)
        print(f"Round {r+1}: All clients finished evaluation")

    return {"status": "federated training done"}

