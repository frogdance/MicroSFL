# Federated Split Learning with Efficient File-Based Gradient Synchronization

## Introduction

The project improves **Split Federated Learning (SFL)** by introducing a **file-based gradient synchronization** mechanism. The system modularizes split servers and clients as independent **microservices**, enabling flexible scaling and deployment tailored to different use cases.

## Features

- File-based gradient synchronization for efficient model updates.
- Microservice-oriented design: split servers and clients can be deployed and scaled independently.
- Support for balanced and imbalanced (Non-IID) client data setups.

## Prerequisites

- Docker Desktop (Windows)
- NVIDIA driver and NVIDIA Container Toolkit

## Quick Start

> **Note:** Services must be started in order: **Database & Backend → Server services → Client services**.  
> Starting them out of order may cause connection failures.

1. Start the database and backend:
```bash
docker-compose -f docker-compose.db-backend.yml up
```

2. Start the Split Server and Federated Server:
```bash
docker-compose -f docker-compose.servers.yml up
```

3. Start clients:
```bash
docker-compose -f docker-compose.clients.yml up
```
or for imbalanced (Non-IID) data:
```bash
docker-compose -f docker-compose.clients_imbalance.yml up
```

4. Open the FastAPI documentation at `<backend_ip>:8000/docs`.  
   Locate the `POST /federated_train_async` API, provide the desired number of **global rounds**, and trigger the federated training process.
