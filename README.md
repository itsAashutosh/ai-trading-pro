# AI Trading Pro — Reinforcement Learning Trading Research Platform

AI Trading Pro is a full-stack algorithmic trading research platform designed to simulate, train, evaluate, and deploy reinforcement-learning based trading strategies in a controlled environment.

Unlike typical trading bots, this system focuses on **research-grade experimentation** — enabling safe testing of AI trading behavior before real capital is ever involved.

---

## 🚀 What This Project Actually Solves

Most ML trading projects only train a model.

They do NOT provide:

* data ingestion pipeline
* training orchestration
* persistent state
* evaluation metrics
* live simulation environment
* strategy lifecycle

This platform provides the complete workflow:

**Data → Training → Evaluation → Simulation → Analytics → Inference**

The goal is to behave like a mini quant research platform, not a script.

---

## 🏗 System Architecture

```
backend/
│
├── routes/        → API layer (request validation only)
├── services/      → business logic & trading engine
├── agents/        → reinforcement learning models
├── tasks/         → background training workers
├── models/        → database schema
├── metrics/       → performance measurement
├── audit/         → trading audit logs
├── guards/        → safety constraints
└── config/        → runtime configuration
```

### Architectural Principles

* App Factory Pattern
* Service Layer Separation
* Background Worker Training
* Stateful RL Agent Persistence
* Scheduler-driven market updates
* Non-blocking API execution

---

## ⚙️ Core Capabilities

### Trading Simulation

* Portfolio holdings tracking
* Trade execution engine
* Market price updates (scheduled)
* Order lifecycle tracking
* Performance analytics

### Reinforcement Learning

* Q-Learning agent
* Persistent Q-table storage
* Training job system
* CSV historical dataset training
* Prediction inference API

### Analytics

* Portfolio performance metrics
* Reward history
* Training progression
* Strategy evaluation

### System Features

* PostgreSQL persistence
* Background job worker
* Structured logging
* Modular backend architecture
* Safe simulation environment (paper trading)

---

## 📦 Installation

```bash
git clone https://github.com/itsAashutosh/ai-trading-pro.git
cd ai-trading-pro

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create `.env`:

```
DATABASE_URL=postgresql://username:password@localhost:5432/ai_trading_db
SECRET_KEY=dev-secret
FINNHUB_API_KEY=your_api_key
```

---

## ▶️ Run Server

```bash
python run.py
```

Server runs at:

```
http://127.0.0.1:5001
```

---

## 🧪 Training Workflow

### 1) Upload Dataset

```
POST /api/rl-training/upload
```

CSV must contain:

```
Date, Open, High, Low, Close, Volume
```

### 2) Start Training

```
POST /api/rl-training/start
```

Parameters:

* episodes
* learning_rate
* discount_factor
* epsilon
* epsilon_decay
* initial_balance

Training runs asynchronously in background worker.

### 3) Monitor Training

```
GET /api/rl-training/jobs/<user_id>
```

### 4) Get Results

```
GET /api/rl-training/job/<job_id>/results
```

Returns:

* final balance
* total reward
* profit %
* equity curve

### 5) Predict Action

```
POST /api/rl-agent/<job_id>/predict
```

Returns:

```
BUY / HOLD / SELL + confidence
```

---

## 📊 Example System Flow

1. Upload historical market data
2. Train RL agent
3. Evaluate performance
4. Simulate trading decisions
5. Analyze portfolio metrics
6. Iterate strategy

---

## ⚠️ Important Disclaimer

This project is a research simulation platform.

It does NOT:

* connect to real brokers
* execute live trades
* provide financial advice

Designed strictly for experimentation and learning.

---

## 🧭 Roadmap

Planned evolution:

* Risk engine (position sizing & max drawdown)
* Matching engine with partial fills
* Backtesting framework
* Strategy management lifecycle
* Multi-agent competitions
* Paper trading websocket simulation

---

## 👨‍💻 Author

**Aashutosh Pandey**
Backend Systems • Reinforcement Learning • Algorithmic Trading
