# 📈 AI Trading Pro — Reinforcement Learning Based Trading Platform

A full-stack AI trading platform that simulates a real brokerage system and trains a Reinforcement Learning agent to learn optimal trading strategies from historical market data.

The system combines:

• Realistic trading engine  
• Portfolio management backend  
• Live market data integration  
• Q-Learning training pipeline  
• Analytics dashboard  
• Model persistence & inference  

---

## 🧠 Key Idea

Instead of manually designing trading strategies, this project allows an AI agent to **learn how to trade** by interacting with a market environment.

The agent learns:

> When to BUY  
> When to HOLD  
> When to SELL  

based purely on reward optimization.

---

## 🏗️ System Architecture

```
Browser UI
    ↓
Flask Backend API
    ↓
PostgreSQL Database
    ↓
RL Training Engine (Q-Learning)
    ↓
Saved Model (Q-Table)
```

---

## ⚙️ Core Features

### Trading Platform
- User portfolio management
- Buy / Sell order execution
- Watchlist tracking
- Cash balance updates
- Trade history storage

### Live Market Integration
- Real-time stock prices via Finnhub API
- Automatic background price updates (scheduler)
- Dynamic portfolio valuation

### Reinforcement Learning Engine
- Custom trading environment
- Q-Learning agent
- CSV historical data training
- Reward-based learning
- Model persistence (.pkl)
- State-action storage in DB

### Analytics & Visualization
- Portfolio performance metrics
- Training reward graphs
- Portfolio value history
- RL training progress tracking

### Model Inference
- Load trained Q-table
- Predict optimal action for market state
- Strategy simulation

---

## 🤖 Reinforcement Learning Details

| Component | Implementation |
|--------|------|
Environment | Custom TradingEnv |
Agent | Tabular Q-Learning |
Actions | Buy / Hold / Sell |
Reward | Portfolio Profit |
State | Market features derived from price history |
Persistence | Database + Pickle |

---

## 📂 Project Structure

```
trading/
│
├── trading_backend.py        # Main Flask server
├── utils/
│   ├── environment.py        # Trading environment
│   └── agent.py              # Q-learning agent
│
├── templates/                # Frontend UI
│   ├── index.html
│   ├── trading.html
│   ├── portfolio.html
│   ├── analytics.html
│   ├── news.html
│   └── rl_training.html
│
├── uploads/                  # Uploaded datasets (ignored)
├── models/                   # Trained models (ignored)
├── results/                  # Training outputs (ignored)
│
└── README.md
```

---

## 🚀 Running Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-pro.git
cd ai-trading-pro
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Database
Make sure PostgreSQL is running and update DB URL inside:

```
trading_backend.py
```

### 5️⃣ Run Server
```bash
python trading_backend.py
```

Open:

```
http://127.0.0.1:8000
```

---

## 🧪 Training the AI Agent

1. Open **RL Training page**
2. Upload historical CSV data
3. Configure parameters
4. Start training

The agent will learn trading policy and store:

```
models/q_table_job_<id>.pkl
results/results_<id>.json
```

---

## 📊 Example Learning Outcome

The agent gradually learns:

Early episodes → random trading  
Later episodes → profit-maximizing strategy

This demonstrates policy improvement via reward feedback.

---

## 🔐 Important Notes

Ignored from GitHub:
```
models/
uploads/
results/
.venv/
```

These are runtime artifacts and generated automatically.

---

## 🎯 Learning Objectives

This project demonstrates:

- Reinforcement Learning in finance
- Full-stack system design
- API architecture
- Background job scheduling
- Model lifecycle management
- Data-driven decision systems

---

## ⚠️ Disclaimer
Educational trading simulator only.  
Not financial advice or real trading software.

---

## 👨‍💻 Author
**Aashutosh Pandey**

AI/ML Engineer • Backend Developer • Systems Builder
