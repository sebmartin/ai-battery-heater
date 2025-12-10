# **Battery Heating Controller — Learning & Build Roadmap**

This document outlines the **step-by-step sequence** to learn the required ML + agent concepts while building the real system.

## **Philosophy: One Evolving System**

You will build **one production system** that evolves through each step. Every feature you add stays in the codebase and improves over time. No throwaway prototypes.

- **Start simple**: Rule-based controller, mock sensors
- **Add intelligence**: ML predictions, agent reasoning
- **Make it robust**: Monitoring, retraining, deployment
- **At the end**: Delete the `course/` folder and deploy a working system

Each step introduces one new concept and **enhances** the existing code.

---

## **0. Project Overview**

You will build an intelligent controller that:

* Reads temperature sensors + weather + irradiance
* Predicts future battery temperature (ML model)
* Makes heating decisions (agent)
* Sends commands to a Raspberry Pi heater controller
* Logs data
* Retrains over time to improve

This roadmap teaches **just enough theory** at each step to progress the project.

---

# **1. Foundations**

## **1.1: Environment & Tools**

### **Theory**

* What is an ML workflow (data → model → evaluation → deployment)
* What is an agent (decision-maker that calls tools)

### **Project Task**

Set up the foundation:

* Create Python environment
* Install: `pandas`, `scikit-learn`, `requests`, `langchain`, `langgraph`, `fastapi`, `uvicorn`
* Create basic structure:
  ```
  src/
    agent/      # Agent logic (starts empty)
    ml/         # ML training & inference
    collector/  # Data collection
    api/        # REST API for Pi communication
    config/     # Configuration
  data/         # SQLite DB and logs
  models/       # Trained model artifacts
  course/       # Learning materials (delete when done)
  ```
* Initialize `src/config/settings.py` with basic config (DB path, API keys, etc.)

---

## **1.2: Define the Data Model**

### **Theory**

* What is a feature?
* What makes data useful for prediction?
* Introduction to the agent decision loop (observe → reason → act)

### **Project Task**

Define your data schema:

* timestamp
* battery_temp
* ambient_temp
* irradiance_forecast
* predicted_sunrise
* heater_state
* etc.

Also sketch the agent decision loop:
1. Observe: collect sensor data + forecasts
2. Reason: evaluate conditions, predict outcomes
3. Act: set heater ON/OFF
4. Log: record decision + outcome

**Deliverables:**
* `data/schema.sql` — SQLite schema with tables: `readings`, `decisions`, `forecasts`
* `course/notes/agent_loop.md` — Architecture sketch
* `src/collector/models.py` — Python data classes matching the schema

---

# **2. Data Pipeline**

## **2.1: Sensor + Weather Collection**

### **Theory**

* Feature engineering basics
* Why timestamps matter
* Why weather forecasts matter

### **Project Task**

Build the data collection service:

* Create `src/collector/sensors.py` with mock sensor interface (returns random temps for now)
* Create `src/collector/weather.py` to fetch forecast from Open-Meteo API (free, no key needed)
* Create `src/collector/collect.py` that runs every 1 minute:
  - Reads sensors
  - Fetches weather forecast
  - Inserts into SQLite `readings` and `forecasts` tables
* Run as a systemd service or background process

**Deliverable:** A continuously running collector that builds your dataset

---

## **2.2: Basic Visualization**

### **Theory**

* How to inspect and understand the dataset
* Concepts: distributions, correlations

### **Project Task**

Create an exploration notebook:

* `course/notebooks/01_explore.ipynb`
* Load data from SQLite
* Plot temperature vs. time, temperature vs. irradiance
* Identify correlations
* Clean obvious outliers (update collector if needed)

**Note:** Notebooks stay in `course/` folder. Production code goes in `src/`.

---

# **3. First Machine Learning Model**

## **3.1: Train a Baseline Predictor**

### **Theory**

* Supervised learning
* Train/test split
* Regression basics (RandomForest or GradientBoosting)

### **Project Task**

Build training pipeline:

* Create `src/ml/features.py` — Feature engineering (lags, rolling averages, time features)
* Create `src/ml/train.py` — Training script:
  - Loads data from SQLite
  - Splits train/test
  - Trains RandomForest to predict battery temp 60 min ahead
  - Saves model to `models/predictor_v1.pkl`
* Create `src/ml/predict.py` — Inference function that loads model and returns prediction

**Deliverable:** Reusable training pipeline that outputs versioned models

---

## **3.2: Model Evaluation**

### **Theory**

* MSE/MAE
* Underfitting vs overfitting
* Why baselines matter

### **Project Task**

Add evaluation to training pipeline:

* Update `src/ml/train.py` to compute MAE/MSE on test set
* Generate plots: predicted vs actual, residuals
* Save metrics to `models/predictor_v1_metrics.json`
* Log evaluation results

**Deliverable:** Every trained model comes with evaluation metrics

---

## **3.3: Model Interpretability**

### **Theory**

* Feature importance: which inputs drive predictions?
* Prediction confidence and uncertainty
* Error analysis: when does the model fail?

### **Project Task**

Create interpretability analysis:

* Add to `src/ml/train.py`: save feature importance plots
* Create notebook `course/notebooks/02_interpret_model.ipynb`:
  - Load trained model
  - Analyze feature importance
  - Plot residuals vs. features
  - Identify when model fails
* Document insights in `course/notes/model_insights.md`

**Deliverable:** Understanding of what drives predictions

---

# **4. Decision-Making Logic**

## **4.1: Rule-Based Controller (No ML Yet)**

### **Theory**

* Control systems
* Hard safety rules
* Why simple rules help debugging

### **Project Task**

Build the first controller:

* Create `src/agent/controller.py` with a `Controller` base class
* Implement `RuleBasedController`:
  - If battery < 0°C and irradiance > threshold → heat ON
  - If battery > 5°C → heat OFF
  - Never heat at night
* Create `src/agent/actuator.py` — Mock heater interface (just logs for now)
* Create `src/agent/runner.py` — Main loop that:
  - Fetches latest sensor data
  - Runs controller
  - Actuates heater
  - Logs decision to `decisions` table

**Deliverable:** Working rule-based controller that logs decisions

---

## **4.2: Calculating Energy Cost vs. Expected Gain**

### **Theory**

* Expected value
* Estimating PV energy gain

### **Project Task**

Add energy calculation:

* Create `src/agent/energy.py` with:
  - `estimate_pv_energy(irradiance, hours)` — estimates solar gain
  - `heater_energy_cost(hours)` — heater consumption
  - `net_energy_gain(action, forecast)` — Δ E = E_PV - E_heater
* Update `RuleBasedController` to use energy calculation in decision logic

**Deliverable:** Energy-aware decision making

---

## **4.3: Agent Wrapper Around Rule-Based Controller**

### **Theory**

* Tool-calling pattern: wrapping logic in callable functions
* Separation of decision logic from execution
* Agent as a coordinator

### **Project Task**

Refactor into tool-based architecture:

* Create `src/agent/tools.py` with tool functions:
  - `@tool get_battery_temp()` — returns current battery temp
  - `@tool get_ambient_temp()` — returns ambient temp
  - `@tool get_forecast()` — returns irradiance forecast
  - `@tool set_heater(state: bool)` — actuates heater
  - `@tool log_decision(...)` — logs to DB
* Refactor `RuleBasedController` to use tools instead of direct calls
* This prepares for LangGraph integration later

**Deliverable:** Tool-based architecture that's agent-ready

---

# **5. Integrating ML Into Decisions**

## **5.1: Use the ML Model to Predict Future Battery Temp**

### **Theory**

* ML-assisted control
* Predictive control (MPC-lite)

### **Project Task**

Enhance controller with ML:

* Add `@tool predict_temp(action, horizon)` that:
  - Loads the trained model
  - Predicts battery temp in N hours given action (heat ON/OFF)
  - Returns prediction
* Implement `MLAssistedController(Controller)`:
  - Predicts temp with heater ON vs OFF
  - Calculates energy gain for each scenario
  - Chooses action that maximizes net energy
* Update `runner.py` to use `MLAssistedController`

**Deliverable:** ML-powered decision making

---

## **5.2: Online Learning**

### **Theory**

* Data drift
* Rolling windows
* Retraining schedule

### **Project Task**

Add automated retraining:

* Create `src/ml/retrain.py` script:
  - Fetches last 7 days of data from SQLite
  - Retrains model
  - Saves as `models/predictor_v{N}.pkl` with incremented version
  - Updates symlink `models/predictor_latest.pkl`
* Set up cron job or systemd timer to run every 6 hours
* Update `predict_temp` tool to load `predictor_latest.pkl`

**Deliverable:** Self-improving system that adapts to new data

---

## **5.3: Model Monitoring & Drift Detection**

### **Theory**

* Detecting when a model degrades
* Comparing prediction error over time
* Validation before deployment

### **Project Task**

Add production monitoring:

* Create `src/ml/monitor.py`:
  - Add new table `model_metrics` to track prediction errors over time
  - After each prediction, log actual vs predicted (when actual becomes available)
  - Calculate rolling MAE over 24 hours
* Update `retrain.py`:
  - Before deploying new model, evaluate on last 48 hours
  - Only update symlink if new model is better
  - Log model version changes to `model_deployments` table
* Create dashboard query: `course/notebooks/03_model_performance.ipynb`

**Deliverable:** Production ML monitoring and safe deployment

---

# **6. Agentic AI Layer**

## **6.1: Build Tools**

### **Theory**

* Agents call tools
* Tools are modular wrappers
* Separation of concerns

### **Project Task**

You already have tools from Step 4.3! Now make them LangChain-compatible:

* Update `src/agent/tools.py` to use `@tool` decorator from LangChain
* Ensure all tools have:
  - Proper type hints
  - Docstrings (LLMs use these!)
  - Error handling
* Test each tool independently

**Deliverable:** LangChain-compatible tool suite

---

## **6.2: Build the Agent Graph**

### **Theory**

* Agent = decision function
* Graph = workflow (can be deterministic or LLM-based)
* Guardrails & safety checks
* Two approaches:
  - **Classical agent**: Rule-based workflow, tools return structured data
  - **LLM-based agent**: Uses Claude/GPT for reasoning, tools provide context

For this project, we'll start with a **hybrid approach**:
* Use a deterministic graph for structure and safety
* Optionally add LLM reasoning for edge cases or explanations

### **Project Task**

Build LangGraph agent:

* Create `src/agent/graph.py`:
  - Define StateGraph with nodes for each decision step
  - Use conditional edges for branching logic
  - Implement safety guardrails as graph constraints
* Implement `AgentController(Controller)`:
  - Uses LangGraph to orchestrate decision making
  - Calls tools via graph execution
  - Logs full decision trace
* **Start with deterministic graph** (no LLM yet)
* Optional: Add Claude node for decision explanation

**Deliverable:** Graph-based agent replacing hand-coded controller logic

---

## **6.3: Expose Agent to Raspberry Pi**

### **Theory**

* Edge → Cloud communication
* Low-power client patterns

### **Project Task**

Build REST API and Pi client:

* Create `src/api/server.py` (FastAPI):
  - `POST /decision` — accepts sensor readings, returns heater command
  - `GET /status` — returns system health
  - Runs agent graph on each request
* Create `pi/client.py`:
  - Reads real sensors (DS18B20 temp sensors via 1-wire)
  - POSTs to API every 5 minutes
  - Actuates heater via GPIO
  - Falls back to safe mode if API unreachable
* Update `actuator.py` to support both mock and GPIO modes

**Deliverable:** Distributed system (Pi ↔ Cloud/LAN server)

---

# **7. Evaluation & Iteration**

## **7.1: Offline Replay Testing**

### **Theory**

* Counterfactual testing
* Simulated reward

### **Project Task**

Create evaluation framework:

* Build `src/agent/simulator.py`:
  - Replays historical data
  - Runs different controller versions in parallel
  - Computes metrics: energy net, heating cycles, temp violations
* Create `course/notebooks/04_controller_comparison.ipynb`:
  - Compare RuleBased vs MLAssisted vs Agent controllers
  - Visualize decisions over time
  - Statistical significance testing
* Document winning strategy in `course/notes/evaluation_results.md`

**Deliverable:** Data-driven understanding of controller performance

---

## **7.2: Real Deployment**

### **Project Task**

Production deployment:

* Set up systemd services:
  - `battery-collector.service` — runs data collection
  - `battery-api.service` — runs FastAPI server
  - `battery-retrain.timer` — triggers retraining every 6 hours
* Deploy to always-on machine (home server or cloud VM)
* Deploy Pi client to Raspberry Pi
* Run for 2 weeks, monitoring:
  - Decision logs
  - Model drift metrics
  - API latency
  - Pi connection stability
* Document issues and fixes in `course/notes/deployment_log.md`

**Deliverable:** Real-world system running autonomously

---

## **7.3: Beyond Supervised Learning — Introduction to RL**

### **Theory**

* Limitations of supervised learning (learns from past, not from outcomes)
* Reinforcement learning basics: state, action, reward
* Contextual bandits: simple RL for decision-making
* How RL can improve heating decisions over time

### **Project Task**

Add RL capability:

* Create `src/agent/rl_controller.py`:
  - Implement `BanditController` using contextual bandit (epsilon-greedy)
  - State: (battery_temp, ambient, irradiance, hour)
  - Actions: [heat_on, heat_off]
  - Reward: calculated retrospectively from energy gain
* Add reward calculation to decision logger
* Train bandit on 30 days of logged decisions
* Compare bandit vs supervised learning in simulator
* Document RL insights in `course/notes/rl_experiments.md`

**Deliverable:** Introduction to RL and outcome-based learning

---

# **8. Optional Extensions (Choose Any)**

### **8.1: Advanced Reinforcement Learning**

Train a full Q-learning or DQN model for multi-step planning.

### **8.2: Multi-step forecasting**

Predict full temperature trajectory over next 6-12 hours.

### **8.3: Better physics modeling**

Hybrid physical + ML predictor using heat transfer equations.

### **8.4: Edge ML**

Compress the model to run locally on the Pi (quantization, distillation).

### **8.5: LLM-Based Agent**

Replace deterministic graph with full LLM reasoning for decisions.

---

# **Summary Roadmap**

1. **Foundations** — Environment, data model, architecture
2. **Data Pipeline** — Collector service + exploration
3. **ML Basics** — Train, evaluate, interpret predictor
4. **Control Logic** — Rule-based controller + energy model + tools
5. **ML Integration** — ML-assisted controller + auto-retraining + monitoring
6. **Agent Layer** — LangChain tools + LangGraph + REST API + Pi client
7. **Production** — Evaluation, deployment, RL experiments

---

## **The Evolution: From Simple to Intelligent**

| Step | System State | What's Running |
|------|--------------|----------------|
| 2 | Data collection only | Mock sensors → SQLite |
| 4 | Rule-based control | RuleBasedController decides when to heat |
| 5 | ML-powered control | MLAssistedController uses predictions |
| 6 | Agent-based control | AgentController orchestrates via LangGraph |
| 7 | Production system | Pi client ↔ API server ↔ auto-retraining |

**At every step, you have a working system.** Each phase enhances the previous one.

---

## **Estimated Timeline**

At a comfortable self-paced learning speed (5-10 hours/week):

* **Phase 1 (Foundations)**: 2-3 weeks — Steps 1-2
* **Phase 2 (ML Basics)**: 2-3 weeks — Steps 3-3.3
* **Phase 3 (Control Logic)**: 2-3 weeks — Steps 4-4.3
* **Phase 4 (ML Integration)**: 2-3 weeks — Steps 5-5.3
* **Phase 5 (Agent Layer)**: 3-4 weeks — Steps 6-6.3
* **Phase 6 (Production)**: 3-4 weeks — Steps 7-7.3

**Total**: ~17-23 weeks (4-6 months) for full system + RL

---

## **Final State: Production-Ready System**

When you're done, your repo will contain:

```
src/
  agent/          # Controllers (rule-based, ML, agent, RL)
  ml/             # Training, prediction, monitoring, retraining
  collector/      # Data collection service
  api/            # FastAPI server
  config/         # Settings
pi/               # Raspberry Pi client
data/             # SQLite database
models/           # Trained model artifacts
scripts/          # Deployment scripts (systemd, etc.)
README.md         # Production documentation
```

**You can delete `course/` and have a deployable system.**

---

## **Next Steps**

Ready to start? Begin with **Step 1.1: Environment & Tools**.
