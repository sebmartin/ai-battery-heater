# AI Battery Heater Controller

An intelligent controller that uses machine learning and agentic AI to optimize battery heating for solar power systems.

## What It Does

- **Predicts** future battery temperature using ML models
- **Decides** when to heat the battery using an AI agent
- **Optimizes** for maximum solar energy capture vs. heating cost
- **Adapts** over time through continuous learning

## System Overview

```
Sensors → Data Collection → ML Prediction → Agent Decision → Heater Control
                ↓                                    ↓
            SQLite DB  ←─────── Model Retraining ───┘
```

The system runs on a central server and communicates with a Raspberry Pi running VenusOS that controls the physical heater and integrates with Victron hardware.

## Project Structure

```
src/
  server/     # Server-side code (runs on central server)
    agent/      # Decision-making logic (controllers, tools, graphs)
    ml/         # Machine learning (training, prediction, monitoring)
    collector/  # Data collection from sensors and weather APIs
    api/        # REST API for Pi communication
    config/     # Configuration management

  client/     # Raspberry Pi client code (runs on VenusOS)

data/         # SQLite database and logs
models/       # Trained ML model artifacts
course/       # Learning materials (see course/README.md)
```

## Getting Started

See [course/syllabus.md](course/syllabus.md) for the full learning roadmap.

## Features

- **Rule-based control** with safety guardrails
- **ML-powered predictions** using historical data
- **Agentic decision-making** via LangGraph
- **Automatic retraining** to adapt to changing conditions
- **Model monitoring** and drift detection
- **REST API** for edge device communication
- **Offline evaluation** framework for testing strategies

## Technology Stack

- **ML**: scikit-learn, pandas
- **Agent**: LangChain, LangGraph
- **API**: FastAPI, uvicorn
- **Data**: SQLite
- **Hardware**: Raspberry Pi with VenusOS, Victron system integration, DS18B20 temperature sensors

## Status

This is a learning project built to understand ML and agentic AI in a real production context.
