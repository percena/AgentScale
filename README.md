# 🔥 AgentScale: A Scalable Microservices-based Agent Orchestration Framework

<div align="center">

[**English**](README.md) | [**简体中文**](README.zh.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Microservices](https://img.shields.io/badge/architecture-microservices-orange.svg)](https://microservices.io/)

</div>

Welcome to AgentScale, the next-generation microservices-based agent orchestration framework. With its intelligent routing, stateful conversation management, and scalable architecture, AgentScale stands out as a paradigm shift in how we conceptualize, build, and deploy the AI agents ecosystem.

## 🌟 Key Features

- **Stateful Conversation Management**: Separates stateful conversations from stateless agent services, enabling personalized and context-aware interactions.
- **Intelligent Query Routing**: Utilizes intent analysis and functionality matching to direct queries to the most appropriate agent service.
- **Unified API Gateway**: Provides a single point of entry for seamless interaction with various agent services.
- **Microservices Architecture**: Ensures scalability, resilience, and easy integration of new agent services.
- **Automatic Service Registration**: Simplifies the process of adding and managing new agent services.
- **Pluggable Agent Services**: Supports easy integration of custom agent services, including a demo RAG (Retrieval-Augmented Generation) agent currently.

## 🚀 Getting Started

### Prerequisites

- Docker
- Docker Compose

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/M1n9X/AgentScale.git
   cd AgentScale
   ```

2. Add .env file:
   ```text
   OPENAI_API_KEY=sk-xxx
   ```

3. Build and run the services:

   ```sh
   docker-compose up --build
   ```

4. The API Gateway will be available at `http://localhost:8000`.

## 💻 Usage

Interact with AgentScale through the API Gateway:

```sh
# Health Check
curl -X GET "http://localhost:8000/health"

# Sample Query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text":"What is RAG (Retrieval-Augmented Generation)?"}'

# RAG Indexing (for demo purposes)
curl -X POST "http://localhost:9000/index_pdf" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "data/FakeFile.pdf"}'

# RAG Query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text":"Summarize the main points of the FakeFile document."}'
```

## 🏗️ Architecture

AgentScale leverages a microservices architecture with the following key components:

- **API Gateway**: FastAPI-based entry point for all client requests.
- **Orchestrator**: Core component for intelligent query routing and agent management.
- **Agent Services**: Modular and independently scalable AI agent implementations.
- **Message Queue**: RabbitMQ for efficient inter-service communication.
- **Service Discovery**: Consul for dynamic service registration and discovery.
- **Database**: PostgreSQL for persistent storage of chat histories and other data.

## 📁 Project Structure

```
agentscale/
├── src/
│   └── agentscale/
│       ├── agents/         # Agent interfaces
│       ├── api/            # API Gateway and route definitions
│       ├── core/           # Core orchestration logic
│       ├── db/             # Database models and utilities
│       ├── rag/            # RAG agent implementation
│       ├── services/       # Service discovery and message queue
│       └── utils/          # Helper utilities
├── tests/                  # Test suite
├── data/                   # Sample data for demo
├── docs/                   # Documentation
└── docker-compose.yml      # Docker composition file
```

## 🛠️ Development

For local development:

1. Install dependencies:

   ```sh
   poetry install
   # or
   pip install -e .
   ```

2. Run tests:
   ```sh
   nox
   ```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.