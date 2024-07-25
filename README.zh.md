# 🔥 AgentScale: 基于微服务架构的可扩展智能体编排框架

<div align="center">

[**English**](README.md) | [**简体中文**](README.zh.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Microservices](https://img.shields.io/badge/architecture-microservices-orange.svg)](https://microservices.io/)

</div>

欢迎使用 AgentScale，这是一个基于微服务的下一代智能体编排框架。凭借其智能路由、有状态会话管理和可扩展的架构，AgentScale 在如何概念化、构建和部署 AI 智能体生态系统方面，开创了一种全新的范式。

## 🌟 主要特性

- **有状态会话管理**: 将有状态会话与无状态代理服务分离,实现个性化和上下文感知的交互。
- **智能查询路由**: 利用意图分析和功能匹配将查询引导至最合适的代理服务。
- **统一 API 网关**: 提供单一入口点,实现与各种代理服务的无缝交互。
- **微服务架构**: 确保可扩展性、弹性和新代理服务的易集成性。
- **自动服务注册**: 简化新代理服务的添加和管理过程。
- **可插拔代理服务**: 支持轻松集成自定义代理服务,目前包括一个演示用的 RAG(检索增强生成)代理。

## 🚀 快速开始

### 前置条件

- Docker
- Docker Compose

### 安装

1. 克隆仓库:

   ```sh
   git clone https://github.com/M1n9X/AgentScale.git
   cd AgentScale
   ```

2. 添加 .env 文件:
   ```text
   OPENAI_API_KEY=sk-xxx
   ```

3. 构建并运行服务:

   ```sh
   docker-compose up --build
   ```

4. API 网关位于 `http://localhost:8000`。

## 💻 使用方法

通过API网关与AgentScale交互:

```sh
# 健康检查
curl -X GET "http://localhost:8000/health"

# 示例查询
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text":"什么是RAG (检索增强生成)?"}'

# RAG索引 (用于演示)
curl -X POST "http://localhost:9000/index_pdf" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "data/FakeFile.pdf"}'

# RAG查询
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text":"总结FakeFile文档的主要观点。"}'
```
## 🏗️ 架构

AgentScale 利用微服务架构，具有以下关键组件：

- **API 网关**: 基于 FastAPI 的所有客户端请求的入口点。
- **协调器**: 智能查询路由和智能体管理的核心组件。
- **智能体服务**: 模块化且独立可扩展的 AI 智能体实现。
- **消息队列**: 使用 RabbitMQ 进行高效的服务间通信。
- **服务发现**: 使用 Consul 实现动态的服务注册和发现。
- **数据库**: 使用 PostgreSQL 存储聊天记录和其他数据。

## 📁 项目结构

```
agentscale/
├── src/
│   └── agentscale/
│       ├── agents/         # 智能体接口
│       ├── api/            # API 网关和路由定义
│       ├── core/           # 核心编排逻辑
│       ├── db/             # 数据库模型和工具
│       ├── rag/            # RAG 智能体实现
│       ├── services/       # 服务发现和消息队列
│       └── utils/          # 辅助工具
├── tests/                  # 测试套件
├── data/                   # 用于演示的样本数据
├── docs/                   # 文档
└── docker-compose.yml      # Docker 编排文件
```

## 🛠️ 开发

对于本地开发:

1. 安装依赖:

   ```sh
   poetry install
   # 或
   pip install -e .
   ```

2. 运行测试:
   ```sh
   nox
   ```

## 🤝 贡献

我们欢迎贡献!请查看我们的[贡献指南](CONTRIBUTING.md)了解更多详情。

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 💬 社区交流
欢迎👏微信扫码, 加入我们的社区讨论!
<p align="center">
  <img src="./docs/assets/wechat.jpeg" width="300px" />
</p>