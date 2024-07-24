import logging

import uvicorn

from agentscale.api.gateway import app
from agentscale.core.config import ConfigManager
from agentscale.services.queue import RabbitMQQueue


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ConfigManager()
rabbitmq_queue = RabbitMQQueue(
    host=config.get("rabbitmq_host"), port=config.get("rabbitmq_port")
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API Gateway")
    try:
        await rabbitmq_queue.connect()
        logger.info("RabbitMQ connection established")
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API Gateway")
    await rabbitmq_queue.close()


if __name__ == "__main__":
    uvicorn.run(app, host=config.get("api_host"), port=config.get("api_port"))
