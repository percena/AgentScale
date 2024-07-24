import logging

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel

from agentscale.core.config import ConfigManager
from agentscale.core.orchestrator import Orchestrator
from agentscale.services.queue import RabbitMQQueue


app = FastAPI()
config = ConfigManager()
orchestrator = None  # We'll initialize this in the startup event
rabbitmq_queue = RabbitMQQueue(
    host=config.get("rabbitmq_host"), port=config.get("rabbitmq_port")
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Query(BaseModel):
    text: str


@app.post("/query")
async def process_query(query: Query, request: Request):
    try:
        client_host = request.client.host
        logger.info(f"Received query from {client_host}: {query.text}")
        if orchestrator is None:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        result = await orchestrator.process_query(query.text, "conversation_id")
        logger.info(f"Processed query. Result: {result}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    global orchestrator
    logger.info("Starting up API Gateway")
    try:
        await rabbitmq_queue.connect()
        logger.info("RabbitMQ connection established")

        # Initialize Orchestrator
        orchestrator = Orchestrator()
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API Gateway")
    await rabbitmq_queue.close()
    # Add any other necessary shutdown logic here


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
