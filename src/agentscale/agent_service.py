import asyncio
import logging
import sys

from agentscale.agents.coding import CodingAgent
from agentscale.agents.rag import RAGAgent
from agentscale.core.config import ConfigManager
from agentscale.services.discovery import ConsulClient
from agentscale.services.queue import RabbitMQQueue


# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_agent(agent_type):
    try:
        config = ConfigManager()
        agent_classes = {
            "rag": RAGAgent,
            "coding": CodingAgent,
        }

        if agent_type not in agent_classes:
            logger.error(f"Unknown agent type: {agent_type}")
            return

        agent = agent_classes[agent_type]()
        consul_client = ConsulClient(
            host=config.get("consul_host"),
            port=config.get("consul_port"),
        )
        rabbitmq_queue = RabbitMQQueue(
            host=config.get("rabbitmq_host"), port=config.get("rabbitmq_port")
        )

        await rabbitmq_queue.connect()  # Ensure RabbitMQ connection is established

        service_id = f"{agent_type}_agent_{asyncio.get_running_loop().time()}"
        consul_client.register_service(
            agent_type,
            service_id,
            config.get("agent_host", "localhost"),
            config.get("agent_port", 5000),
            agent.capabilities,
        )

        logger.info(f"Running {agent_type} agent...")

        async def process_message(data):
            query = data["query"]
            result = agent.process(query)
            await rabbitmq_queue.publish(f"{agent_type}_response", {"result": result})

        # Subscribe to the queue and start processing messages
        await rabbitmq_queue.subscribe(f"{agent_type}_queue", process_message)

        # Keep the agent running
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logger.exception(f"Exception occurred while running the agent: {e}")
    finally:
        if "rabbitmq_queue" in locals():
            await rabbitmq_queue.close()
        if "consul_client" in locals():
            consul_client.deregister_service(service_id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python agent_service.py <agent_type>")
        sys.exit(1)

    agent_type = sys.argv[1]

    try:
        asyncio.run(run_agent(agent_type))
    except KeyboardInterrupt:
        logger.info("Agent service stopped by user.")
    except Exception as e:
        logger.exception(f"Exception occurred in main loop: {e}")
