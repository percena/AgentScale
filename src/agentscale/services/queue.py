import asyncio
import json
import logging
from typing import Any
from typing import Callable
from typing import Dict

import aio_pika


logger = logging.getLogger(__name__)


class RabbitMQQueue:
    def __init__(self, host="localhost", port=5672, max_retries=5, retry_delay=5):
        self.rabbitmq_url = f"amqp://{host}:{port}/"
        self.connection = None
        self.channel = None
        self.subscribers: Dict[str, set[Callable]] = {}
        self.message_handler_tasks = {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def connect(self):
        for attempt in range(self.max_retries):
            try:
                if not self.connection:
                    self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
                if not self.channel:
                    self.channel = await self.connection.channel()
                    await self.channel.set_qos(prefetch_count=1)
                logger.info("Successfully connected to RabbitMQ")
                return
            except aio_pika.exceptions.AMQPConnectionError as e:
                logger.warning(
                    f"Failed to connect to RabbitMQ (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(
                        self.retry_delay * (2**attempt)
                    )  # Exponential backoff
                else:
                    raise

    async def send_message(self, queue_name: str, message: Dict[str, Any]):
        await self.connect()
        queue = await self.channel.declare_queue(queue_name, durable=True)
        await queue.publish(aio_pika.Message(body=json.dumps(message).encode()))

    async def receive_message(self, queue_name: str) -> Dict[str, Any]:
        await self.connect()
        queue = await self.channel.declare_queue(queue_name, durable=True)
        incoming_message = await queue.get(timeout=5)
        if incoming_message:
            async with incoming_message.process():
                return json.loads(incoming_message.body)
        return None

    async def publish(self, event: str, data: Dict[str, Any]):
        await self.connect()
        exchange = await self.channel.declare_exchange(
            event, aio_pika.ExchangeType.FANOUT
        )
        await exchange.publish(
            aio_pika.Message(body=json.dumps(data).encode()), routing_key=""
        )

    async def subscribe(self, event: str, callback: Callable):
        await self.connect()

        if event not in self.subscribers:
            self.subscribers[event] = set()

        self.subscribers[event].add(callback)

        exchange = await self.channel.declare_exchange(
            event, aio_pika.ExchangeType.FANOUT
        )
        queue = await self.channel.declare_queue(exclusive=True)
        await queue.bind(exchange)

        if event not in self.message_handler_tasks:
            self.message_handler_tasks[event] = asyncio.create_task(
                self._handle_messages(queue, event)
            )

    async def _handle_messages(self, queue: aio_pika.Queue, event: str):
        async with queue.iterator() as queue_iter:
            async for incoming_message in queue_iter:
                async with incoming_message.process():
                    data = json.loads(incoming_message.body)
                    for callback in self.subscribers.get(event, []):
                        asyncio.create_task(callback(data))

    async def close(self):
        for task in self.message_handler_tasks.values():
            task.cancel()
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()
        logger.info("RabbitMQ connection closed")
