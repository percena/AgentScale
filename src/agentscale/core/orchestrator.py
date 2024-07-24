import asyncio
import datetime
import inspect
import json
import logging
from inspect import iscoroutinefunction
from typing import Dict
from typing import List
from typing import Tuple

import hnswlib
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agentscale.agents.base import BaseAgent
from agentscale.agents.coding import CodingAgent
from agentscale.agents.rag import RAGAgent
from agentscale.core.config import ConfigManager
from agentscale.db.chat_history import get_chat_history
from agentscale.db.chat_history import initialize_database
from agentscale.db.chat_history import save_chat_history


logger = logging.getLogger(__name__)
load_dotenv()


class Orchestrator:
    def __init__(self):
        self.config = ConfigManager()
        self.client = AsyncOpenAI(api_key=self.config.get("OPENAI_API_KEY"))
        self.agents: Dict[str, BaseAgent] = {
            "RAGAgent": RAGAgent(),
            "CodingAgent": CodingAgent(),
        }
        self.index = None
        self.window_size = 5
        self.ef_construction = 300
        self.M = 24
        # Initialize the database
        asyncio.create_task(initialize_database())
        # Initialize the index asynchronously
        asyncio.create_task(self._initialize_hnsw_index())
        logger.info("Orchestrator initialized")

    def _get_function_descriptions(self) -> List[Tuple[str, str, str]]:
        return [
            (f"{agent.__class__.__name__}.{name}", name, inspect.getdoc(method))
            for agent in self.agents.values()
            for name, method in inspect.getmembers(agent, predicate=inspect.ismethod)
            if not name.startswith("_")
        ]

    async def _initialize_hnsw_index(self):
        function_descriptions = self._get_function_descriptions()
        if not function_descriptions:
            logger.error("No function descriptions available to create index.")
            raise ValueError("No function descriptions available to create index.")
        logger.info(f"Retrieved {len(function_descriptions)} function descriptions")
        embeddings = await asyncio.gather(
            *[self._get_openai_embedding(desc) for _, _, desc in function_descriptions]
        )
        if not embeddings:
            logger.error("No embeddings generated for function descriptions.")
            raise ValueError("No embeddings generated for function descriptions.")
        dim = len(embeddings[0])
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(
            max_elements=len(embeddings), ef_construction=self.ef_construction, M=self.M
        )
        self.index.add_items(embeddings, list(range(len(embeddings))))
        logger.info("HNSW index initialized successfully")

    async def _get_openai_embedding(self, text):
        try:
            response = await self.client.embeddings.create(
                input=text, model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {str(e)}")
            raise

    async def _find_most_similar_function(self, query):
        if self.index is None:
            logger.info(
                "HNSW index not initialized. Call _initialize_hnsw_index() first."
            )
            await self._initialize_hnsw_index()
        query_embedding = await self._get_openai_embedding(query)
        try:
            closest_indices, _ = self.index.knn_query(np.array([query_embedding]), k=1)
            return self._get_function_descriptions()[closest_indices[0][0]]
        except RuntimeError as e:
            logger.error(f"Error in HNSW query: {str(e)}")
            if "Cannot return the results in a contiguous 2D array" in str(e):
                logger.warning("No results found in HNSW query. Returning None.")
                return None
            raise

    async def _get_gpt_response(self, prompt, model="gpt-4o", max_tokens=None):
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    async def _get_relevant_chat_history(
        self, conversation_id: str, query: str, k: int = 3
    ) -> List[Dict[str, str]]:
        logger.info(f"Getting relevant chat history for conversation {conversation_id}")
        full_history = await get_chat_history(conversation_id, limit=k)
        if not full_history:
            logger.warning(f"No chat history found for conversation {conversation_id}")
            return []

        query_embedding = await self._get_openai_embedding(query)
        history_embeddings = await asyncio.gather(
            *[self._get_openai_embedding(msg["message"]) for msg in full_history]
        )

        temp_index = hnswlib.Index(space="cosine", dim=len(query_embedding))
        temp_index.init_index(
            max_elements=max(len(history_embeddings), k),
            ef_construction=self.ef_construction,
            M=self.M,
        )
        temp_index.add_items(history_embeddings, list(range(len(history_embeddings))))

        try:
            closest_indices, _ = temp_index.knn_query(
                np.array([query_embedding]), k=min(k, len(history_embeddings))
            )
            return [
                self._format_chat_message(full_history[i]) for i in closest_indices[0]
            ]
        except RuntimeError as e:
            logger.error(f"Error in kNN query: {str(e)}")
            logger.warning("Falling back to returning most recent chat history")
            # Fallback: return the most recent k messages
            return full_history[-k:]

    async def _get_recent_chat_history(
        self, conversation_id: str
    ) -> List[Dict[str, str]]:
        """
        Fetch the most recent chat history from the database, including timestamps.
        """
        history = await get_chat_history(conversation_id, limit=self.window_size)
        return [self._format_chat_message(msg) for msg in history]

    @staticmethod
    def _format_chat_message(msg: Dict[str, str]) -> Dict[str, str]:
        return {
            "role": msg["role"],
            "message": msg["message"],
            "timestamp": msg["timestamp"].isoformat(),
        }

    @staticmethod
    def _format_chat_history(history: List[Dict[str, str]]) -> str:
        """
        Format chat history for logging and LLM input.
        """
        return "\n".join(
            f"[{msg['timestamp']}] {msg['role']}: {msg['message']}" for msg in history
        )

    async def _invoke_function_calling(
        self, func, func_name, func_description, func_signature, step
    ):
        function_call_response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": step}],
            functions=[
                {
                    "name": func_name,
                    "description": func_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param: {
                                "type": "string",
                                "description": f"Parameter {param}",
                            }
                            for param in func_signature.parameters
                        },
                        "required": list(func_signature.parameters.keys()),
                    },
                }
            ],
            function_call={"name": func_name},
        )
        function_args = json.loads(
            function_call_response.choices[0].message.function_call.arguments
        )

        if iscoroutinefunction(func):
            return await func(**function_args)
        else:
            return await asyncio.to_thread(func, **function_args)

    async def _generate_final_response(
        self,
        query,
        plan,
        execution_results,
        recent_chat_history,
        retrieved_history,
        constraints,
        personalization,
    ) -> str:
        simplified_results = [
            f"Step: {result['step']}\n"
            f"Function: {result['function']}\n"
            f"Result: {result['result'] if result['result'] else result['error']}"
            for result in execution_results
        ]
        chat_summary = await self._summarize_chat_history(
            recent_chat_history, retrieved_history
        )

        prompt = f"""
        Generate a concise response (50-150 words) to the current query. 
        You can refer to the plan, its execution results and the summarized chat history, but keep in mind that the final answer should only respond to the query.

        Current Query: {query}
        Plan: {json.dumps(plan, indent=2)}
        Execution Results:
        {'-' * 40}
        {"".join(simplified_results)}
        {'-' * 40}
        Chat Context Summary: {chat_summary}
        Constraints: {constraints}
        Personalization: {personalization}

        Concise Response (50-150 words):
        """

        return await self._get_gpt_response(prompt, max_tokens=150)

    async def _summarize_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Summarize the given chat history."""
        # Remove duplicates while preserving order
        unique_history = self._remove_duplicates(chat_history)
        summary_prompt = f"""
        Summarize the following chat history in 2-3 sentences, focusing on the main topics discussed:

        {self._format_chat_history(unique_history)}

        Summary:
        """
        return await self._get_gpt_response(
            summary_prompt, model="gpt-3.5-turbo", max_tokens=100
        )

    @staticmethod
    def _parse_timestamp(timestamp):
        if isinstance(timestamp, datetime.datetime):
            return timestamp
        elif isinstance(timestamp, str):
            try:
                return datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                return datetime.datetime.min
        else:
            return datetime.datetime.min

    @staticmethod
    def _remove_duplicates(history):
        seen = set()
        return [
            item
            for item in history
            if item["message"] not in seen and not seen.add(item["message"])
        ]

    async def _extend_query(self, query: str, chat_summary: str) -> str:
        prompt = f"""
        Rewrite the following query to make it more specific and unambiguous. Use the chat summary for context if relevant.

        Original Query: {query}
        Chat Summary: {chat_summary}

        Extended Query:
        """
        return await self._get_gpt_response(prompt, max_tokens=100)

    async def _analyze_query(
        self,
        query: str,
        extended_query: str,
        func_descriptions: List[Tuple[str, str, str]],
        chat_summary: str,
    ) -> Dict[str, any]:
        prompt = f"""
        Analyze the following query and determine if it should be handled by the AI language model or routed to an external function.
        Consider the extended query, available functions, and chat history summary.

        Original Query: {query}
        Extended Query: {extended_query}
        Available Functions:
        {json.dumps([(full_name, desc) for full_name, _, desc in func_descriptions], indent=2)}
        Chat History Summary: {chat_summary}

        Provide your analysis in the following format:
        ROUTE_TO_EXTERNAL: [true/false]
        REASONING: [Brief explanation for the routing decision]
        SELECTED_FUNCTION: [Full name of the selected function if routing to external, otherwise "None"]
        """
        response = await self._get_gpt_response(prompt, max_tokens=250)
        return self._parse_analysis_response(response)

    def _parse_analysis_response(self, response: str) -> Dict[str, any]:
        lines = response.strip().split("\n")
        result = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "route_to_external":
                    result[key] = value.lower() == "true"
                elif key in ["reasoning", "selected_function"]:
                    result[key] = value

        if "route_to_external" not in result:
            result["route_to_external"] = False
        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"
        if "selected_function" not in result:
            result["selected_function"] = None

        return result

    async def process_query(
        self,
        user_input: str,
        conversation_id: str,
        constraints: str = "",
        personalization: str = "",
    ) -> str:
        try:
            if self.index is None:
                logger.info("HNSW index not initialized. Initializing now.")
                await self._initialize_hnsw_index()

            recent_history = await self._get_recent_chat_history(conversation_id)
            relevant_history = await self._get_relevant_chat_history(
                conversation_id, user_input
            )
            # Sort by timestamp, handling potential string and datetime types
            combined_history = sorted(
                recent_history + relevant_history,
                key=lambda x: self._parse_timestamp(x.get("timestamp")),
            )
            chat_summary = await self._summarize_chat_history(combined_history)

            extended_query = await self._extend_query(user_input, chat_summary)
            logger.info(f"Extended query: {extended_query}")

            func_descriptions = self._get_function_descriptions()
            analysis = await self._analyze_query(
                user_input, extended_query, func_descriptions, chat_summary
            )
            logger.info(f"Query analysis: {json.dumps(analysis, indent=2)}")

            if not analysis["route_to_external"]:
                logger.info("Query will be handled by LLM. Generating response.")
                prompt = f"""
                Answer the following question concisely. Consider the chat history summary, constraints, and personalization.
                Question: {user_input}
                Extended Query: {extended_query}
                Chat History Summary: {chat_summary}
                Constraints: {constraints}
                Personalization: {personalization}
                """
                response = await self._get_gpt_response(prompt, max_tokens=250)
            else:
                selected_function = analysis["selected_function"]
                logger.info(
                    f"Query will be routed to external function: {selected_function}"
                )

                agent_name, func_name = selected_function.split(".")
                agent = self.agents.get(agent_name)
                if not agent:
                    raise ValueError(f"Agent {agent_name} not found.")

                func = getattr(agent, func_name, None)
                if not func:
                    raise ValueError(
                        f"Function {func_name} not found in agent {agent_name}."
                    )

                func_description = next(
                    (
                        desc
                        for full_name, _, desc in func_descriptions
                        if full_name == selected_function
                    ),
                    None,
                )
                if not func_description:
                    raise ValueError(
                        f"Description for function {selected_function} not found."
                    )

                func_signature = inspect.signature(func)
                response = await self._invoke_function_calling(
                    func, func_name, func_description, func_signature, extended_query
                )

            logger.info(f"Final response generated: {response}")

            await save_chat_history(conversation_id, user_input, "user")
            await save_chat_history(conversation_id, response, "assistant")

            return response
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            return f"An error occurred: {str(e)}."
