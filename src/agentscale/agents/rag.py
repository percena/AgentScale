import os

import requests
from dotenv import load_dotenv

from .base import BaseAgent


# Load environment variables
load_dotenv()


class RAGAgent(BaseAgent):
    @property
    def capabilities(self):
        return ["information", "retrieval", "answer"]

    def __init__(self):
        self.knowledge_base = {}  # Initialize your knowledge base here
        self.api_url = os.environ.get("RAG_API_URL", "http://rag_api:9000")

    def rag_query(self, query: str) -> str:
        """
        Generate an answer based on the query using retrieved documents from knowledge base.

        Args:
        query (str): The user's question or query.

        Returns:
        str: The generated answer.
        """
        try:
            print("\nStarting RAG API Querying...")

            payload = {"query": query}
            headers = {"Content-Type": "application/json"}

            response = requests.post(
                f"{self.api_url}/query", json=payload, headers=headers
            )

            print(f"Response Status: {response.status_code}")
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            print(f"Error querying RAG API: {e}")
            return []

        # print("\nTesting API endpoints...")
        # base_url = "http://localhost:9000"

        # # Set requests to bypass proxies
        # no_proxy = {
        #     "http": None,
        #     "https": None,
        # }

        # # # Test indexing a PDF
        # # pdf_path = "PerksPlus.pdf"
        # # index_response = requests.post(
        # #     f"{base_url}/index_pdf", params={"file_path": pdf_path}
        # # )
        # # print(f"Index PDF Response: {index_response.status_code}")
        # # print(index_response.json())

        # # Test querying
        # test_queries = [
        #     "What is the main topic of the document?",
        #     # "What are the key points discussed in the document?",
        #     # "Can you summarize the introduction?",
        #     # "What conclusions does the document draw?",
        # ]

        # for query in test_queries:
        #     print(f"\nQuery: {query}")
        #     query_response = requests.post(
        #         f"{base_url}/query", params={"query": query}, proxies=no_proxy
        #     )
        #     print(f"Response Status: {query_response.status_code}")
        #     print(f"Response: {query_response.json()}")

    # def generate_answer(self, query: str, documents: list) -> str:
    #     """
    #     Generate an answer based on the query and retrieved documents.

    #     Args:
    #     query (str): The user's question or query.
    #     documents (list): A list of relevant documents.

    #     Returns:
    #     str: The generated answer.
    #     """
    #     # Implement answer generation logic
    #     return f"Based on the documents, the answer to '{query}' is ..."

    # def summarize(self, text: str, max_length: int = 100) -> str:
    #     """
    #     Summarize the given text.

    #     Args:
    #     text (str): The text to summarize.
    #     max_length (int): The maximum length of the summary.

    #     Returns:
    #     str: The generated summary.
    #     """
    #     # Implement summarization logic
    #     return f"Summary of '{text[:20]}...' (max length: {max_length})"
