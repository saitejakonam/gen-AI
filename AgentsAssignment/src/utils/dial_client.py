"""
EPAM DIAL API Client for OpenAI access (Async-Enabled).

This module provides both synchronous and asynchronous interfaces
to interact with OpenAI models through EPAM's DIAL service.

Async methods are implemented using asyncio.to_thread to safely
wrap the AzureOpenAI synchronous SDK.
"""

import os
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


class DIALClient:
    """
    Client for interacting with EPAM DIAL API.

    - Sync methods preserved (for simple usage)
    - Async wrappers provided (for LangGraph async agents)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("DIAL_API_KEY", "<YOUR_API_KEY_HERE>")
        self.model = model
        self.azure_endpoint = "https://ai-proxy.lab.epam.com"
        self.api_version = "2024-02-01"

        try:
            if not self.api_key or self.api_key == "<YOUR_API_KEY_HERE>":
                print("🚨 DIAL API Key not found. Please set DIAL_API_KEY.")
                self.client = None
                return

            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
            )

            print("✅ DIAL Client initialized successfully!")

        except Exception as e:
            print(f"🔥 Error initializing DIAL client: {e}")
            self.client = None

    # ==========================================================
    # 🔹 SYNCHRONOUS METHODS
    # ==========================================================

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> str:
        if not self.client:
            return "❌ DIAL client not properly initialized."

        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=float(os.getenv("DIAL_TEMPERATURE", "0.7")),
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"❌ Error calling DIAL API: {e}"

    def generate_response(self, context: str, customer_query: str) -> str:
        """
        Generate a customer-facing response.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional hotel customer service agent.\n"
                    "Your name is Konam.\n"
                    "Hotel name: Stay Inn.\n"
                    "Be polite, calm, and helpful."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Customer Message:\n{customer_query}\n\n"
                    "Respond professionally:"
                ),
            },
        ]
        return self.get_completion(messages)

    def classify_intent(self, text: str) -> str:
        """
        Classify customer intent deterministically.

        Returns:
            COMPLAINT | COMPLIMENT | OTHER
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify customer intent.\n"
                    "Respond with ONLY one word:\n"
                    "COMPLAINT, COMPLIMENT."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        result = self.get_completion(messages)
        return result.strip().upper()

    # ==========================================================
    # 🔹 ASYNC WRAPPERS (FOR LANGGRAPH)
    # ==========================================================

    async def get_completion_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> str:
        return await asyncio.to_thread(
            self.get_completion,
            messages,
            model,
        )

    async def generate_response_async(
        self,
        context: str,
        customer_query: str,
    ) -> str:
        return await asyncio.to_thread(
            self.generate_response,
            context,
            customer_query,
        )

    async def classify_intent_async(self, text: str) -> str:
        return await asyncio.to_thread(
            self.classify_intent,
            text,
        )


# ==========================================================
# 🧪 ASYNC TEST (OPTIONAL)
# ==========================================================

async def test_dial_connection_async():
    print("🧪 Testing DIAL API connection (async)...")

    client = DIALClient()

    if not client.client:
        print("❌ DIAL client initialization failed")
        return

    response = await client.get_completion_async(
        [{"role": "user", "content": "Explain technical debt in one sentence."}]
    )

    intent = await client.classify_intent_async(
        "The room was dirty and AC was not working."
    )

    print(f"📝 Async Response: {response}")
    print(f"🧠 Classified Intent: {intent}")


if __name__ == "__main__":
    asyncio.run(test_dial_connection_async())
