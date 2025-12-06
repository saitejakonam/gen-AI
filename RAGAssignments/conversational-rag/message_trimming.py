"""
Smart Message Trimming System for Conversational RAG
----------------------------------------------------

Features:
- Trim history based on max number of messages
- Trim history based on max token count
- Summarize older messages to reduce token usage
- Preserve important conversation context
"""

import os
import tiktoken
from typing import List, Dict
from utils.dial_client import DIALClient


class MessageTrimmer:
    def __init__(
        self,
        max_messages: int = 10,
        max_tokens: int = 2000,
        summarize_threshold: int = 8
    ):
        """
        Args:
            max_messages: Maximum number of messages to keep before trimming
            max_tokens: Maximum allowed token count
            summarize_threshold: Number of messages beyond which summarization is triggered
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self.llm = DIALClient()
        self.encoder = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Token Counting Utility
    # ------------------------------------------------------------------
    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in a list of chat messages."""
        total = 0
        for msg in messages:
            total += len(self.encoder.encode(msg["content"]))
        return total

    # ------------------------------------------------------------------
    # Summarization for long conversations
    # ------------------------------------------------------------------
    def summarize_messages(self, messages: List[Dict]) -> Dict:
        """
        Summaries older messages using DIAL API.
        
        Returns:
            A single summarized message dict.
        """
        text_to_summarize = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in messages]
        )

        prompt = f"""
Summarize the following conversation into 4–6 bullet points.
Keep only crucial context needed for follow-up questions.

Conversation:
{text_to_summarize}
"""

        summary = self.llm.get_completion([
            {"role": "system", "content": "You are a conversation summarizer."},
            {"role": "user", "content": prompt}
        ])

        return {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}"
        }

    # ------------------------------------------------------------------
    # Main Trimming Logic
    # ------------------------------------------------------------------
    def trim(self, messages: List[Dict]) -> List[Dict]:
        """
        Returns a trimmed message list based on:
        - max_messages
        - max_tokens
        - summarization of older messages
        """
        # If small enough, nothing to trim
        if len(messages) <= self.max_messages:
            if self.count_tokens(messages) <= self.max_tokens:
                return messages

        # Step 1: Summarize older messages
        if len(messages) > self.summarize_threshold:
            old_messages = messages[:-self.summarize_threshold]
            recent_messages = messages[-self.summarize_threshold:]

            summary_msg = self.summarize_messages(old_messages)
            messages = [summary_msg] + recent_messages

        # Step 2: Ensure max_messages constraint
        if len(messages) > self.max_messages:
            messages = messages[-self.max_messages:]

        # Step 3: Ensure token count constraint
        while self.count_tokens(messages) > self.max_tokens and len(messages) > 2:
            # Always remove older messages first but preserve summary if exists
            messages.pop(1)

        return messages


# ----------------------------------------------------------------------
# CLI Testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Tell me about machine learning."},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Give me examples."},
        {"role": "assistant", "content": "Sure: supervised, unsupervised..."},
        {"role": "user", "content": "Explain supervised learning."},
        {"role": "assistant", "content": "Supervised learning uses labeled data..."},
    ]

    trimmer = MessageTrimmer(max_messages=5, max_tokens=800)
    trimmed = trimmer.trim(sample)

    print("\n=== Trimmed Messages ===")
    for m in trimmed:
        print(m, "\n")
