import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

load_dotenv()

def solve_logic_puzzle(puzzle_text):
    """
    Solves a logic puzzle using Chain-of-Thought reasoning.
    Returns JSON: {"order": ["Person1", "Person2", ..."]}
    """

    prompt = f"""
You are an expert at solving logic puzzles using structured, step-by-step reasoning.

Your task:
- Read the puzzle carefully.
- Use Chain of Thought (CoT) reasoning to deduce the answer.
- Finally, output ONLY valid JSON in the format:
  {{"order": ["front", "to", "back"]}}

### Example of Chain-of-Thought on a DIFFERENT puzzle:

Puzzle:
"Three animals — Cat, Dog, and Rabbit — are sitting on a bench. The Dog is not at either end. The Cat is left of the Rabbit. Determine the seating order left to right."

Reasoning:
1. Dog cannot be leftmost or rightmost → Dog must be in the middle.
2. Remaining positions: leftmost and rightmost → Cat and Rabbit.
3. Cat is left of Rabbit → Cat = leftmost, Rabbit = rightmost.

Final Answer:
{{
  "order": ["Cat", "Dog", "Rabbit"]
}}

### Now solve THIS puzzle using similar reasoning:

Puzzle:
"{puzzle_text}"

Give your final answer in JSON only, without explanation.
"""

    API_KEY = os.getenv("DIAL_API_KEY")
    AZURE_ENDPOINT = "https://ai-proxy.lab.epam.com"
    API_VERSION = "2024-02-01"

    # Recommended model
    DEPLOYMENT_NAME = "gpt-4"

    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a logic expert who outputs ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # Extract response text
    result_text = response.choices[0].message.content

    # Parse JSON
    try:
        return json.loads(result_text)
    except:
        return {"error": "Invalid JSON returned", "raw_output": result_text}


# Test
puzzle = "Four friends, Alex, Ben, Chris, and David, are standing in a line. Chris is not at either end. Ben is directly in front of Alex. David is somewhere behind Chris. Determine the order of the friends from front to back."
print(solve_logic_puzzle(puzzle))
