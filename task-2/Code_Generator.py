import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

load_dotenv()

def generate_python_function(description):
    """
    Generates Python code from a natural language description.
    Returns ONLY Python code as a string.
    """

    prompt = f"""
You are an expert Python code generator.

Your rules:
- Read the user's description carefully.
- Output ONLY valid Python code.
- No explanations, no comments, no markdown, no backticks.

### Example

Description:
"Create a Python function named 'add_numbers' that takes two integers and returns their sum."

Output:
def add_numbers(a, b):
    return a + b

### Now generate code for this description:

"{description}"

Remember: ONLY return Python code.
"""

    API_KEY = os.getenv("DIAL_API_KEY")
    AZURE_ENDPOINT = "https://ai-proxy.lab.epam.com"
    API_VERSION = "2024-02-01"
    DEPLOYMENT_NAME = "gpt-4"   # safer, faster, recommended

    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You output ONLY raw Python code."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# Test
desc = "Create a Python function named 'calculate_average' that takes a list of numbers and returns their average."
print(generate_python_function(desc))
