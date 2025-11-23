import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv
load_dotenv()

def classify_and_prioritize(ticket_text):
    """
    Classifies a support ticket and assigns priority.
    Returns a JSON object: {"category": "...", "priority": "..."}
    """

    prompt = f"""
You are an expert support ticket classifier.

Your task:
1. Classify the ticket into ONE of the following:
   - Bug
   - Feature Request
   - Question
   - Praise
   - Complaint

2. Assign a priority level:
   - High: Critical issues, system down, security issues, blocking bugs
   - Medium: Major inconvenience but not fully blocking
   - Low: Minor bugs, questions, praise, small improvements

Return ONLY valid JSON with keys: category, priority.

### Examples

Ticket: "The login button does nothing when clicked."
Response:
{{
  "category": "Bug",
  "priority": "High"
}}

Ticket: "Can you add dark mode? It would be amazing!"
Response:
{{
  "category": "Feature Request",
  "priority": "Low"
}}

Ticket: "Why am I seeing two entries on my dashboard?"
Response:
{{
  "category": "Question",
  "priority": "Low"
}}

Ticket: "Great work team! Loving the new UI."
Response:
{{
  "category": "Praise",
  "priority": "Low"
}}

Ticket: "The app keeps crashing and I'm losing my progress!"
Response:
{{
  "category": "Complaint",
  "priority": "High"
}}

### Now classify the following ticket:

Ticket: "{ticket_text}"
"""
    API_KEY = os.getenv("DIAL_API_KEY","ed6b44cff7d9448696d2d4df02bed37f") 
    AZURE_ENDPOINT = "https://ai-proxy.lab.epam.com"
    API_VERSION = "2024-02-01"
    DEPLOYMENT_NAME = "gpt-4"

    client = AzureOpenAI(
      api_key=API_KEY,
      api_version=API_VERSION,
      azure_endpoint=AZURE_ENDPOINT
    )
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a support ticket classifier returning strict JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # Extract the JSON from model response
    result_text = response.choices[0].message.content

    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned", "raw_output": result_text}


ticket = "The payment page keeps freezing when I try to check out."
print(classify_and_prioritize(ticket))
