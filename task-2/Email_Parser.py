import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

load_dotenv()

def parse_email_body(email_text):
    """
    Extracts the primary (sender's) signature block:
    Returns JSON:
    {
        "name": "...",
        "email": "...",
        "phone": "..."
    }
    or None if no clear sender info is found.
    """

    prompt = f"""
You are an expert at analyzing messy email bodies and extracting the *sender’s* signature block.

Your job:
1. Find the *most likely* signature belonging to the sender.
   - Usually near the bottom of the last message.
   - Ignore signatures inside reply chains, forwards, quoted emails, or older messages.
   - If multiple names/emails appear, choose the one closest to the most recent message.
2. Extract:
   - Full name
   - Email address
   - Phone number (if available)
3. If signature cannot be reliably found → return: null

### Example (for a DIFFERENT email):

Email Body:
"
Hi team,

Thanks for your help earlier. Please let me know if you need anything else.

Best regards,
Rachel Kim  
Senior Analyst  
rachel.kim@company.com  
+1 (202) 555-8901

-----Original Message-----
From: Steve Carter <steve.c@older.com>
"
Expected Output:
{{
  "name": "Rachel Kim",
  "email": "rachel.kim@company.com",
  "phone": "+1 (202) 555-8901"
}}

### Now extract sender signature from this email:

Email Body:
\"\"\"{email_text}\"\"\"

Return ONLY JSON or null.
"""

    API_KEY = os.getenv("DIAL_API_KEY")
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
            {"role": "system", "content": "Extract the sender's signature. Output ONLY JSON or null."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    # If model returned literal null
    if text.lower() == "null":
        return None

    try:
        return json.loads(text)
    except:
        return {"error": "Invalid JSON", "raw_output": text}


# Test
email_example = """
Hi John,

Thanks for the update. Let's finalize this tomorrow.

Regards,
Alex Morgan
Product Manager
alex.morgan@company.com
+44 7700 900123

-----Forwarded message-----
From: Support Team <help@company.com>
"""

print(parse_email_body(email_example))
