
import tiktoken
from transformers import AutoTokenizer

# Sample Texts (Add more if you want)
texts = {
    "English": "Large language models are transforming artificial intelligence.",
    "Hindi": "भाषाई मॉडल कृत्रिम बुद्धिमत्ता के क्षेत्र में क्रांति ला रहे हैं।",
    "Japanese": "大規模言語モデルは人工知能を変革しています。",
    "Telugu": "పెద్ద భాషా నమూనాలు కృత్రిమ మేధస్సును మార్చుతున్నాయి."
}

# --------------------------
# OpenAI Tokenizers (via tiktoken)
# --------------------------
def count_openai_tokens(model_name, text):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


# --------------------------
# Mistral & LLaMA Tokenizers (HuggingFace)
# --------------------------
def load_hf_tokenizer(model_id):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading {model_id}: {e}")
        return None


tokenizers = {
    "Mistral-Nemo": load_hf_tokenizer("mistralai/Mistral-Nemo-Instruct-2407"),
    "LLaMA (Open Tokenizer)": load_hf_tokenizer("hf-internal-testing/llama-tokenizer")
}

# --------------------------
# Calculation
# --------------------------
results = {}

for lang, text in texts.items():
    results[lang] = {
        "GPT-3.5-Turbo": count_openai_tokens("gpt-3.5-turbo", text),
        "GPT-4": count_openai_tokens("gpt-4", text),
        "GPT-4o": count_openai_tokens("gpt-4o", text),
    }

    # HF tokenizers
    for name, tok in tokenizers.items():
        if tok:
            results[lang][name] = len(tok.encode(text))

# --------------------------
# Print Results
# --------------------------
import pandas as pd

df = pd.DataFrame(results).T
print(df)
