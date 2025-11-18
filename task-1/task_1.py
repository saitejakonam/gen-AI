# import tiktoken

# # For GPT-3.5-Turbo or GPT-4:
# enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
# num_tokens_1 = len(enc.encode("Your text here"))
# print(num_tokens_1)
# # For GPT-4o:
# enc = tiktoken.encoding_for_model('gpt-4o')
# num_tokens_2 = len(enc.encode("Your text here"))
# print(num_tokens_2)


# # ...existing code...
# import tiktoken
# from typing import Dict, List

# # Text samples (different languages / types)
# TEXTS: Dict[str, str] = {
#     "english_short": "Your text here",
#     "english_long": "This is a longer English paragraph intended to show how tokenization changes "
#                     "with more content. It includes punctuation, numbers like 1234, and some code: x = 10",
#     "spanish": "Hola, ¿cómo estás? Esto es una prueba de tokenización en español.",
#     "chinese": "这是一个测试，用于检查中文的分词情况。",
#     "code_snippet": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
#     "emoji": "Hello 👋🌍 — testing emojis and special characters 👍"
# }

# OPENAI_MODELS: List[str] = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]

# def tokens_with_tiktoken(model: str, text: str) -> int:
#     """
#     Try to get an encoding for the given model using tiktoken.
#     Fall back to cl100k_base if model-specific encoding is unavailable.
#     """
#     try:
#         enc = tiktoken.encoding_for_model(model)
#     except Exception:
#         enc = tiktoken.get_encoding("cl100k_base")
#     return len(enc.encode(text))

# def compare_openai_models(texts: Dict[str, str], models: List[str]) -> None:
#     print("OpenAI model token counts:")
#     for model in models:
#         for name, txt in texts.items():
#             count = tokens_with_tiktoken(model, txt)
#             print(f"{model:12} | {name:14} | {count}")

# # Optional: try Hugging Face tokenizers for Mistral and Llama-3.1 if transformers is installed.
# def compare_hf_tokenizers(texts: Dict[str, str]) -> None:
#     try:
#         from transformers import AutoTokenizer
#     except Exception:
#         print("\nTransformers not available or cannot be imported; skipping Mistral/LLama comparisons.")
#         return

#     hf_models = {
#         "mistral (hf)": "mistralai/mistral-7b",        # common HF name; may require internet / model access
#         "llama-3.1 (hf)": "meta-llama/Llama-3-1b"      # may require authentication / be unavailable
#     }

#     print("\nHugging Face tokenizer token counts (if available):")
#     for label, hf_name in hf_models.items():
#         try:
#             tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
#         except Exception as e:
#             print(f"{label:20} | failed to load tokenizer for '{hf_name}': {e}")
#             continue

#         for name, txt in texts.items():
#             ids = tok(txt, return_tensors=None, add_special_tokens=False)["input_ids"]
#             print(f"{label:20} | {name:14} | {len(ids)}")

# if __name__ == "__main__":
#     compare_openai_models(TEXTS, OPENAI_MODELS)
#     compare_hf_tokenizers(TEXTS)
# ...existing code...

# --------------------------------------------------------
# Token Count Comparison: GPT-3.5, GPT-4, GPT-4o, Mistral, LLaMA
# --------------------------------------------------------

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
