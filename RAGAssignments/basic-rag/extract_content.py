"""
Web Content Extraction using LangChain WebBaseLoader

Implements:
1. WebBaseLoader to extract webpage content
2. Cleaning + preprocessing of extracted text
3. Text chunking with overlap
4. Saving processed chunks with metadata
5. Handling edge cases (empty content, invalid URL, large documents)

Usage:
    python extract_content.py --url https://example.com
"""

import os
import argparse
import json
from dotenv import load_dotenv

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def extract_content_from_url(url: str):
    """
    Extract and preprocess content from a URL.

    Args:
        url (str): Website URL to extract content from

    Returns:
        list: List of processed text chunks with metadata
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

    except Exception as e:
        print(f"❌ Error loading URL '{url}': {str(e)}")
        return []

    if not docs:
        print("⚠️ No content found at the provided URL.")
        return []

    # Combine all extracted content into one large string
    full_text = "\n".join([doc.page_content for doc in docs]).strip()

    if not full_text:
        print("⚠️ Extracted content is empty after preprocessing.")
        return []

    print(f"🔍 Extracted {len(full_text)} characters from the webpage.")

    # Chunk the text
    chunks = chunk_text(full_text)

    # Attach metadata
    processed_chunks = [
        {"content": chunk, "metadata": {"source": url, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]

    print(f"📦 Total chunks generated: {len(processed_chunks)}")
    return processed_chunks


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Chunk text using LangChain RecursiveCharacterTextSplitter.

    Args:
        text (str): Raw text
        chunk_size (int): Chunk size in characters
        overlap (int): Overlap between chunks

    Returns:
        list: List of text chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    chunks = splitter.split_text(text)
    return chunks


def save_chunks(chunks: list, output_dir: str = "data/extracted_content"):
    """
    Save processed chunks to JSON files.

    Args:
        chunks (list): List of chunks with metadata
        output_dir (str): Directory to save outputs
    """

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "chunks.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(chunks)} chunks to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract content from web pages")
    parser.add_argument("--url", required=True, help="URL to extract content from")
    parser.add_argument("--output", default="data/extracted_content", help="Output directory")

    args = parser.parse_args()

    print(f"🌐 Extracting content from: {args.url}")

    chunks = extract_content_from_url(args.url)

    if chunks:
        save_chunks(chunks, args.output)
    else:
        print("❌ No chunks to save. Extraction failed.")
