# RAG_Sanskrit_<InternName>/code/rag_light_format.py

from retriever import build_index, load_text_file, retrieve
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# -----------------------------
# Load lightweight model (Flan-T5-base)
# -----------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------------
# Split text into proper story chunks
# -----------------------------
def split_into_chunks(text):
    """
    Split text into chunks by story headings, keeping the heading with its content.
    This ensures each story (भक्तस्य, मूर्खभृत्यस्य, कालीदासस्य, वृद्धायाः, शीतं बहु बाधति)
    is isolated and retrieved correctly.
    """
    pattern = (
        r'(भक्तस्य.*?)(?=मूर्खभृत्यस्य|कालीदासस्य|वृद्धायाः|शीतं बहु बाधति|$)|'
        r'(मूर्खभृत्यस्य.*?)(?=भक्तस्य|कालीदासस्य|वृद्धायाः|शीतं बहु बाधति|$)|'
        r'(कालीदासस्य.*?)(?=भक्तस्य|मूर्खभृत्यस्य|वृद्धायाः|शीतं बहु बाधति|$)|'
        r'(वृद्धायाः.*?)(?=भक्तस्य|मूर्खभृत्यस्य|कालीदासस्य|शीतं बहु बाधति|$)|'
        r'(शीतं बहु बाधति.*?)(?=भक्तस्य|मूर्खभृत्यस्य|कालीदासस्य|वृद्धायाः|$)'
    )
    raw_chunks = re.findall(pattern, text, flags=re.S)
    chunks = [chunk.strip() for group in raw_chunks for chunk in group if chunk.strip()]
    return chunks

# -----------------------------
# Answer query using RAG (chunk only)
# -----------------------------
def answer_query(query, index, chunks):
    retrieved_chunks = retrieve(query, index, chunks)

    if not retrieved_chunks:
        return query, "(No relevant context found)"

    # Take only the top (most relevant) chunk
    retrieved_text = retrieved_chunks[0].strip()

    print("\n--- Retrieved Chunk ---")
    print(retrieved_text)

    return query, retrieved_text

# -----------------------------
# Main workflow (one query at a time)
# -----------------------------
if __name__ == "__main__":
    # Load Sanskrit dataset
    text = load_text_file(
        r"C:\Users\Administrator\Downloads\RAG_Sanskrit_Dhanshri_Supratkar\RAG_Sanskrit_Dhanshri_Supratkar\data\Rag-docs.txt"
    )
    chunks = split_into_chunks(text)
    index, chunks = build_index(chunks)

    # Take one query from user
    query = input("Enter your query: ")
    q, ctx = answer_query(query, index, chunks)

    print("\n--- Query ---")
    print(q)
    print("\n--- Retrieved Chunk ---")
    print(ctx)
