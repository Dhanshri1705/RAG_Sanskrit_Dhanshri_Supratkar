# RAG_Sanskrit_<InternName>/code/rag_light.py

from retriever import build_index, split_into_chunks, load_text_file, retrieve
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# Load lightweight model (Flan-T5-base)
# -----------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------------
# Answer query using RAG
# -----------------------------
def answer_query(query, index, chunks):
    # Retrieve context from dataset
    retrieved = "\n".join(retrieve(query, index, chunks))
    prompt = f"Answer the question based on the text:\n{retrieved}\n\nQuestion: {query}"

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Main workflow
# -----------------------------
if __name__ == "__main__":
    # Load Sanskrit dataset
    text = load_text_file("./data/Rag-docs.txt")
    chunks = split_into_chunks(text)
    index, chunks = build_index(chunks)

    print("Type your query (or 'exit' to quit):")
    while True:
        query = input("> ")
        if query.lower() == "exit":
            break
        response = answer_query(query, index, chunks)
        print(f"\nAnswer: {response}\n")
