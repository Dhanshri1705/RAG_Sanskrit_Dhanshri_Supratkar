# RAG_Sanskrit_Dhanshri_Supratkar/code/rag_light_format.py

from retriever import build_index, load_text_file, retrieve
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# -----------------------------
# Load Flan-T5-base model (CPU-friendly)
# -----------------------------
print("Loading Flan-T5 model... please wait.")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()  # Set to evaluation mode (no training)
print("Model loaded successfully!")

# -----------------------------
# Split text into story chunks by Sanskrit headings
# -----------------------------
def split_into_chunks(text):
    """
    Split Sanskrit corpus into story-wise chunks.
    Each heading starts a new chunk.
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
# Generate answer using Flan-T5
# -----------------------------
def generate_answer(retrieved_text, query):
    """
    Takes the retrieved Sanskrit chunk + user query
    and generates an English summary using Flan-T5.
    """
    # Build a clear prompt for Flan-T5
    prompt = (
        f"You are given a Sanskrit passage. "
        f"Answer the following query based on the passage in English.\n\n"
        f"Passage: {retrieved_text}\n\n"
        f"Query: {query}\n\n"
        f"Answer:"
    )

    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",   # PyTorch tensors
        truncation=True,        # Cut off if too long
        max_length=512          # Flan-T5 input limit
    )

    # Generate output (no gradient needed for inference)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=200,        # Max length of generated answer
            num_beams=4,               # Beam search for better quality
            early_stopping=True,       # Stop when answer is complete
            no_repeat_ngram_size=2     # Avoid repetitive phrases
        )

    # Decode the generated token IDs back to text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -----------------------------
# Full RAG pipeline: Retrieve + Generate
# -----------------------------
def answer_query(query, index, chunks):
    """
    1. Retrieve most relevant Sanskrit chunk
    2. Generate English answer using Flan-T5
    """
    retrieved_chunks = retrieve(query, index, chunks)

    if not retrieved_chunks:
        return query, "(No relevant context found)", "(Cannot generate answer)"

    # Top-1 most relevant chunk
    retrieved_text = retrieved_chunks[0].strip()

    # Generate answer from retrieved chunk
    print("\nGenerating answer... (this may take a moment on CPU)")
    generated_answer = generate_answer(retrieved_text, query)

    return query, retrieved_text, generated_answer

# -----------------------------
# Main workflow
# -----------------------------
if __name__ == "__main__":

    # Step 1 - Load Sanskrit corpus
    print("\nLoading Sanskrit dataset...")
    text = load_text_file(
        r"C:\Users\Administrator\Downloads\RAG_Sanskrit_Dhanshri_Supratkar\RAG_Sanskrit_Dhanshri_Supratkar\data\Rag-docs.txt"
    )

    # Step 2 - Split into chunks
    chunks = split_into_chunks(text)
    print(f"Total chunks created: {len(chunks)}")

    # Step 3 - Build FAISS index
    print("Building FAISS index...")
    index, chunks = build_index(chunks)
    print("Index built successfully!")

    # Step 4 - Accept user query
    print("\n" + "="*50)
    query = input("Enter your Sanskrit query: ")

    # Step 5 - Retrieve + Generate
    q, retrieved_text, answer = answer_query(query, index, chunks)

    # Step 6 - Print results
    print("\n" + "="*50)
    print("QUERY:")
    print(q)

    print("\n" + "="*50)
    print("RETRIEVED SANSKRIT CHUNK:")
    print(retrieved_text)

    print("\n" + "="*50)
    print("GENERATED ANSWER (Flan-T5):")
    print(answer)
    print("="*50)
