from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ✅ Embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# -----------------------------
# Load Sanskrit text file
# -----------------------------
def load_text_file(path):
    with open(r"C:\Users\Administrator\Downloads\RAG_Sanskrit_Dhanshri_Supratkar\RAG_Sanskrit_Dhanshri_Supratkar\data\Rag-docs.txt", "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# Split text into chunks
# -----------------------------
def split_into_chunks(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

# -----------------------------
# Build FAISS index
# -----------------------------
def build_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

# -----------------------------
# Retrieve relevant chunks
# -----------------------------
def retrieve(query, index, chunks, k=3):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

# -----------------------------
# Example usage (test only)
# -----------------------------
if __name__ == "__main__":
    text = load_text_file("sanskrit_data.txt")
    chunks = split_into_chunks(text)
    index, chunks = build_index(chunks)

    query = "कालीदासस्य ग्रन्थाः के के?"
    results = retrieve(query, index, chunks)
    print("Retrieved Chunks:\n", results)
