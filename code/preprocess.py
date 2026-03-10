import re

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text: str):
    """
    Preprocess Sanskrit text:
    - Split by story headings
    - Ensure one chunk per heading
    - Optionally split long chunks (~300 chars) by danda/newline
    """
    headings = ["भक्तस्य", "मूर्खभृत्यस्य", "कालीदासस्य", "वृद्धायाः", "शीतं बहु बाधति"]
    pattern = "(" + "|".join(headings) + r").*?(?=" + "|".join(headings) + "|$)"
    raw_chunks = re.findall(pattern, text, flags=re.S)

    # Deduplicate by heading
    seen = set()
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        heading = next((h for h in headings if h in chunk), None)
        if heading and heading in seen:
            continue
        seen.add(heading)

        # If chunk is long, split internally
        if len(chunk) > 400:
            sentences = re.split(r'।|\n', chunk)
            current = []
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                current.append(s)
                if len(" ".join(current)) > 300:
                    chunks.append(" ".join(current))
                    current = []
            if current:
                chunks.append(" ".join(current))
        else:
            chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    filepath = r"C:\Users\Administrator\Downloads\RAG_Sanskrit_Dhanshri_Supratkar\RAG_Sanskrit_Dhanshri_Supratkar\data\Rag-docs.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
    cleaned = clean_text(raw_text)
    chunks = preprocess_text(cleaned)
    print(f"✅ Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---\n{chunk[:300]}...")
