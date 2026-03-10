# RAG_Sanskrit_Dhanshri_Supratkar
# Sanskrit Document Retrieval-Augmented Generation (RAG) System

A lightweight Retrieval-Augmented Generation (RAG) pipeline for Sanskrit text retrieval and summarization using CPU-based models.  
The system allows users to query Sanskrit documents and retrieve the most relevant story chunk, optionally generating a concise summary.

---

# Project Overview

This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) syste built for Sanskrit documents.

The pipeline performs:

1. Document Loading – Reads Sanskrit text corpus.
2. Preprocessing – Cleans text and splits it into chunks.
3. Retrieval – Finds the most relevant chunk for a query.
4.Generation – Generates a summary using a lightweight language model.
5. Query Interface – Allows users to input queries and see results.

The system is designed to run fully on CPU, making it lightweight and easy to deploy.

---

# 📂 Project Folder Structure


RAG_Sanskrit_Dhanshri_Supratkar/
│
├── code/
│   ├── preprocess.py
│   ├── retriever.py
│   └── rag_light.py
|   ├── gpt4all-falcon-q4_0.gguf
│   └── gpt4all-model.bi
│   └── doc_to_txt.py
│   └── light.py
│
├── data/
│   └── Rag-docs.txt
│   └── Rag-docs.docx
│
├── report/
│   └── RAG_Report.pdf
│
├── README.md



Folder Explanation

| Folder/File        | Description                                |
|--------------------|--------------------------------------------|
| `code/`            | Contains all Python implementation scripts |
| `data/`            | Sanskrit text dataset used for retrieval   |
| `report/`          | Final project report (PDF)                 |
| `requirements.txt` | Python dependencies                        |
| `README.md`        | Project documentation                      |
| `venv/`            | Python virtual environment                 |

---

# ⚙️ Installation & Setup

#1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd RAG_Sanskrit_Dhanshri_Supratkar
````

---

 2️⃣ Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

 ▶️ How to Run the Project

## Step 1 — Preprocess Sanskrit Documents

Run the preprocessing script to clean and split the corpus.

```bash
python code/preprocess.py
```

This script will:

* Load the Sanskrit document
* Clean unwanted characters
* Split the text into manageable chunks
* Display preview of chunks

Example output:

```
Total chunks created: 5
Preview chunk:
भक्तस्य कथा...
```

---

## Step 2 — Run the RAG Query System

Start the interactive query interface:

```bash
python code/rag_light_format.py
```

Example query:

```
भक्तस्य कथा
```

Output example:

```
Retrieved Passage:
भक्तस्य कथा...

Generated Summary:
This story describes the devotion of a disciple...
```

---

# System Architecture


User Query
     │
     ▼
Retriever
     │
     ▼
Relevant Document Chunk
     │
     ▼
Generator (Flan-T5)
     │
     ▼
Final Response

### Components

| Component       | Role                                           |
| --------------- | ---------------------------------------------- |
| Preprocessor    | Cleans Sanskrit text and splits it into chunks |
| Retriever       | Finds the most relevant chunk for the query    |
| Generator       | Summarizes retrieved content                   |
| Query Interface | Allows user interaction                        |

---

#  Performance Notes

* Runs entirely on CPU
* Suitable for small to medium text corpora
* Retrieval accuracy depends on story heading
* Chunk size optimized for Flan-T5 token limit

For larger datasets, FAISS indexin improves retrieval speed.

---

#  Technologies Used

| Technology            | Purpose                     |
| --------------------- | --------------------------- |
| Python                | Programming language        |
| Transformers          | Language model inference    |
| Sentence Transformers | Text embeddings             |
| FAISS                 | Efficient similarity search |
| PyTorch               | Model backend               |

---

#  Example Queries

```
भक्तस्य कथा
वृद्धायाः कथा
राज्ञः कथा
```

Each query retrieves only the relevant story chunk.

---

# 📄 Deliverables

This repository includes:

*  Complete Python implementation
*  Sanskrit sample dataset
*  Retrieval-Augmented Generation pipeline
*  Project documentation
*  Technical report

---

# Future Improvements

Possible enhancements:

* Support multiple document formats (PDF, DOCX)
* Add vector database (Chroma / Pinecone
* Improve semantic retrieval
* Deploy as a web application
* Add multilingual query support

---

