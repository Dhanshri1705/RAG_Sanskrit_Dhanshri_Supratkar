from docx import Document

def docx_to_txt(docx_path, txt_path):
    """
    Convert a .docx file (English or Sanskrit) into plain .txt.
    Keeps paragraph breaks and ensures UTF-8 encoding.
    """
    doc = Document(r"C:\Users\Administrator\Downloads\RAG_Sanskrit_Dhanshri_Supratkar\RAG_Sanskrit_Dhanshri_Supratkar\data\Rag-docs.docx")
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))

if __name__ == "__main__":
    docx_to_txt("../data/Rag-docs.docx", "../data/Rag-docs.txt")
    print("Conversion complete: Rag-docs.txt created.")
