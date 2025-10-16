"""
Take raw CSV/Resume text and create cleaned resume text.
Save outputs to data/parsed_resumes.csv with columns: candidate_id,name,email,resume_text
"""
import os, csv, pandas as pd
from config import Config
import pdfplumber
from docx import Document

def extract_text_from_pdf(path):
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text.append(p.extract_text() or "")
        return "\n".join(text)
    except Exception:
        return ""

def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def normalize(text):
    import re
    t = text or ""
    t = re.sub(r"\r\n", "\n", t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

if __name__ == "__main__":
    # Example: read AI_Resume_Screening_final.csv for columns resume_text (or similar)
    df = pd.read_csv(os.path.join(Config.DATA_DIR, "AI_Resume_Screening_final.csv"), engine="python")
    # try common columns
    if 'resume_text' in df.columns:
        df['parsed_resume'] = df['resume_text'].fillna("").apply(normalize)
    else:
        # fall back: combine columns
        text_cols = [c for c in df.columns if 'resume' in c.lower() or 'cv' in c.lower() or 'summary' in c.lower()]
        if text_cols:
            df['parsed_resume'] = df[text_cols].fillna("").agg(" ".join, axis=1).apply(normalize)
        else:
            # try combining all text columns
            df['parsed_resume'] = df.astype(str).agg(" ".join, axis=1).apply(normalize)
    parsed_path = os.path.join(Config.DATA_DIR, "parsed_resumes.csv")
    df[['parsed_resume']].to_csv(parsed_path, index=False)
    print("Saved parsed resumes to", parsed_path)
