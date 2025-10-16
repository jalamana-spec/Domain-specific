"""
Generate embeddings for job descriptions and candidate resumes and save them.
Outputs:
 - data/job_embeddings.pkl
 - data/candidate_embeddings.pkl
"""
import os, joblib, json
import pandas as pd
import numpy as np
from config import Config
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(Config.EMBEDDING_MODEL)

def embed_texts(texts, chunk=256):
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

if __name__ == "__main__":
    data_dir = Config.DATA_DIR
    # Jobs
    jobs_path = os.path.join(data_dir, "job_title_des.csv")
    dfj = pd.read_csv(jobs_path, engine="python")
    ttexts = []
    for _, r in dfj.iterrows():
        title = str(r.get('title') or r.get('JobTitle') or "")
        desc = str(r.get('description') or r.get('JobDesc') or "")
        ttexts.append((title + ". " + desc).strip())
    job_embs = embed_texts(ttexts)
    job_out = os.path.join(data_dir, "job_embeddings.pkl")
    joblib.dump({"rows": dfj, "embeddings": job_embs}, job_out)
    print("Saved job embeddings to", job_out)

    # Candidates
    resumes_path = os.path.join(data_dir, "AI_Resume_Screening_final.csv")
    dfr = pd.read_csv(resumes_path, engine="python")
    # find resume text column
    if 'resume_text' in dfr.columns:
        texts = dfr['resume_text'].fillna("").tolist()
    else:
        # attempt to find long text column
        candidate_cols = [c for c in dfr.columns if dfr[c].dtype == object and dfr[c].str.len().mean() > 50]
        texts = dfr[candidate_cols[0]].fillna("").tolist() if candidate_cols else dfr.astype(str).agg(" ".join, axis=1).tolist()
    cand_embs = embed_texts(texts)
    cand_out = os.path.join(data_dir, "candidate_embeddings.pkl")
    joblib.dump({"rows": dfr, "embeddings": cand_embs}, cand_out)
    print("Saved candidate embeddings to", cand_out)
