"""
Demo: given a job row index and top-K, show best candidates using saved embeddings + matcher.
"""
import os, joblib, numpy as np
from config import Config
from sklearn.metrics.pairwise import cosine_similarity

data_dir = Config.DATA_DIR
jobdb = joblib.load(os.path.join(data_dir,"job_embeddings.pkl"))
candb = joblib.load(os.path.join(data_dir,"candidate_embeddings.pkl"))
matcher = joblib.load(os.path.join(data_dir,"matcher_rf.pkl"))

job_rows = jobdb['rows']
job_embs = jobdb['embeddings']
cand_rows = candb['rows']
cand_embs = candb['embeddings']

def score_pair(jidx, cidx):
    emb_sim = float(cosine_similarity([job_embs[jidx]],[cand_embs[cidx]])[0,0])
    # naive skill ratio approximated by token overlap
    job_text = str((job_rows.iloc[jidx].get('title') or "") + " " + (job_rows.iloc[jidx].get('description') or ""))
    cand_text = str((cand_rows.iloc[cidx].get('resume_text') or "") if 'resume_text' in cand_rows.columns else cand_rows.iloc[cidx].astype(str).agg(" ".join))
    job_tokens = set(job_text.lower().split()[:200])
    cand_tokens = set(cand_text.lower().split()[:200])
    overlap = len(job_tokens.intersection(cand_tokens))
    skill_ratio = overlap / max(1, len(job_tokens))
    features = [[emb_sim, skill_ratio]]
    score = matcher.predict(features)[0]
    return score, emb_sim, skill_ratio

if __name__ == "__main__":
    jidx = 0
    scores = []
    for ci in range(len(cand_embs)):
        s, e, k = score_pair(jidx, ci)
        scores.append((ci, s, e, k))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:20]
    for ci, s, e, k in scores_sorted:
        print(f"Candidate idx {ci}, score {s:.4f}, emb_sim {e:.4f}, skill_ratio {k:.4f}")
