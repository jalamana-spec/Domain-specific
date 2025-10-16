"""
Creates a skill mapping JSON from skills_with_aliases_100.csv
"""
import os, json
import pandas as pd
from config import Config

def build_mapping():
    path = os.path.join(Config.DATA_DIR, "skills_with_aliases_100.csv")
    df = pd.read_csv(path, engine='python')
    mapping = {}
    # assume columns skill, aliases
    for _, r in df.iterrows():
        skill = str(r.get('skill') or r.get('Skill') or r.get('SkillName', '')).strip()
        aliases = r.get('aliases') or r.get('aliases') or r.get('Aliases') or ""
        if pd.isna(aliases): aliases = ""
        parts = [p.strip().lower() for p in str(aliases).split(';') if p.strip()]
        parts.append(skill.lower())
        mapping[skill.lower()] = list(sorted(set(parts)))
    out_path = os.path.join(Config.DATA_DIR, "skill_mapping.json")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print("Wrote skill mapping to", out_path)

if __name__ == "__main__":
    build_mapping()
