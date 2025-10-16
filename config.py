import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://smartcv_user:admin123@localhost:5432/smartcv"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
