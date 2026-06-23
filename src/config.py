import os
from pathlib import Path

class AppConfig:
    """Centralized governance layer managing system paths, hyper-parameters, and model states."""
    
    # Base Directories Management
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_STORAGE_DIR = str(DATA_DIR / "chroma_db")
    SPARSE_INDEX_PATH = str(DATA_DIR / "sparse_index.pkl")
    
    # Model Architecture Configurations
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    GENERATION_MODEL_NAME = "llama-3.1-8b-instant"
    JUDGE_MODEL_NAME = "llama-3.3-70b-versatile"
    
    # Text Processing Hyper-parameters
   # Insaaf se ye parameters change karo Line 21-24 par:
    # Text Processing Hyper-parameters
    CHUNK_SIZE = 1500  # 500 se badhaakar 1500 characters (approx 250-300 words) karo
    CHUNK_OVERLAP = 300 # Boundary protection badhaakar 300 karo
    RETRIEVAL_TOP_K = 10
    RERANK_TOP_N = 5   # Top candid ates re-ranker ko thode zyada bhejo (3 se 5)
    
    # Security Credentials Gateway
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    @classmethod
    def validate_environment(cls):
        """Enforces environment validation checks before firing system runtimes."""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "CRITICAL SYSTEM FAILURE: 'GROQ_API_KEY' is missing from the environment settings. "
                "Execution terminated."
            )
        # Ensure data infrastructure paths exist on the OS natively
        os.makedirs(cls.DATA_DIR, exist_ok=True)

# Instantiate a global configurations instance to import across application boundaries
config = AppConfig()