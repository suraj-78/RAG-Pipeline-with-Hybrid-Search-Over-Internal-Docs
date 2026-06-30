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
    
    # Text Processing Hyper-parameters (optimized for layout structural boundaries)
    CHUNK_SIZE = 1500  # Target chunk size in characters (approx 250-300 words)
    CHUNK_OVERLAP = 300 # Character overlap to prevent text splitting at boundaries
    RETRIEVAL_TOP_K = 10
    RERANK_TOP_N = 5   # Number of top candidate chunks to supply to cross-encoder
    
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