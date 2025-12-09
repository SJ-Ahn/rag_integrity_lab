
import sys
import logging
from llama_cpp import Llama

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = "models/llama3.1/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

try:
    logger.info("Attempting to load model with n_ctx=16384")
    llm = Llama(
        model_path=model_path,
        n_ctx=16384,
        n_gpu_layers=-1,
        verbose=True
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise
