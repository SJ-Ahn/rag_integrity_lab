
import sys
import logging
from llama_cpp import Llama

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = "models/llama3.1/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

try:
    logger.info("Loading model...")
    # Low n_ctx for quick test
    llm = Llama(
        model_path=model_path,
        n_ctx=2048, 
        n_gpu_layers=-1,
        verbose=False
    )
    
    logger.info("Testing max_tokens=0...")
    # Simple prompt
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Count from 1 to 50."}],
        max_tokens=0, # Expecting 'unlimited' (up to context)
        temperature=0.1
    )
    content = response["choices"][0]["message"]["content"]
    logger.info(f"Response length: {len(content)}")
    logger.info("Success with max_tokens=0")
    
except Exception as e:
    logger.error(f"Failed with max_tokens=0: {e}")

    try:
        logger.info("Testing max_tokens=-1...")
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Count from 1 to 50."}],
            max_tokens=-1, 
            temperature=0.1
        )
        content = response["choices"][0]["message"]["content"]
        logger.info(f"Response length: {len(content)}")
        logger.info("Success with max_tokens=-1")
    except Exception as e2:
        logger.error(f"Failed with max_tokens=-1: {e2}")
        sys.exit(1)
