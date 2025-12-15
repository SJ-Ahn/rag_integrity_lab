import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from config.settings import settings

print(f"LLM_PROVIDER: {settings.LLM_PROVIDER}")
print(f"OPENAI_MODEL: {settings.OPENAI_MODEL}")
