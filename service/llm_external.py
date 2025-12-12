import logging
import os
from typing import Optional

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from service.llm_base import BaseLLMService
from config.settings import settings

LOGGER = logging.getLogger("service.llm.external")

class OpenAILLMService(BaseLLMService):
    def __init__(self, api_key: Optional[str] = None):
        if openai is None:
            raise RuntimeError("openai package is not installed.")
        
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API Key is missing. Set OPENAI_API_KEY in settings.")
            
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = settings.OPENAI_MODEL

    def generate_answer(self, query: str, context_chunks: list[dict], history: list = []) -> str:
        # Simplified Context Building for External API
        context_text = ""
        for idx, chunk in enumerate(context_chunks[:5], start=1):
            context_text += f"[{idx}] {chunk.get('text', '')}\n\n"
            
        system_prompt = (
            "당신은 구글의 'Gemini'와 같은 고성능 AI 어시스턴트입니다."
            "누구인지 물어거나 자기소개할때는 AWS 기술 어시스턴트라고 소개해주세요."
            "주어진 [Context]를 바탕으로 질문에 대해 풍부하고 구조적인 답변을 제공하세요.\n"
            "답변 스타일은 전문적이고 구조적이어야 하며, 인용 표기([1])를 반드시 포함하세요."
        )
        
        user_message = f"""[Context]
{context_text}

[Question]
{query}
"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history (Simplified: External APIs usually handle larger context, 
        # so we just pass recent history without complex token counting for now)
        # Limit to last 10 messages to be safe
        for msg in history[-10:]:
             role = msg.get("role", "user")
             if role not in ["user", "assistant"]:
                 role = "user"
             messages.append({"role": role, "content": msg.get("content", "")})
             
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            LOGGER.error(f"OpenAI API Error: {e}")
            raise

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            LOGGER.error(f"OpenAI Text Gen Error: {e}")
            raise


class GeminiLLMService(BaseLLMService):
    def __init__(self, api_key: Optional[str] = None):
        if genai is None:
            raise RuntimeError("google-generativeai package is not installed.")
            
        self.api_key = api_key or settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API Key is missing. Set GEMINI_API_KEY in settings.")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)

    def generate_answer(self, query: str, context_chunks: list[dict], history: list = []) -> str:
        context_text = ""
        for idx, chunk in enumerate(context_chunks[:5], start=1):
            context_text += f"[{idx}] {chunk.get('text', '')}\n\n"
            
        system_instructions = (
            "당신은 AWS 기술 어시스턴트입니다.\n"
            "주어진 [Context]를 바탕으로 질문에 대해 답변하세요.\n"
            "답변에 인용 번호([1])를 포함하세요.\n"
            "구조적이고 전문적인 스타일을 유지하세요."
        )

        # Gemini supports system instructions in some versions, but prompts are safer.
        # We construct a chat session.
        chat = self.model.start_chat(history=[])
        
        # Convert history format if needed, but for 'gemini-pro' REST API, 
        # usually we send the prompt directly or manage chat object.
        # Here we use a single turn approach for simplicity in RAG, 
        # effectively treating history as context.
        
        full_prompt = f"""{system_instructions}

[Previous Chat Context]
{history[-5:]}

[Context]
{context_text}

[Question]
{query}
"""
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            LOGGER.error(f"Gemini API Error: {e}")
            raise

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            LOGGER.error(f"Gemini Text Gen Error: {e}")
            raise
