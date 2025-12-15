import logging
import pathlib
from typing import Optional

try:
    from llama_cpp import Llama  # type: ignore
except ImportError:
    Llama = None

from service.llm_base import BaseLLMService
from config.settings import settings

LOGGER = logging.getLogger("service.llm.local")

class LocalLLMService(BaseLLMService):
    def __init__(self, model_path: str = settings.LOCAL_MODEL_PATH, n_ctx: int = settings.LLM_CONTEXT_WINDOW):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed.")
        
        self.n_ctx = n_ctx
        
        # Resolve Generic Path if it's a directory or a specific file
        # Logic adapted from original app.py but simplified to trust settings.LOCAL_MODEL_PATH primarily
        # unless it is clearly a directory requiring search.
        chunk_model_path = pathlib.Path(model_path)
        if chunk_model_path.is_dir():
             gguf_files = list(chunk_model_path.rglob("*.gguf"))
             if not gguf_files:
                 raise FileNotFoundError(f"No GGUF file found in {model_path}")
             model_file = str(gguf_files[0])
        else:
             model_file = model_path

        LOGGER.info(f"Loading Local LLM from {model_file} (n_ctx={self.n_ctx}, n_gpu_layers=-1)")
        
        try:
            self.llm = Llama(
                model_path=model_file,
                n_ctx=self.n_ctx,
                n_gpu_layers=-1,
                verbose=False
            )
        except Exception as e:
            # If default path fails, try to fallback to old logic or fail gracefully
             LOGGER.error(f"Failed to load model from {model_file}: {e}")
             raise

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            output = self.llm(
                prompt,
                max_tokens=kwargs.get("max_tokens", 512),
                stop=kwargs.get("stop", []),
                echo=False
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            LOGGER.error(f"Local LLM Text Gen Error: {e}")
            raise

    def _manage_history(self, system_prompt: str, user_message: str, history: list) -> list:
        """
        Calculates expected tokens and truncates history if it exceeds context window.
        """
        SAFETY_BUFFER = 2000
        MAX_INPUT_TOKENS = self.n_ctx - SAFETY_BUFFER
        
        if not history:
             return []

        def count_tokens(text: str) -> int:
             try:
                 return len(self.llm.tokenize(text.encode("utf-8", errors="ignore")))
             except Exception:
                 return len(text) // 3

        sys_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(user_message)
        
        needed = sys_tokens + user_tokens
        limit_for_history = MAX_INPUT_TOKENS - needed
        
        if limit_for_history <= 0:
            return []
            
        selected_history = []
        current_history_tokens = 0
        
        for msg in reversed(history):
            msg_tokens = count_tokens(msg.get("content", ""))
            if current_history_tokens + msg_tokens > limit_for_history:
                break
            selected_history.insert(0, msg)
            current_history_tokens += msg_tokens
            
        return selected_history

    def generate_answer(self, query: str, context_chunks: list[dict], history: list = []) -> str:
        # Build Context
        context_text = ""
        total_len = 0
        limit = 10000 
        
        for idx, chunk in enumerate(context_chunks[:5], start=1):
            text = chunk.get('text', '')
            if total_len + len(text) > limit:
                break
            context_text += f"[{idx}] {text}\n\n"
            total_len += len(text)
        
        system_prompt = (
            "당신은 구글의 'Gemini'와 같은 고성능 AI 어시스턴트입니다."
            "누구인지 물어거나 자기소개할때는 AWS 기술 어시스턴트라고 소개해주세요."
            "주어진 [Context]를 바탕으로 질문에 대해 풍부하고 구조적인 답변을 제공하세요.\n\n"
            "**답변 스타일 가이드 (Gemini Style):**\n"
            "1. **전문적이고 포괄적인 구조**: 답변은 '서론(요약) -> 본론(상세 설명) -> 결론(추가 제안)'의 흐름을 갖추세요.\n"
            "2. **구조화된 서식 활용**: 가독성을 높이기 위해 적절한 **헤더(###)**, **글머리 기호(•)**, **번호 매기기**, **굵은 글씨**를 적극적으로 사용하세요.\n"
            "3. **충실한 분량**: 너무 짧게 줄이지 말고, 사용자가 충분히 이해할 수 있도록 상세한 뉘앙스와 맥락을 설명하세요.\n"
            "4. **명확한 근거 표기**: [Context]의 내용을 인용할 때는 문장이나 단락 끝에 반드시 `[1]`, `[2]`와 같이 출처 번호를 명시하세요.\n"
            "5. **소개 및 마무리**: 시작할 때 사용자의 질문을 이해했음을 부드럽게 표현하고, 끝맺음 말로 도움이 더 필요한지 물어보세요.\n"
            "6. **톤앤매너**: 자신감 있고 친절하며 전문적인 '해요체'를 사용하세요."
        )
        
        user_message = f"""[Context]
{context_text}

[Question]
{query}

[Answer]
"""
        managed_history = self._manage_history(system_prompt, user_message, history)
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in managed_history:
             role = msg.get("role", "user")
             if role not in ["user", "assistant"]:
                 role = "user"
             messages.append({"role": role, "content": msg.get("content", "")})
             
        messages.append({"role": "user", "content": user_message})
        
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=0
        )
        
        return response["choices"][0]["message"]["content"]
