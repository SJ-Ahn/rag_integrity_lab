import logging
import re
import json
from dataclasses import dataclass
from typing import Optional, Literal

from config.settings import settings
from service.llm_factory import get_llm_service

LOGGER = logging.getLogger("service.router")

@dataclass
class RouteResult:
    intent: Literal["chitchat", "search_query"]
    confidence: float
    # If chitchat, this might contain the pre-generated response (optimization)
    direct_response: Optional[str] = None

class ChatRouter:
    def __init__(self):
        # Greetings and casual conversation patterns
        self.greetings = {
            "안녕", "안녕하세요", "하이", "헬로", "반가워", "누구니", "자기소개", "고마워", "감사",
            "hello", "hi", "thanks", "tired", "힘들어", "기분"
        }
        
    def _classify_rule_based(self, query: str) -> Optional[RouteResult]:
        """
        Fast, rule-based classification. Returns None if ambiguous.
        """
        query_clean = query.strip().lower()
        
        # 1. Very short queries -> Chitchat
        if len(query_clean) < 2:
            return RouteResult(intent="chitchat", confidence=1.0, direct_response="네, 듣고 있습니다.")

        # 2. Strong signals
        strong_chitchat_phrases = ["누구니", "자기소개", "who are you", "너는 누구", "너는?", "너 뭐야"]
        if any(phrase in query_clean for phrase in strong_chitchat_phrases):
            return RouteResult(
                intent="chitchat", 
                confidence=1.0, 
                direct_response="저는 AWS 문서 기반 기술 지원 어시스턴트입니다."
            )

        # 3. Simple Keyword check
        query_no_punct = re.sub(r'[^\w\s]', '', query_clean)
        tokens = set(query_no_punct.split())
        common_greetings = self.greetings.intersection(tokens)
        
        if common_greetings:
            if len(tokens) <= 5:
                # Simple greeting
                return RouteResult(
                    intent="chitchat", 
                    confidence=0.9, 
                    # Let the handler decide the text, or return simple default
                    direct_response="안녕하세요! AWS 서비스에 대해 궁금한 점이 있으신가요?"
                )
                
        return None

    def _classify_llm_based(self, query: str) -> RouteResult:
        """
        Slow, accurate LLM-based classification.
        Generates response for chitchat to save a second call.
        """
        llm = get_llm_service()
        if not llm:
            LOGGER.warning("LLM Service unavailable for routing. Fallback to Search.")
            return RouteResult(intent="search_query", confidence=0.0)

        # Sanitize query for prompt injection
        safe_query = query.strip().replace("\n", " ").replace('"', "'") 
        
        # Prompt engineering for combined Classification + Chitchat Response
        # Using more explicit "Input/Output" format which works better with Llama-3 completion
        prompt = f"""
        당신은 AI 라우터입니다. 사용자의 질문을 분석하여 오직 JSON 형식으로만 응답하세요. 다른 설명은 하지 마세요.
        
        [규칙]
    1. 질문이 잡담(인사, 감정, 일반 대화)이거나 **자신의 역할/기능을 묻는 질문**이면 "intent": "chitchat", "response": "적절한 답변"을 반환하세요.
    2. 질문이 기술적인 내용(AWS, EC2, 개발, IT)이거나 정보 검색이 필요하면 "intent": "search_query"만 반환하세요.
    
    [예시]
    Input: "안녕"
    Output: {{"intent": "chitchat", "response": "안녕하세요! 무엇을 도와드릴까요?"}}

    Input: "너는 누구니?"
    Output: {{"intent": "chitchat", "response": "저는 AWS 문서를 기반으로 답변해드리는 AI 어시스턴트입니다."}}
    
    Input: "EC2 인스턴스 어떻게 만들어?"
    Output: {{"intent": "search_query"}}
    
    Input: "기분 좋아"
    Output: {{"intent": "chitchat", "response": "다행이네요! 좋은 기운으로 함께 AWS 문제를 해결해봐요!"}}

        Input: "{safe_query}"
        Output:
        """
        
        try:
            # Use generate_text with stop tokens to prevent hallucination of new examples
            # This is CRITICAL for Local Llama models
            response_text = llm.generate_text(prompt, stop=["Input:", "Output:", "\n\n", "```"])
            
            # Parse JSON response
            # Improved Extraction: Find text between first '{' and last '}'
            LOGGER.info(f"Raw Router LLM Output: {response_text}")
            match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if match:
                cleaned_text = match.group(1)
            else:
                # Fallback to simple cleaning if regex fails (e.g. no braces found)
                cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
            
            LOGGER.info(f"Cleaned Router Payload: {cleaned_text}")
            data = json.loads(cleaned_text)
            
            intent = data.get("intent", "search_query")
            direct_response = data.get("response")
            
            if intent not in ["chitchat", "search_query"]:
                intent = "search_query"
                
            return RouteResult(intent=intent, confidence=0.9, direct_response=direct_response)
            
        except Exception as e:
            LOGGER.error(f"Routing LLM Error: {e}")
            
        return RouteResult(intent="search_query", confidence=0.5)

    def route(self, query: str) -> RouteResult:
        # 1. Rule Based
        decision = self._classify_rule_based(query)
        if decision:
            return decision
            
        # 2. LLM Based
        return self._classify_llm_based(query)

    def get_chitchat_response(self, query: str) -> str:
        # This function is deprecated by the RouteResult structure but kept for compatibility if needed.
        return "안녕하세요."

