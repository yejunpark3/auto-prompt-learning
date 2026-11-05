"""강화학습(자연스러운 통화) 도구. 시나리오를 모른다: 통화 기록과 시나리오 중립 프롬프트(prompts.py)만 쓴다.

심판·리뷰어·사실 검사기는 통화를 만드는 모델과 다른 서버(Mac mini의 기본 14B)를 쓴다. `.env`의
JUDGE_LLM_MODEL / JUDGE_LLM_API_BASE / JUDGE_API_KEY / JUDGE_LLM_TEMPERATURE / JUDGE_MAX_TOKENS로 정하고,
비우면 통화용 설정(LLM_*)을 쓴다.
"""
import os

from agent import LLMConfig

JUDGE_MAX_TOKENS = 3000  # 한 번의 심판·검사 응답 한도(thinking 포함). 느린 서버의 요청 시간 제한에 걸리지 않게 낮춘다


def judge_config(**overrides) -> LLMConfig:
    kw: dict = {}
    if model := os.environ.get("JUDGE_LLM_MODEL"):
        kw["model"] = model
    if "JUDGE_LLM_API_BASE" in os.environ:  # 빈 문자열이면 호스팅 모델(api_base 없음)
        kw["api_base"] = os.environ["JUDGE_LLM_API_BASE"] or None
    if key := os.environ.get("JUDGE_API_KEY"):
        kw["api_key"] = key
    if temp := os.environ.get("JUDGE_LLM_TEMPERATURE"):
        kw["temperature"] = float(temp)
    kw["plan_max_tokens"] = int(os.environ.get("JUDGE_MAX_TOKENS") or JUDGE_MAX_TOKENS)
    return LLMConfig(**{**kw, **overrides})
