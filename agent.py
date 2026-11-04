import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import litellm
from dotenv import load_dotenv

from prompts import END_CALL_NAME, HANGUP_TOOL_NAMES, with_plan

# 실행 위치(cwd)와 무관하게 프로젝트 루트의 .env를 읽는다. 이미 설정된 환경변수는 덮어쓰지 않는다.
load_dotenv(Path(__file__).parent / ".env")

# 모델 설정은 .env에서 온다(.env.example 참고).
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "")
# 비어 있으면 api_base 없이 호스팅 모델로 호출한다.
DEFAULT_API_BASE = os.environ.get("LLM_API_BASE") or None
# 로컬 서버용 자리표시 키. optiq serve는 'sk-optiq-' 접두사를 요구하고 mlx_lm.server는 검사하지 않는다.
API_KEY = os.environ.get("API_KEY") or None
# 비어 있으면 Qwen3.5 권장값 1.0. 다른 모델은 .env의 LLM_TEMPERATURE로 낮춘다.
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE") or 1.0)

# Qwen3.5 thinking 모드 권장 샘플링 (temperature 1.0, top_p 0.95, top_k 20,
# presence_penalty 1.5). presence_penalty는 1.5에서 외국어 단어가 섞여 0.5로 낮췄다.
# mlx_lm.server는 generation_config.json을 읽지 않으므로 요청마다 넘긴다.
_MLX_EXTRA_BODY = {
    "top_k": 20,
    "presence_penalty": 0.5,
    # .env의 LLM_EXTRA_BODY(JSON)로 덮어쓰거나 더한다. 예: {"presence_penalty": 0, "min_p": 0.05}
    **json.loads(os.environ.get("LLM_EXTRA_BODY") or "{}"),
}

# max_tokens로 잘려 닫는 태그가 없는 <think> 블록도 끝까지 지운다.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.DOTALL)
_LABEL_RE =re.compile(r"^\s*[\[(]?[가-힣A-Za-z0-9 ]{1,12}[\])]?\s*:\s*")
_ACTION_RE = re.compile(r"\([^)]*\)|\*[^*]*\*")
# 서버가 파싱하지 못하고 텍스트로 남긴 종료 표시: `<end_call>`, `[END CALL]` 등.
_HANGUP_ALT = "|".join(HANGUP_TOOL_NAMES)
_END_TAG_RE = re.compile(
    rf"</?(?P<name>{_HANGUP_ALT})\s*/?>|\[\s*END[ _-]?CALL\s*\]", re.IGNORECASE
)
# 모델이 흉내 내는 `[PHONE_CALL_STARTED]`, `[]` 같은 대괄호 토큰. 음성 발화가 아니다.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
# 이모지와 한자·가나 등 한글이 아닌 CJK 문자. 음성 발화에는 나올 수 없다.
_NON_SPEECH_RE = re.compile(
    "[\u2600-\u27bf\U0001f000-\U0001faff\ufe0f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"
)
# 한 발화 안에서 도구를 부르고 결과를 받아 다시 생성하는 최대 횟수.
MAX_TOOL_ROUNDS = 3
# 학습자는 위키를 읽고 고치느라 도구를 더 많이 부른다.
LEARN_TOOL_ROUNDS = 8
MAX_TOOL_RESULT_CHARS = 3000
# 이보다 짧은 계획은 계획이 아니다("user" 한 단어만 답한 적이 있다). 붙이지 않고 계획 없이 진행한다.
MIN_PLAN_CHARS = 20
_QUOTES = "\"'“”‘’「」"
_DOUBLE_QUOTE_RE = re.compile("[\"“”「」]")


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    api_base: str | None = DEFAULT_API_BASE
    api_key: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = 0.95
    # thinking 토큰도 포함되는 한도라 곧 think 최대 길이다. 넘으면 대사가 비어 한 번 재시도한다.
    max_tokens: int = 2048
    # 통화 전 계획 생성의 한도(thinking 포함). 0이면 계획 단계를 건너뛴다.
    plan_max_tokens: int = 8192
    extra_body: dict | None = None

    def __post_init__(self):
        if not self.model:
            raise ValueError("모델이 지정되지 않았습니다. .env의 LLM_MODEL을 채우거나 --model을 주세요.")
        # 로컬 서버(api_base 지정)일 때만 자리표시 키와 MLX 전용 옵션을 쓴다.
        # 호스팅 모델은 api_key=None으로 두어 litellm이 환경변수 키를 쓰게 한다.
        if self.api_base:
            if self.api_key is None:
                self.api_key = API_KEY
            if self.extra_body is None:
                self.extra_body = dict(_MLX_EXTRA_BODY)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("api_key")
        return d


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    return text.split("</think>")[-1]


def _hangup_name(tool_calls, content: str) -> str:
    """통화를 끝내는 도구 이름을 돌려준다. 없으면 빈 문자열."""
    for tc in tool_calls or []:
        if tc.function.name in HANGUP_TOOL_NAMES:
            return tc.function.name
    visible = _strip_think(content)
    for block in _TOOL_CALL_RE.findall(visible):
        for name in HANGUP_TOOL_NAMES:
            if name in block:
                return name
    m = _END_TAG_RE.search(visible)
    if m:
        return (m.group("name") or END_CALL_NAME).lower()
    return ""


def clean_utterance(text: str) -> str:
    """모델 출력에서 thinking, tool call 원문, 화자 라벨, 행동 묘사, 감싼 따옴표를 걷어낸다."""
    text = _strip_think(text)
    text = _TOOL_CALL_RE.sub("", text)
    text = _LABEL_RE.sub("", text, count=1)  # `[김민수]: ` 라벨은 대괄호 제거보다 먼저
    text = _END_TAG_RE.sub("", text)
    text = _BRACKET_RE.sub("", text)
    text = _ACTION_RE.sub("", text)
    text = _NON_SPEECH_RE.sub("", text)
    text = _DOUBLE_QUOTE_RE.sub("", text)  # 음성 발화에 따옴표는 없다. 중간에 짝 없이 남는 것도 지운다
    return re.sub(r"\s+", " ", text).strip().strip(_QUOTES).strip()


_SENTENCE_END_RE = re.compile(r"(?<=[^\d\W][.!?。])\s+")
# 한 발화의 최대 문장 수. .env의 CALLER_MAX_SENTENCES, CALLEE_MAX_SENTENCES로 바꿀 수 있다(기본 2).
MAX_SENTENCES = {
    "caller": int(os.environ.get("CALLER_MAX_SENTENCES") or 2),
    "callee": int(os.environ.get("CALLEE_MAX_SENTENCES") or 2),
}


def limit_sentences(text: str, n: int) -> str:
    return " ".join(_SENTENCE_END_RE.split(text)[:n])


@dataclass
class PhoneAgent:
    role: str  # "caller" | "callee"
    persona: dict | None
    name: str
    system_prompt: str
    llm: LLMConfig
    tools: list[dict] | None = None
    # (이름, 인자)를 받아 결과 문자열을 돌려주는 도구 실행기. end_call 외의 도구 호출을 처리한다.
    tool_runner: Callable[[str, str], str] | None = None
    # 상대의 말(user 메시지) 앞에 붙이는 라벨. 예: "[낯선 사람] ". 모델이 상대를 도와줄 사용자로 여기지 않게 한다.
    partner_label: str = ""
    messages: list[dict] = field(init=False)
    # 이 통화에서 이 에이전트가 모델로 만든 발화 수. 수신자의 고정 인사는 세지 않는다.
    spoken: int = field(init=False, default=0)

    def __post_init__(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _complete(
        self, messages=None, max_tokens=None, tools="default"
    ) -> tuple[str, str, dict]:
        kwargs = dict(
            model=self.llm.model,
            messages=self.messages if messages is None else messages,
            api_base=self.llm.api_base,
            api_key=self.llm.api_key,
            temperature=self.llm.temperature,
            top_p=self.llm.top_p,
            max_tokens=max_tokens or self.llm.max_tokens,
        )
        if self.llm.extra_body:
            kwargs["extra_body"] = self.llm.extra_body
        tools = self.tools if tools == "default" else tools
        if tools:
            kwargs["tools"] = tools
        t0 = time.perf_counter()
        resp = litellm.completion(**kwargs)
        message = resp.choices[0].message
        # 서버가 생성 엔진 오류를 빈 응답(토큰 0)으로 돌려주는 경우가 있다. 모델의 빈 응답으로
        # 기록하지 않고 오류로 올려서 통화가 `error`로 끝나게 한다(리뷰·학습에서 빠진다).
        if (
            resp.usage
            and resp.usage.completion_tokens == 0
            and not message.content
            and not message.tool_calls
        ):
            raise RuntimeError("서버가 빈 응답(완성 토큰 0)을 돌려줬습니다")
        meta = {
            "latency_s": round(time.perf_counter() - t0, 2),
            "finish_reason": resp.choices[0].finish_reason,
            "usage": resp.usage.model_dump() if resp.usage else None,
            "reasoning": getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None),
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in message.tool_calls or []
                if tc.function.name not in HANGUP_TOOL_NAMES
            ],
        }
        content = message.content or ""
        # 서버가 tool_calls로 파싱하지 못하고 content에 <tool_call> 원문을 남기는 경우도 본다.
        # thinking 안에서 언급만 한 호출은 세지 않도록 thinking을 먼저 걷어낸다.
        ended = _hangup_name(message.tool_calls, content)
        return content, ended, meta

    def plan(self, request: str) -> dict | None:
        """통화 전에 계획을 한 번 세워 시스템 프롬프트의 [Plan]에 붙인다.

        위키 도구가 있으면 계획을 세우는 동안 조회할 수 있다(end_call은 제외).
        비었거나 max_tokens에서 잘린 계획은 붙이지 않고 None을 돌려준다.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": request},
        ]
        plan_tools = [
            t for t in self.tools or [] if t["function"]["name"] not in HANGUP_TOOL_NAMES
        ]
        raw, _, meta, lookups = self._generate(
            messages=messages,
            max_tokens=self.llm.plan_max_tokens,
            tools=plan_tools,
            final_tools=[],
        )
        text = _strip_think(raw).strip()
        if len(text) < MIN_PLAN_CHARS or meta["finish_reason"] == "length":
            return None
        self.system_prompt = with_plan(self.system_prompt, text)
        self.messages[0]["content"] = self.system_prompt
        return {"text": text, **meta, **({"wiki": lookups} if lookups else {})}

    def _run_tools(self, messages: list[dict], raw: str, calls: list[dict]) -> list[dict]:
        """도구 호출을 실행해 assistant/tool 메시지를 messages에 붙이고 조회 기록을 돌려준다."""
        messages.append(
            {
                "role": "assistant",
                "content": _TOOL_CALL_RE.sub("", _strip_think(raw)).strip(),
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    }
                    for c in calls
                ],
            }
        )
        lookups = []
        for c in calls:
            result = self.tool_runner(c["name"], c["arguments"] or "{}")
            messages.append(
                {"role": "tool", "tool_call_id": c["id"], "content": result[:MAX_TOOL_RESULT_CHARS]}
            )
            lookups.append({"name": c["name"], "arguments": c["arguments"], "result": result})
        return lookups

    def _generate(
        self, messages=None, max_tokens=None, tools=None, final_tools=None,
        max_rounds=MAX_TOOL_ROUNDS,
    ) -> tuple[str, str, dict, list[dict]]:
        """도구 호출을 처리하며 한 번 생성한다. 마지막 라운드는 final_tools만 열어 반드시 글로 답하게 한다.

        기본값은 통화 발화용: messages=self.messages, 모든 도구, 마지막 라운드는 end_call만.
        """
        messages = self.messages if messages is None else messages
        if tools is None:
            tools = self.tools
        if final_tools is None:
            final_tools = [t for t in self.tools or [] if t["function"]["name"] in HANGUP_TOOL_NAMES]
        has_lookup = any(t["function"]["name"] not in HANGUP_TOOL_NAMES for t in tools or [])
        lookups: list[dict] = []
        for round_ in range(max_rounds + 1):
            last = round_ == max_rounds or not (self.tool_runner and has_lookup)
            raw, ended, meta = self._complete(
                messages=messages,
                max_tokens=max_tokens,
                tools=final_tools if last else tools,
            )
            calls = meta.pop("tool_calls")
            if last or ended or not calls:
                break
            lookups += self._run_tools(messages, raw, calls)
        return raw, ended, meta, lookups

    def respond(self, incoming: str) -> tuple[str, str, dict]:
        self.messages.append({"role": "user", "content": f"{self.partner_label}{incoming}"})
        first = self.spoken == 0
        # 첫 발화는 끊기일 수 없다. 작은 모델이 인사하려다 end_call을 잘못 부르는 일이 잦아,
        # 위키 도구가 있으면 그것만 준다. 도구가 하나도 없는 요청은 모델이 장문 사고로 흘러
        # max_tokens에서 잘리므로 위키 도구가 없으면 기본 목록(end_call)을 두고 호출만 무시한다.
        lookup_tools = [t for t in self.tools or [] if t["function"]["name"] not in HANGUP_TOOL_NAMES]
        opening = {"tools": lookup_tools} if first and lookup_tools else {}
        text, ended, meta, lookups, ignored = "", "", {}, [], 0
        for _ in range(2):  # 빈 응답이면 한 번 재시도 (바로 끊은 경우는 제외)
            raw, ended, meta, used = self._generate(**opening)
            lookups += used
            text = clean_utterance(raw)
            if ended and (first or text.endswith("?")):
                # 첫 발화나 질문으로 끝나는 발화는 작별이 아니다. 끊기 호출만 무시한다.
                ended, ignored = "", ignored + 1
            if not ended and (n := MAX_SENTENCES.get(self.role)):  # 마무리 인사·사과는 자르지 않는다
                limited = limit_sentences(text, n)
                if limited != text:
                    meta["truncated_from"] = text
                text = limited
            # 도구를 쓴 뒤 비었으면 다시 시도하지 않는다 (이미 늘어난 기록 위에서 호출이 불어난다).
            if text or ended or used:
                break
        if ignored:
            meta["ignored_end_call"] = ignored
        if lookups:
            meta["wiki"] = lookups
        self.messages.append({"role": "assistant", "content": text})
        self.spoken += 1
        return text, ended, meta
