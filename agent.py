import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import litellm

from prompts import END_CALL_NAME, with_plan

# 로컬에서 `optiq serve`로 띄워 둔 9B 서버.
DEFAULT_MODEL = "openai/mlx-community/Qwen3.5-9B-OptiQ-4bit:think"
DEFAULT_API_BASE = "http://127.0.0.1:8080/v1"
# optiq serve는 'sk-optiq-' 접두사의 키를 요구한다. mlx_lm.server는 키를 검사하지 않는다.
LOCAL_API_KEY = "sk-optiq-local"

# Qwen3.5 thinking 모드 권장 샘플링 (temperature 1.0, top_p 0.95, top_k 20,
# presence_penalty 1.5). presence_penalty는 1.5에서 외국어 단어가 섞여 0.5로 낮췄다.
# mlx_lm.server는 generation_config.json을 읽지 않으므로 요청마다 넘긴다.
_MLX_EXTRA_BODY = {
    "top_k": 20,
    "presence_penalty": 0.5,
}

# max_tokens로 잘려 닫는 태그가 없는 <think> 블록도 끝까지 지운다.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.DOTALL)
_LABEL_RE =re.compile(r"^\s*[\[(]?[가-힣A-Za-z0-9 ]{1,12}[\])]?\s*:\s*")
_ACTION_RE = re.compile(r"\([^)]*\)|\*[^*]*\*")
# 서버가 파싱하지 못하고 텍스트로 남긴 종료 표시: `<end_call>`, `[END CALL]` 등.
_END_TAG_RE = re.compile(
    rf"</?{END_CALL_NAME}\s*/?>|\[\s*END[ _-]?CALL\s*\]", re.IGNORECASE
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
_QUOTES = "\"'“”‘’「」"


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    api_base: str | None = DEFAULT_API_BASE
    api_key: str | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    # thinking 토큰도 포함되는 한도라 곧 think 최대 길이다. 넘으면 대사가 비어 한 번 재시도한다.
    max_tokens: int = 2048
    # 통화 전 계획 생성의 한도(thinking 포함). 0이면 계획 단계를 건너뛴다.
    plan_max_tokens: int = 8192
    extra_body: dict | None = None

    def __post_init__(self):
        # 로컬 서버(api_base 지정)일 때만 자리표시 키와 MLX 전용 옵션을 쓴다.
        # 호스팅 모델은 api_key=None으로 두어 litellm이 환경변수 키를 쓰게 한다.
        if self.api_base:
            if self.api_key is None:
                self.api_key = LOCAL_API_KEY
            if self.extra_body is None:
                self.extra_body = dict(_MLX_EXTRA_BODY)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("api_key")
        return d


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    return text.split("</think>")[-1]


def clean_utterance(text: str) -> str:
    """모델 출력에서 thinking, tool call 원문, 화자 라벨, 행동 묘사, 감싼 따옴표를 걷어낸다."""
    text = _strip_think(text)
    text = _TOOL_CALL_RE.sub("", text)
    text = _LABEL_RE.sub("", text, count=1)  # `[김민수]: ` 라벨은 대괄호 제거보다 먼저
    text = _END_TAG_RE.sub("", text)
    text = _BRACKET_RE.sub("", text)
    text = _ACTION_RE.sub("", text)
    text = _NON_SPEECH_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().strip(_QUOTES).strip()


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
    messages: list[dict] = field(init=False)

    def __post_init__(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _complete(
        self, messages=None, max_tokens=None, tools="default"
    ) -> tuple[str, bool, dict]:
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
        meta = {
            "latency_s": round(time.perf_counter() - t0, 2),
            "finish_reason": resp.choices[0].finish_reason,
            "usage": resp.usage.model_dump() if resp.usage else None,
            "reasoning": getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None),
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in message.tool_calls or []
                if tc.function.name != END_CALL_NAME
            ],
        }
        content = message.content or ""
        # 서버가 tool_calls로 파싱하지 못하고 content에 <tool_call> 원문을 남기는 경우도 본다.
        # thinking 안에서 언급만 한 호출은 세지 않도록 thinking을 먼저 걷어낸다.
        ended = any(
            tc.function.name == END_CALL_NAME for tc in message.tool_calls or []
        ) or any(
            END_CALL_NAME in block
            for block in _TOOL_CALL_RE.findall(_strip_think(content))
        ) or bool(_END_TAG_RE.search(_strip_think(content)))
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
        plan_tools = [t for t in self.tools or [] if t["function"]["name"] != END_CALL_NAME]
        raw, _, meta, lookups = self._generate(
            messages=messages,
            max_tokens=self.llm.plan_max_tokens,
            tools=plan_tools,
            final_tools=[],
        )
        text = _strip_think(raw).strip()
        if not text or meta["finish_reason"] == "length":
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
    ) -> tuple[str, bool, dict, list[dict]]:
        """도구 호출을 처리하며 한 번 생성한다. 마지막 라운드는 final_tools만 열어 반드시 글로 답하게 한다.

        기본값은 통화 발화용: messages=self.messages, 모든 도구, 마지막 라운드는 end_call만.
        """
        messages = self.messages if messages is None else messages
        if tools is None:
            tools = self.tools
        if final_tools is None:
            final_tools = [t for t in self.tools or [] if t["function"]["name"] == END_CALL_NAME]
        has_lookup = any(t["function"]["name"] != END_CALL_NAME for t in tools or [])
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

    def respond(self, incoming: str) -> tuple[str, bool, dict]:
        self.messages.append({"role": "user", "content": incoming})
        text, ended, meta, lookups = "", False, {}, []
        for _ in range(2):  # 빈 응답이면 한 번 재시도 (바로 끊은 경우는 제외)
            raw, ended, meta, used = self._generate()
            lookups += used
            text = clean_utterance(raw)
            # 도구를 쓴 뒤 비었으면 다시 시도하지 않는다 (이미 늘어난 기록 위에서 호출이 불어난다).
            if text or ended or used:
                break
        if lookups:
            meta["wiki"] = lookups
        self.messages.append({"role": "assistant", "content": text})
        return text, ended, meta
