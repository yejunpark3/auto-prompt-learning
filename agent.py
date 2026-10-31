import re
import time
from dataclasses import asdict, dataclass, field

import litellm

# 로컬에서 `optiq serve`로 띄워 둔 9B 서버. ':no-think'는 thinking을 끈 변형이다.
DEFAULT_MODEL = "openai/mlx-community/Qwen3.5-9B-OptiQ-4bit:no-think"
DEFAULT_API_BASE = "http://127.0.0.1:8080/v1"
# optiq serve는 'sk-optiq-' 접두사의 키를 요구한다. mlx_lm.server는 키를 검사하지 않는다.
LOCAL_API_KEY = "sk-optiq-local"

# Qwen3.5 generation_config.json 권장 샘플링 (temperature 0.7, top_p 0.8, top_k 20,
# presence_penalty 1.5). mlx_lm.server는 이 파일을 읽지 않으므로 요청마다 넘긴다.
_MLX_EXTRA_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "top_k": 20,
    "presence_penalty": 1.5,
}

# max_tokens로 잘려 닫는 태그가 없는 <think> 블록도 끝까지 지운다.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_LABEL_RE = re.compile(r"^\s*[\[(]?[가-힣A-Za-z0-9 ]{1,12}[\])]?\s*:\s*")
_ACTION_RE = re.compile(r"\([^)]*\)|\*[^*]*\*")
_QUOTES = "\"'“”‘’「」"


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    api_base: str | None = DEFAULT_API_BASE
    api_key: str | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 256
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


def clean_utterance(text: str, hangup_token: str | None = None) -> str:
    """모델 출력에서 thinking, 화자 라벨, 행동 묘사, 감싼 따옴표를 걷어낸다."""
    text = _THINK_RE.sub("", text)
    text = text.split("</think>")[-1]
    token_found = bool(hangup_token and hangup_token in text)
    if hangup_token:
        text = text.replace(hangup_token, "")
    text = _LABEL_RE.sub("", text, count=1)
    text = _ACTION_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().strip(_QUOTES).strip()
    return text + (f" {hangup_token}" if token_found else "")


@dataclass
class PhoneAgent:
    role: str  # "caller" | "callee"
    persona: dict
    name: str
    system_prompt: str
    llm: LLMConfig
    hangup_token: str | None = None
    messages: list[dict] = field(init=False)

    def __post_init__(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _complete(self) -> tuple[str, dict]:
        kwargs = dict(
            model=self.llm.model,
            messages=self.messages,
            api_base=self.llm.api_base,
            api_key=self.llm.api_key,
            temperature=self.llm.temperature,
            top_p=self.llm.top_p,
            max_tokens=self.llm.max_tokens,
        )
        if self.llm.extra_body:
            kwargs["extra_body"] = self.llm.extra_body
        t0 = time.perf_counter()
        resp = litellm.completion(**kwargs)
        meta = {
            "latency_s": round(time.perf_counter() - t0, 2),
            "usage": resp.usage.model_dump() if resp.usage else None,
        }
        return resp.choices[0].message.content or "", meta

    def respond(self, incoming: str) -> tuple[str, dict]:
        self.messages.append({"role": "user", "content": incoming})
        text, meta = "", {}
        for _ in range(2):  # 빈 응답이면 한 번 재시도
            raw, meta = self._complete()
            text = clean_utterance(raw, self.hangup_token)
            if text:
                break
        self.messages.append({"role": "assistant", "content": text})
        return text, meta
