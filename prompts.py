from pathlib import Path

END_CALL_NAME = "end_call"
END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": END_CALL_NAME,
        "description": "전화 통화를 종료한다",
        "parameters": {"type": "object", "properties": {}},
    },
}

RING_CUE = "[PHONE_CALLING_FROM_STRANGER]"
CALLEE_GREETING = "여보세요?"
WRAP_UP_HINT ="(통화가 길어지고 있습니다. 자연스럽게 마무리하세요.)"

PLAN_REQUEST = "통화를 걸기 전이다. [Goal]을 이루기 위한 통화 계획을 세워라. {wiki}대사는 쓰지 말고 계획만."
PLAN_WIKI_HINT = "필요하면 위키 도구로 상품·응대 지식을 찾아 근거로 삼아라. "

SYSTEM_PROMPT_PATH =str(Path(__file__).parent / "prompt" / "system.md")

CHANNEL = "[Channel]\nKOREAN_VOICE_PHONE_CALL (한 번에 한두 문장, 대사만)"


def caller_prompt(profile: str, purpose: str) -> str:
    return f"[Persona]\n{profile}\n{CHANNEL}\n[Role]\nCALLER\n[Goal]\n{purpose}"


def callee_prompt(profile: str) -> str:
    return f"[Persona]\n{profile}\n{CHANNEL}\n[Role]\nCALLEE"


def with_plan(system_prompt: str, plan: str) -> str:
    return f"{system_prompt}\n[Plan] (내부 메모. 대사로 읽지 말 것)\n{plan}"


def load_caller_profile(path: str = SYSTEM_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").strip()
