END_CALL_NAME = "end_call"
END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": END_CALL_NAME,
        "description": "전화 통화를 종료한다",
        "parameters": {"type": "object", "properties": {}},
    },
}

RING_CUE = "[PHONE_CALLING_FROM_UNKNOWN_NUMBER]"
WRAP_UP_HINT = "(통화가 길어지고 있습니다. 자연스럽게 마무리하세요.)"

CHANNEL = "[Channel]\nKOREAN_VOICE_PHONE_CALL (한 번에 한두 문장, 대사만)"


def caller_prompt(profile: str, purpose: str) -> str:
    return f"[Persona]\n{profile}\n{CHANNEL}\n[Role]\nCALLER\n[Goal]\n{purpose}"


def callee_prompt(profile: str) -> str:
    return f"[Persona]\n{profile}\n{CHANNEL}\n[Role]\nCALLEE"
