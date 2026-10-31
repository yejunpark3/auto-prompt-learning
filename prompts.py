END_CALL_NAME = "end_call"
END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": END_CALL_NAME,
        "description": "전화 통화를 종료한다",
        "parameters": {"type": "object", "properties": {}},
    },
}

RING_CUE = "(전화벨이 울립니다. 발신자 표시: 모르는 번호) 전화를 받아 첫 마디만 하세요."
WRAP_UP_HINT = "(통화가 길어지고 있습니다. 자연스럽게 마무리하세요.)"

def caller_prompt(profile: str, purpose: str) -> str:
    return f"[Persona]\n{profile}\n[Goal]\n{purpose}"


def callee_prompt(profile: str) -> str:
    return f"[Persona]\n{profile}"
