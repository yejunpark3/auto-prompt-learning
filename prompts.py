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

REPORT_SPAM_NAME = "report_spam"
REPORT_SPAM_TOOL = {
    "type": "function",
    "function": {
        "name": REPORT_SPAM_NAME,
        "description": "스팸·원치 않는 전화로 신고하고 통화를 끊는다",
        "parameters": {"type": "object", "properties": {}},
    },
}
# 호출하면 통화가 끝나는 도구들.
HANGUP_TOOL_NAMES = (END_CALL_NAME, REPORT_SPAM_NAME)

RING_CUE = "[PHONE_CALLING_FROM_STRANGER]"
CALLEE_GREETING = "여보세요?"
WRAP_UP_HINT ="(통화가 길어지고 있습니다. 자연스럽게 마무리하세요.)"

PLAN_REQUEST = "통화를 걸기 전이다. [Goal]을 이루기 위한 통화 계획을 세워라. {wiki}대사는 쓰지 말고 계획만."
PLAN_WIKI_HINT = "필요하면 위키 도구로 관련 지식을 찾아 근거로 삼아라. "

SYSTEM_PROMPT_PATH =str(Path(__file__).parent / "prompt" / "system.md")

CHANNEL = "[Channel]\nKOREAN_VOICE_PHONE_CALL (한 번에 한두 문장, 대사만)"


def caller_prompt(profile: str, purpose: str, callee_info: str) -> str:
    return (
        f"[Persona]\n{profile}\n{CHANNEL}\n[Role]\nCALLER\n"
        f"[Callee]\n{callee_info}[Goal]\n{purpose}\n"
        "[Note]\n수신자의 이름이나 연락처처럼 알지 못하는 정보는 지어내지 않는다."
    )


def callee_prompt(profile: str) -> str:
    return f"[Persona]\n{profile}\n{CHANNEL}\n[Role]\nCALLEE"


def with_plan(system_prompt: str, plan: str) -> str:
    return f"{system_prompt}\n[Plan] (내부 메모. 대사로 읽지 말 것)\n{plan}"


def load_caller_profile(path: str = SYSTEM_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


SUBMIT_REVIEW_NAME = "submit_review"
SUBMIT_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_REVIEW_NAME,
        "description": "통화 평가를 제출한다",
        "parameters": {
            "type": "object",
            "properties": {
                "achieved": {
                    "type": "boolean",
                    "description": "상대가 통화 목적에 해당하는 요청에 명확히 응했는가. 검토·재연락 약속, 거절, 얼버무림은 false",
                },
                "notes": {
                    "type": "string",
                    "description": "그렇게 판단한 근거, 효과적이었던 말, 문제였던 말, 근거 없이 지어낸 정보를 3~5줄로",
                },
            },
            "required": ["achieved", "notes"],
        },
    },
}

REVIEW_REQUEST = (
    "아래는 발신자가 건 전화 기록이다. 통화 목적: {purpose}\n종료 사유: {end_reason}\n\n{transcript}\n\n"
    "이 통화를 평가해 submit_review 도구로 제출하라."
)

LEARN_REQUEST = (
    "너는 전화 발신자의 시스템 프롬프트와 위키를 개선한다. 아래는 최근 {n}통화의 리뷰이다(목적 달성률 {achieved_rate}, 스팸 신고율 {spam_rate}). 우선순위는 스팸 신고를 줄이는 것, 그다음 목적 달성률을 높이는 것이다.\n\n"
    "{reviews}\n\n[이전 학습 이력]\n{history}\n\n[현재 system.md]\n{prompt}\n\n[위키 문서]\n{pages}\n\n"
    "[고칠 수 없는 위키 문서]\n{readonly}\n\n"
    "이력의 각 변경 옆 '변경 후 묶음' 지표로 효과를 판단해, 나빠진 변경은 되돌리고 효과 있던 변경은 유지하며 같은 수정을 되풀이하지 마라(20통화 묶음은 변동이 커서 작은 차이는 우연일 수 있다). "
    "반복되는 문제를 골라 고쳐라. system.md에는 역할과 핵심 원칙만 두고(최대 {max_chars}자), prompt_edit로 현재 문장 중 1~2곳만 고친다. "
    "상황별 요령·금지 사항·지식은 위키 도구로 추가·수정하라. 고칠 수 없는 문서는 건드리지 마라. "
    "위키와 통화 목적에 없는 사실(숫자·조건·연락처 등)은 지어내지 마라. "
    "끝나면 바꾼 내용을 한두 줄로 요약하라."
)
