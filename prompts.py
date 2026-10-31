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


REVIEW_REQUEST = (
    "아래는 텔레마케터(발신)가 건 판매 전화 기록이다. 종료 사유: {end_reason}\n\n{transcript}\n\n"
    "첫 줄에 `점수: N`(1~5, 판매 시도의 자연스러움·정확성·예의), "
    "이어서 잘한 점, 문제점, 근거 없이 지어낸 정보를 3~5줄로 써라."
)

LEARN_REQUEST = (
    "너는 텔레마케터의 시스템 프롬프트와 위키를 개선한다. 아래는 최근 {n}통화의 리뷰이다.\n\n"
    "{reviews}\n\n[현재 system.md]\n{prompt}\n\n[위키 문서]\n{pages}\n\n"
    "반복되는 문제를 골라 고쳐라. 행동 원칙은 prompt_write로 system.md를 짧게(최대 {max_chars}자) 다시 쓰고, "
    "상황별 요령·FAQ는 위키 도구로 추가·수정하라. 위키 `상품/`은 고칠 수 없다. "
    "가격·조건·전화번호 같은 사실은 지어내지 마라. 끝나면 바꾼 내용을 한두 줄로 요약하라."
)
