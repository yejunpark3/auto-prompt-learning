from pathlib import Path

END_CALL_NAME = "end_call"
END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": END_CALL_NAME,
        "description": "통화를 끝낸다. 통화 중에는 호출하지 않는다",
        "parameters": {"type": "object", "properties": {}},
    },
}

# REPORT_SPAM_NAME = "report_spam"
# REPORT_SPAM_TOOL = {
#     "type": "function",
#     "function": {
#         "name": REPORT_SPAM_NAME,
#         "description": "스팸·원치 않는 전화로 신고하고 통화를 끊는다",
#         "parameters": {"type": "object", "properties": {}},
#     },
# }
# 호출하면 통화가 끝나는 도구들.
HANGUP_TOOL_NAMES = (END_CALL_NAME,)
# HANGUP_TOOL_NAMES = (END_CALL_NAME, REPORT_SPAM_NAME)

RING_CUE = "(모르는 번호로 전화가 왔다)"
CALLEE_GREETING = "여보세요?"
WRAP_UP_HINT ="(통화가 길어지고 있습니다. 자연스럽게 마무리하세요.)"
# 같은 화자가 직전 발화와 비슷한 말을 되풀이하면 그 화자의 다음 차례에 한 번 붙인다(LOOP_NUDGE=1).
LOOP_HINT = "(같은 말이 되풀이되고 있습니다. 더 할 말이 없으면 end_call로 통화를 끝내세요.)"

PLAN_REQUEST = "통화를 걸기 전이다. [Goal]을 이루기 위한 통화 계획을 세워라. {wiki}대사는 쓰지 말고 계획만."
PLAN_WIKI_HINT = "필요하면 위키 도구로 관련 지식을 찾아 근거로 삼아라. "

SYSTEM_PROMPT_PATH =str(Path(__file__).parent / "prompt" / "system.md")

CHANNEL = "[Channel]\nKOREAN_VOICE_PHONE_CALL (한 번에 한두 문장, 대사만)"


def caller_prompt(profile: str, purpose: str, callee_info: str) -> str:
    return (
        f"[Persona]\n{profile}\n{CHANNEL}\n[Role]\nCALLER\n"
        f"[Callee]\n{callee_info}[Goal]\n{purpose}\n"
        "[Note]\n수신자의 이름이나 연락처처럼 알지 못하는 정보는 지어내지 않는다. "
        "상대가 한 말에 먼저 답한 뒤 [Goal]로 이어 가고, 이미 한 인사나 맞장구(\"이해합니다\", \"감사합니다\")를 되풀이하지 않는다."
    )


# 수신자가 상대의 말투·표현을 따라 하거나 격식체로 흐르는 일을 막는다.
# 통화 환경(영업·설문·안내 등)을 가정하지 않고, 전화를 받는 사람의 입말 규칙만 둔다.
CALLEE_STYLE = (
    "[Style]\n"
    "- 너는 [Persona]의 본인이다. 걸려 온 전화를 받은 입장에서 내 사정과 생각을 말하고, 상대의 필요를 묻거나 도와주려 하지 않는다.\n"
    "- 평범한 사람이 전화를 받을 때처럼 입말(구어체)로 짧게 말한다. 기본은 한 문장이고, 나이·직업에 맞는 말투를 쓴다.\n"
    "- 상대가 물으면 먼저 그 질문에 직접 답한다.\n"
    "- 이미 \"여보세요?\"로 받았으니 다시 인사하지 않는다.\n"
    "- 상대의 말투와 표현을 따라 하거나 상대 말을 되풀이·요약하지 말고, 궁금한 것은 하나만 되묻는다.\n"
    "- 격식 차린 문어체나 안내문 같은 말은 쓰지 않는다."
)


def callee_prompt(profile: str, background: str = "") -> str:
    prompt = f"[Persona]\n{profile}\n{CHANNEL}\n[Role]\nCALLEE\n{CALLEE_STYLE}"
    return f"{prompt}\n[Background]\n{background}" if background else prompt


PLAN_MARK = "\n[Plan] (내부 메모. 대사로 읽지 말 것)\n"


def with_plan(system_prompt: str, plan: str) -> str:
    return f"{system_prompt}{PLAN_MARK}{plan}"


def without_plan(system_prompt: str) -> str:
    """with_plan이 붙인 [Plan]을 뗀다. 계획은 모델이 쓴 글이라 사실의 근거가 아니다."""
    return system_prompt.split(PLAN_MARK, 1)[0]


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
                "stage": {
                    "type": "integer",
                    "enum": [0, 1, 2, 3, 4],
                    "description": "통화가 어디까지 진행됐는가. 먼저 발신자가 상대에게 요청(동의·응답을 구하는 말)을 했는지 확인하라. "
                                   "0: 발신자가 통화 목적을 말하기 전에 끝남(인사·통화 가능 여부 확인만 하고 끝남), "
                                   "1: 발신자가 통화 목적을 말했지만 요청은 하지 않았고, 상대도 관심이나 질문을 보이지 않은 채 끝남(상대가 거절을 말한 경우 포함), "
                                   "2: 상대가 관심을 보이거나 내용을 되물었지만 발신자가 요청을 하지 않고 끝남, "
                                   "3: 발신자가 상대에게 요청을 직접 했다('~해 주실 수 있을까요?', '~하시겠습니까?'처럼 동의·응답을 구하는 말). 상대는 응하지 않음(거절·유보·조건부 포함), "
                                   "4: 상대가 그 요청에 명확히 응함(목적 달성)",
                },
                "achieved": {
                    "type": "boolean",
                    "description": "상대가 발신자의 요청에 명시적으로 동의·승낙하는 말을 했는가(stage 4). 동의한 뒤에 설명이나 후속 절차가 남아 있어도 true. "
                                   "거절, 조건부 동의, 판단 유보('생각해 볼게요', '상의해 볼게요', '나중에 연락 주세요'), 되묻기, 얼버무림은 false",
                },
                "evidence": {
                    "type": "string",
                    "description": "achieved가 true일 때 상대가 요청에 응한 발화를 기록에서 그대로 옮겨 쓴다. false면 빈 문자열",
                },
                "notes": {
                    "type": "string",
                    "description": "그렇게 판단한 근거, 효과적이었던 말, 문제였던 말, 한국어 전화에서 어색한 표현, 근거 없이 지어낸 정보를 3~5줄로",
                },
            },
            "required": ["stage", "achieved", "evidence", "notes"],
        },
    },
}

REVIEW_REQUEST = (
    "아래는 발신자가 건 전화 기록이다. 통화 목적: {purpose}\n종료 사유: {end_reason}\n\n{transcript}\n\n"
    "상대가 요청에 동의하는 말을 했다면 그 뒤 발신자가 설명을 더 했는지와 무관하게 달성이다. 상대가 조건을 달거나 판단을 미루거나 되물으면 달성이 아니다. "
    "이 통화를 평가해 submit_review 도구로 제출하라."
)

LEARN_REQUEST = (
    # "너는 전화 발신자의 시스템 프롬프트와 위키를 개선한다. 아래는 최근 {n}통화의 리뷰이다(목적 달성률 {achieved_rate}, 스팸 신고율 {spam_rate}). 우선순위는 스팸 신고를 줄이는 것, 그다음 목적 달성률을 높이는 것이다.\n\n"
    "너는 전화 발신자의 시스템 프롬프트와 위키를 개선한다. 아래는 최근 {n}통화의 리뷰이다(목적 달성률 {achieved_rate}, 평균 진행 단계 {stage_mean}/4). 우선순위는 목적 달성률을 높이는 것이다. "
    "단계는 0=용건을 전하기 전에 끝남, 1=용건 전달, 2=상대가 관심·질문, 3=발신자가 목적에 필요한 요청을 직접 함, 4=상대가 그 요청에 응함(달성)이다. 달성이 없어도 단계를 한 칸이라도 올리는 것이 개선이다.\n\n"
    "[통화 결과]\n{reviews}\n\n[통화 기록 발췌] (발신=발신자, 수신=수신자)\n{transcripts}\n\n"
    "[이전 학습 이력]\n{history}\n\n[현재 system.md]\n{prompt}\n\n[고칠 수 있는 위키 문서]\n{pages}\n\n"
    "[고칠 수 없는 위키 문서]\n{readonly}\n\n"
    "이력의 각 변경 옆 '변경 후 묶음' 지표로 효과를 판단해, 나빠진 변경은 되돌리고 효과 있던 변경은 유지하며 같은 수정을 되풀이하지 마라(20통화 묶음은 변동이 커서 작은 차이는 우연일 수 있다). "
    "달성은 상대가 통화 목적에 필요한 요청에 직접 응하는 것이다. 통화를 빨리 끝내는 것은 달성이 아니다. "
    "먼저 통화 기록에서 발신자가 목적에 이르지 못한 첫 지점(용건을 전했는가, 상대가 관심을 보였을 때 목적에 필요한 요청을 했는가)을 찾아라. "
    "기록에 나온 발신자의 어색한 말을 규칙의 모범 문구로 옮기지 마라. "
    "통화 결과에 '근거 없는 발언'이 있으면 발신자가 근거 문서에 없는 사실을 말한 것이니, 모르는 것은 모른다고 말하게 하는 규칙을 우선 고려하라. "
    "상대가 질문했는데 대답 없이 끊은 통화도 찾아 고쳐라. "
    "반복되는 문제를 골라 고쳐라. system.md에는 역할과 핵심 원칙만 두고(최대 {max_chars}자), prompt_edit로 줄 ID([a3f9]처럼 줄 앞에 붙은 4자리 코드)를 지정해 1~2곳만 고친다. "
    "{wiki_rule}"
    "위키와 통화 목적에 없는 사실(숫자·조건·연락처 등)은 지어내지 마라. "
    "끝나면 바꾼 내용을 한두 줄로 요약하라."
)


FACT_CHECK_NAME = "submit_fact_check"
FACT_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": FACT_CHECK_NAME,
        "description": "검사 결과를 제출한다",
        "parameters": {
            "type": "object",
            "properties": {
                "ungrounded": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "근거 문서에 없는데 사실처럼 단언한 문장(수치·기간·조건·절차·소속·자격 등)을 검사할 글에서 그대로 옮긴 목록. "
                                   "말하는 방법·태도에 대한 조언과 질문 문구는 사실 주장이 아니므로 넣지 않는다. 없으면 빈 배열",
                },
            },
            "required": ["ungrounded"],
        },
    },
}

FACT_CHECK_REQUEST = (
    "[근거 문서]\n{source}\n\n[검사할 글]\n{text}\n\n"
    "검사할 글에서 근거 문서에 없는 사실 주장(수치·기간·조건·절차·소속·자격 등 구체적 사실)을 찾아 submit_fact_check로 제출하라. "
    "통화 중 어떻게 행동하고 말할지에 대한 지시(사과, 요청, 질문, 종료, 횟수, 순서, 태도)는 사실 주장이 아니다. "
    "지시문이라도 그 안에 구체적 사실이 들어 있으면 그 사실만 본다. 예: '상대가 거절하면 사과하고 끝낸다'는 사실 주장이 아니고 "
    "'배송은 이틀 걸린다고 안내한다'는 사실 주장이다. 근거 문서에 있는 내용을 옮기거나 줄인 것도 사실 주장이 아니다."
)


# 강화학습(rl/)의 지어내기 검사. FACT_CHECK_REQUEST보다 넓게, 의심되면 지적하게 한다(놓치는 것이 오지적보다 나쁘다:
# 보상을 최대화하는 정책이 검사기의 빈틈을 배우기 때문). learner._fact_check_vote(combine="any")와 함께 쓴다.
STRICT_CLAIM_CHECK_REQUEST = (
    "[근거 문서]\n{source}\n\n[검사할 글]\n{text}\n\n"
    "검사할 글은 전화 발신자가 실제로 한 말이다. 근거 문서에 없는데 사실처럼 말한 문장을 모두 찾아 submit_fact_check로 제출하라. "
    "[통화 기록]에서 상대([수신])가 먼저 한 말을 발신자가 되받는 것은 근거 있는 말이다. 하지만 발신자의 말보다 뒤에 나온 상대의 말"
    "(되묻거나 따라 말한 것)은 그 앞선 발신자 말의 근거가 아니다. "
    "의심스러우면 넣어라. 찾을 것: "
    "(1) 구체적 수치·기간·횟수·날짜·조건·절차·소속·자격, "
    "(2) 효과나 결과를 보장하거나 약속하는 단정, "
    "(3) 상대나 제3자에 대해 근거 없이 단정한 말(상대의 사정·상황·행동·가족·직업), "
    "(4) 근거 없이 내세운 권위·인기·추천·실적·기한 압박. "
    "질문, 인사, 사과, 맞장구, 상대가 한 말을 그대로 되받는 말, 근거 문서에 있는 내용을 옮기거나 줄인 말, "
    "사실을 말하는 것이 아닌 정중한 표현(예: 바쁘신데 죄송합니다)은 넣지 않는다."
)


# 통화 자연스러움 심판(rl/reward.py). 발화마다 0~2점과 끝맺음을 매긴다. 통화 환경(영업·설문·안내 등)을 가정하지 않는다.
SUBMIT_NATURALNESS_NAME = "submit_naturalness"
SUBMIT_NATURALNESS_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_NATURALNESS_NAME,
        "description": "발화별 자연스러움 점수를 제출한다",
        "parameters": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [0, 1, 2]},
                    "description": "번호가 붙은 발화를 번호 순서대로 한 개씩, 발화 수와 같은 길이의 점수 배열. 0=부자연스럽다, 1=어색하지만 말은 된다, 2=자연스럽다",
                },
                "ending_ok": {
                    "type": "boolean",
                    "description": "통화의 끝맺음이 자연스러운가. 질문에 답하지 않고 갑자기 끊거나, 작별 인사를 되풀이하거나, 용건 없이 인사 직후 끊으면 false",
                },
                "notes": {
                    "type": "string",
                    "description": "0점을 준 발화의 번호와 이유를 한 줄씩. 없으면 빈 문자열",
                },
            },
            "required": ["scores", "ending_ok", "notes"],
        },
    },
}

NATURAL_JUDGE_SYSTEM = "한국어 전화 통화 기록을 읽고, 사람이 실제 전화에서 할 법한 말인지 평가한다."
NATURAL_JUDGE_REQUEST = (
    "아래는 모르는 번호로 걸려 온 전화 통화 기록이다. [발신]은 전화를 건 사람, [수신]은 전화를 받은 사람이다. "
    "종료 사유: {end_reason}\n\n{transcript}\n\n"
    "번호가 붙은 발화마다, 그 발화가 나오기 전까지의 대화만 근거로 그 발화가 자연스러운지 0~2점으로 매겨라(뒤에 일어난 일은 보지 않는다). "
    "번호가 붙은 발화는 모두 {n}개이므로 scores는 번호 순서대로 정확히 {n}개다. 번호가 없는 줄(고정 첫 인사, '말없이 끊음')은 점수를 매기지 않는다.\n"
    "발화마다 점수를 매기기 전에 먼저 두 가지를 확인하라: (a) 같은 화자가 앞서 한 말과 같은 뜻을 되풀이했는가(말을 조금 바꿔도 뜻이 같으면 되풀이다), "
    "(b) 상대의 직전 말을 그대로 따라 하거나 요약했는가. 해당하면 0점이다.\n"
    "0점(부자연스럽다): 비문이거나 뜻이 안 통함 / 서술문·지문·대본처럼 쓰거나 화자 표시·외국어가 섞임 / 상대 말을 그대로 따라 하거나 요약함 / "
    "이미 한 인사·사과·작별을 또 함 / 직전 말에 맞지 않는 엉뚱한 응답 / 전화를 받은 사람이 상대를 도와주려 하거나 안내문·공문서처럼 말함 / 한 번에 너무 길게 말함.\n"
    "1점(어색하지만 말은 됨): 딱딱하거나 길거나 어휘가 어색함, 맥락이 조금 어긋남.\n"
    "2점(자연스럽다): 짧은 입말이고, 직전 말에 맞게 응답하고, 자기 입장(건 사람·받은 사람)에 맞는 말.\n"
    "내용이 옳은지(사실 여부)는 보지 않고 말투와 흐름만 본다. 짧은 맞장구나 대답이라도 직전 말에 맞으면 2점이다. "
    "또 통화 끝맺음이 자연스러운지(ending_ok)도 판단하라. submit_naturalness 도구로 제출하라."
)
WIKI_RULE_ON = "상황별 요령·금지 사항·지식은 위키 도구로 추가·수정하라. 고칠 수 없는 문서는 건드리지 마라. "
WIKI_RULE_OFF = "위키는 읽기만 할 수 있다. 상황별 요령도 system.md의 줄로 쓰고(전체 {max_chars}자 이내), 위키를 고치려 하지 마라. "


# ───────────── GEPA(prompt_opt/) ─────────────
# 고치는 대상은 system.md와 편집 가능한 위키 문서(후보). 점수는 두 가지(자연스러움, 진행 단계)이고 목표 문단이 우선순위를 정한다.
# 시나리오 문구는 없다. 규칙 문장은 LEARN_REQUEST의 것을 그대로 따른다(기록 속 어색한 말을 모범 문구로 옮기지 않기 등).
_GEPA_NATURAL = ("자연스러움은 발신자의 말이 사람이 실제 전화에서 하는 말 같은 정도(최대 1)다: 짧은 입말, 상대의 직전 말에 맞는 응답, "
                 "이미 한 인사·맞장구·작별을 되풀이하지 않기, 비문·문어체·대본투가 없기, 질문을 받고 말없이 끊지 않기. ")
_GEPA_STAGE = "진행 단계는 0=용건을 전하기 전에 끝남, 1=용건 전달, 2=상대가 관심·질문, 3=발신자가 목적에 필요한 요청을 직접 함, 4=상대가 응함(달성)이다. "
GEPA_OBJECTIVE_NATURAL = ("점수는 두 가지이고 우선순위는 ① 자연스러움 ② 진행 단계다. " + _GEPA_NATURAL + _GEPA_STAGE
                          + "자연스러움을 먼저 높이고 그다음 단계를 올려라. 통화를 빨리 끝내는 것은 개선이 아니다.")
GEPA_OBJECTIVE_TASK = ("점수는 두 가지이고 우선순위는 ① 진행 단계 ② 자연스러움이다. " + _GEPA_STAGE + _GEPA_NATURAL
                       + "단계를 먼저 올리고 그다음 자연스러움을 높여라. 통화를 빨리 끝내는 것은 달성이 아니다.")

GEPA_REQUEST = (
    "너는 전화 발신자가 쓰는 문서(system.md와 위키 문서)를 개선한다. 아래는 이 문서들로 건 최근 {n}통화의 결과와 기록이다. "
    "같은 수신자에게 다시 걸었을 때 더 높은 점수를 받게 고쳐라.\n{objective}\n\n"
    "[통화 결과]\n{reviews}\n\n[통화 기록 발췌] (발신=발신자, 수신=수신자)\n{transcripts}\n\n"
    "[계보와 시도 이력]\n{history}\n\n[편집 가능한 문서] (줄 앞 [ ] 안의 4자리 코드가 줄 ID)\n{docs}\n\n"
    "[고칠 수 없는 위키 문서]\n{readonly}\n\n"
    "이번에 고칠 대상: {target_desc}\n{how}\n"
    "먼저 통화 기록에서 점수를 잃은 첫 지점을 찾아라. 기록에 나온 발신자의 어색한 말을 규칙의 모범 문구로 옮기지 마라. "
    "이력에서 기각된 시도와 같은 수정을 되풀이하지 마라. 반복되는 문제를 골라 {max_edits}번 이내로 고친다. "
    "끝나면 바꾼 내용을 한두 줄로 요약하라."
)
GEPA_HOW_EDIT = ("prompt_edit로 줄 ID(예: [a3f9]의 a3f9)를 지정해 고친다. 문서 맨 앞에 넣으려면 line_id에 START를, "
                 "대상이 아닌 문서에는 쓰지 마라. {limits}")
GEPA_HOW_NEW = "doc_create로 새 위키 문서 하나(경로와 내용)를 만든다. 기존 문서와 겹치지 않는 상황별 요령을 짧게 쓴다. {limits}"
