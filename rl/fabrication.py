"""발신자가 근거 없는 사실을 말했는가(지어내기). 보상의 바닥값과 학습 양성 선택에 쓴다.

두 검사의 합집합이다. 학습 보상에서는 놓치는 것이 오지적보다 나쁘다(정책이 검사기의 빈틈을 배운다).
- C1 결정적 가드(모델 없음): 발신자 발화의 숫자·URL·이메일이 근거에 없으면 지어낸 것.
  숫자의 단위나 표기는 보지 않는다(시나리오마다 다르다).
- C2 엄격 검사(모델): learner._fact_check_vote(combine="any")로 한 번이라도 지적한 문장. 검사가 끝내 실패하면 None.

근거는 읽기 전용 위키, 통화 목적, 계획을 뗀 발신자 프롬프트다(계획은 모델이 쓴 글이라 근거가 아니다).
상대가 통화 중 **먼저** 한 말도 근거다(발신자가 상대의 말을 되받는 것은 지어내기가 아니다). 발신자 말보다 뒤에 나온
상대의 말은 근거가 아니다: 지어낸 말을 상대가 되묻는 순간 정당화되면 그것이 보상 해킹 통로가 된다.
"""
import learner
from agent import LLMConfig
from prompts import STRICT_CLAIM_CHECK_REQUEST, without_plan

_LABEL = {"caller": "발신", "callee": "수신"}


def spoken(record: dict, role: str) -> list[str]:
    return [t["text"] for t in record["turns"] if t["speaker"] == role and t["turn"] > 0 and t.get("text")]


def base_source(record: dict, docs: list[str] | None = None) -> str:
    """통화와 무관한 근거: 계획을 뗀 발신자 프롬프트, 통화 목적, 읽기 전용 문서(docs=None이면 위키, 주면 그 문서들만)."""
    return learner._fact_source([record], without_plan(record["caller"]["system_prompt"]), docs)


def _callee_before(record: dict, turn: int) -> str:
    return "\n".join(t["text"] for t in record["turns"]
                     if t["speaker"] == "callee" and 0 < t["turn"] < turn and t.get("text"))


def source_text(record: dict, docs: list[str] | None = None) -> str:
    """C2가 보는 근거: base_source에 통화 기록(순서대로)을 붙인다. 순서 규칙은 STRICT_CLAIM_CHECK_REQUEST가 설명한다."""
    lines = [f"[{_LABEL[t['speaker']]}] {t['text']}" for t in record["turns"] if t.get("text")]
    return f"{base_source(record, docs)}\n\n[통화 기록]\n" + "\n".join(lines)


def ungrounded(record: dict, docs: list[str] | None = None) -> tuple[list[str], list[str]]:
    """C1: (근거 없는 숫자, 근거 없는 URL·이메일). 발화마다 그 앞에 나온 상대의 말까지만 근거로 본다."""
    base = base_source(record, docs)
    allowed = learner._numbers(base, strict=True)
    numbers: set[str] = set()
    contacts: set[str] = set()
    for t in record["turns"]:
        if t["speaker"] != "caller" or t["turn"] == 0 or not t.get("text"):
            continue
        before = _callee_before(record, t["turn"])
        numbers |= learner._numbers(t["text"], strict=False) - allowed - learner._numbers(before, strict=False)
        contacts |= {c for c in learner.CONTACT_RE.findall(t["text"]) if c not in base and c not in before}
    return sorted(numbers), sorted(contacts)


def strict_claims(record: dict, llm: LLMConfig, tries: int = 2, docs: list[str] | None = None) -> list[str] | None:
    said = spoken(record, "caller")
    if not said:
        return []
    return learner._fact_check_vote(
        "\n".join(said), source_text(record, docs), llm, decoys=learner._decoys([record], docs=docs),
        runs=2, tries=tries, lenient=True, combine="any", request=STRICT_CLAIM_CHECK_REQUEST,
    )


def detect(record: dict, llm: LLMConfig | None = None, tries: int = 2, docs: list[str] | None = None) -> dict:
    """{"numbers": [...], "contacts": [...], "claims": [...] | None, "fabricated": True | False | None}.
    fabricated=None은 C1이 비었고 C2를 못 돌린(llm 없음·검사 실패) 경우로, 보상에서는 제외한다."""
    numbers, contacts = ungrounded(record, docs)
    # C1이 이미 지어냄을 잡았으면 비싼 C2는 돌리지 않는다(결과가 같다). C2 단독 성능은 벤치가 strict_claims로 잰다.
    claims = None if (numbers or contacts or llm is None) else strict_claims(record, llm, tries, docs)
    if numbers or contacts or claims:
        fabricated: bool | None = True
    else:
        fabricated = None if claims is None else False
    return {"numbers": numbers, "contacts": contacts, "claims": claims, "fabricated": fabricated}
