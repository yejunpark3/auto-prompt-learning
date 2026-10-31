from agent import LLMConfig, PhoneAgent, _strip_think
from prompts import REVIEW_REQUEST

SPAM_END_REASON = "spam_reported_by_callee"


def _transcript(record: dict) -> str:
    label = {"caller": "발신", "callee": "수신"}
    return "\n".join(f"[{label[t['speaker']]}] {t['text']}" for t in record["turns"] if t["text"])


def _parse_achieved(notes: str) -> bool | None:
    first = notes.split("\n", 1)[0]
    if "아니오" in first:
        return False
    return True if "예" in first else None


def review_call(record: dict, llm: LLMConfig, attempts: int = 2) -> dict | None:
    """통화 한 건을 평가한다: {spam, achieved, notes, latency_s}.

    spam은 end_reason으로 정해지고, achieved는 모델이 첫 줄 `달성: 예|아니오`로 답한다.
    신고된 통화는 모델 응답과 무관하게 항상 (spam=True, achieved=False)로 돌려준다(학습에서 빠지면 안 된다).
    그 외 통화는 형식이 틀리면 `attempts`번까지 다시 묻고, 끝내 못 읽거나 비었거나 max_tokens에서
    잘리면 None (대기열에 넣지 않는다).
    """
    spam = record["end_reason"] == SPAM_END_REASON
    agent = PhoneAgent("reviewer", None, "리뷰어", "통화 기록을 평가한다.", llm)
    request = REVIEW_REQUEST.format(end_reason=record["end_reason"], transcript=_transcript(record),
                                    purpose=record.get("purpose", ""))
    notes, latency, achieved = "", 0.0, None
    for _ in range(attempts):
        raw, _, meta = agent._complete(
            messages=[{"role": "system", "content": agent.system_prompt},
                      {"role": "user", "content": request}],
            max_tokens=llm.plan_max_tokens,
            tools=[],
        )
        latency += meta["latency_s"]
        notes = "" if meta["finish_reason"] == "length" else _strip_think(raw).strip()
        achieved = _parse_achieved(notes) if notes else None
        if achieved is not None or spam:
            break
    if spam:
        return {"spam": True, "achieved": False,
                "notes": notes or "(수신자가 스팸으로 신고하고 끊음. 리뷰 생성 실패)", "latency_s": latency}
    if achieved is None:
        return None
    return {"spam": False, "achieved": achieved, "notes": notes, "latency_s": latency}
