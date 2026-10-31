import json
import re

from agent import LLMConfig, PhoneAgent, _strip_think
from prompts import REVIEW_REQUEST, SUBMIT_REVIEW_NAME, SUBMIT_REVIEW_TOOL

SPAM_END_REASON = "spam_reported_by_callee"


def _transcript(record: dict) -> str:
    label = {"caller": "발신", "callee": "수신"}
    return "\n".join(f"[{label[t['speaker']]}] {t['text']}" for t in record["turns"] if t["text"])


_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _content_calls(content: str) -> list[dict]:
    """서버가 tool_calls로 파싱하지 못하고 content에 남긴 `<tool_call>` 블록을 호출 목록으로 바꾼다.

    JSON(`{"name": ..., "arguments": {...}}`)과 Qwen XML(`<function=이름><parameter=키>값</parameter>`)을 읽는다.
    """
    calls = []
    for block in _TOOL_BLOCK_RE.findall(_strip_think(content)):
        try:
            d = json.loads(block)
            if isinstance(d, dict) and isinstance(d.get("name"), str):
                args = d.get("arguments", {})
                calls.append({"name": d["name"], "arguments": args if isinstance(args, str) else json.dumps(args)})
                continue
        except ValueError:
            pass
        if m := re.search(r"<function=(\w+)>", block):
            args = {k: v for k, v in _PARAM_RE.findall(block)}
            if "achieved" in args:  # XML 값은 문자열이라 true/false만 boolean으로 바꾼다
                args["achieved"] = {"true": True, "false": False}.get(args["achieved"].strip().lower(), args["achieved"])
            calls.append({"name": m.group(1), "arguments": json.dumps(args, ensure_ascii=False)})
    return calls


def _parse_review(calls: list[dict]) -> tuple[bool, str] | None:
    """submit_review 호출의 인자에서 (achieved, notes)를 읽는다. 타입이 맞지 않으면 None."""
    for c in calls:
        if c["name"] != SUBMIT_REVIEW_NAME:
            continue
        try:
            a = json.loads(c["arguments"] or "{}")
        except ValueError:
            continue
        if not isinstance(a, dict):
            continue
        achieved, notes = a.get("achieved"), a.get("notes")
        if isinstance(achieved, bool) and isinstance(notes, str) and notes.strip():
            return achieved, notes.strip()
    return None


def review_call(record: dict, llm: LLMConfig, attempts: int = 2) -> dict | None:
    """통화 한 건을 평가한다: {spam, achieved, notes, latency_s}.

    spam은 end_reason으로 정해지고, achieved·notes는 모델이 submit_review 도구로 제출한다(achieved는 boolean).
    신고된 통화는 모델 응답과 무관하게 항상 (spam=True, achieved=False)로 돌려준다(학습에서 빠지면 안 된다).
    그 외 통화는 도구를 부르지 않았거나 인자가 틀리면 `attempts`번까지 다시 묻고, 끝내 못 읽거나
    max_tokens에서 잘리면 None (대기열에 넣지 않는다).
    """
    spam = record["end_reason"] == SPAM_END_REASON
    agent = PhoneAgent("reviewer", None, "리뷰어", "통화 기록을 평가한다.", llm)
    request = REVIEW_REQUEST.format(end_reason=record["end_reason"], transcript=_transcript(record),
                                    purpose=record.get("purpose", ""))
    parsed, latency = None, 0.0
    for _ in range(attempts):
        content, _, meta = agent._complete(
            messages=[{"role": "system", "content": agent.system_prompt},
                      {"role": "user", "content": request}],
            max_tokens=llm.plan_max_tokens,
            tools=[SUBMIT_REVIEW_TOOL],
        )
        latency += meta["latency_s"]
        parsed = None if meta["finish_reason"] == "length" else _parse_review([*meta["tool_calls"], *_content_calls(content)])
        if parsed or spam:
            break
    if spam:
        return {"spam": True, "achieved": False,
                "notes": parsed[1] if parsed else "(수신자가 스팸으로 신고하고 끊음. 리뷰 생성 실패)",
                "latency_s": latency}
    if parsed is None:
        return None
    return {"spam": False, "achieved": parsed[0], "notes": parsed[1], "latency_s": latency}
