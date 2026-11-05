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
            if args.get("stage", "").strip().isdigit():
                args["stage"] = int(args["stage"].strip())
            calls.append({"name": m.group(1), "arguments": json.dumps(args, ensure_ascii=False)})
    return calls


def tool_calls_of(content: str, meta: dict) -> list[dict]:
    """응답에서 도구 호출을 모은다: 서버가 파싱한 것 + content에 남은 `<tool_call>` 블록.

    둘 다 없고 content도 비었는데 reasoning 안에 `<tool_call>` 블록이 있으면 그것을 쓴다. 실서버 관찰: 모델이 `</think>` 없이
    도구 호출을 쓰면 reasoning 파서(deepseek_r1)가 출력 전체를 reasoning으로 넘겨 content와 tool_calls가 모두 비는데, 호출 자체는
    온전했다(리뷰어가 통화의 약 1/3에서 실패). content가 있으면 reasoning은 보지 않는다: 사고 안에서 언급만 한 호출을 세지 않으려는
    agent.py의 원칙을 지킨다(모델의 실제 답이 아예 없을 때만 쓴다)."""
    calls = [*meta["tool_calls"], *_content_calls(content)]
    if not calls and not _strip_think(content).strip():
        calls = _content_calls(meta.get("reasoning") or "")
    return calls


def _parse_review(calls: list[dict]) -> tuple[bool, str, int | None, str] | None:
    """submit_review 호출의 인자에서 (achieved, notes, stage, evidence)를 읽는다. achieved·notes 타입이 맞지 않으면 None.

    stage는 0~4 정수가 아니면 None(옛 형식 호환), evidence는 문자열이 아니면 빈 문자열."""
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
            stage = a.get("stage")
            stage = stage if isinstance(stage, int) and not isinstance(stage, bool) and 0 <= stage <= 4 else None
            evidence = a.get("evidence")
            return achieved, notes.strip(), stage, evidence.strip() if isinstance(evidence, str) else ""
    return None


def _norm(s: str) -> str:
    return re.sub(r"[\s.,!?~…'\"“”‘’]+", "", s)


_SPEAKER_LABELS = {"caller": "발신", "callee": "수신"}  # _transcript가 줄 머리에 붙이는 화자 표시
_LABEL_RE = re.compile(r"^\s*\[(발신|수신)\]\s*")


def _evidence_ok(record: dict, evidence: str) -> bool:
    """수락 근거로 인용한 말이 실제 수신자 발화에 있는가(리뷰어 환각으로 달성률이 부풀지 않게 한다).

    리뷰어는 대화록의 줄 머리 화자 표시("[수신] ")까지 그대로 베껴 쓰는 일이 흔하다(실측: 올바른 수락 3/3이 이 때문에 취소됐다).
    그래서 화자 표시는 떼고 비교한다. 수신 표시가 붙은 줄이 있으면 그 줄만 본다(발신자 말을 섞어 인용한 경우 대비).
    남은 모든 줄이 수신자 발화에 들어 있어야 한다."""
    raw = [l for l in evidence.splitlines() if l.strip()]
    callee_lines = [l for l in raw if _LABEL_RE.match(l) and _LABEL_RE.match(l).group(1) == _SPEAKER_LABELS["callee"]]
    lines = [_LABEL_RE.sub("", l) for l in (callee_lines or raw)]
    quoted = [_norm(l) for l in lines if len(_norm(l)) >= 2]
    if not quoted:
        return False
    callee = _norm("".join(t["text"] for t in record["turns"] if t["speaker"] == "callee" and t["text"]))
    return all(q in callee for q in quoted)


def _finalize(record: dict, parsed: tuple[bool, str, int | None, str]) -> tuple[bool, str, int]:
    """달성은 근거 인용이 기록에 있을 때만 인정하고, 단계는 달성 여부와 어긋나지 않게 맞춘다(보수적으로)."""
    achieved, notes, stage, evidence = parsed
    if achieved and not _evidence_ok(record, evidence):
        achieved = False
        notes += " (수락 근거 인용이 수신자 발화에 없어 달성을 취소함)"
    stage = 4 if achieved else min(stage if stage is not None else 0, 3)
    return achieved, notes, stage


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
        parsed = None if meta["finish_reason"] == "length" else _parse_review(tool_calls_of(content, meta))
        if parsed or spam:
            break
    if spam:
        return {"spam": True, "achieved": False, "stage": min(parsed[2] or 0, 3) if parsed else 0,
                "notes": parsed[1] if parsed else "(수신자가 스팸으로 신고하고 끊음. 리뷰 생성 실패)",
                "latency_s": latency}
    if parsed is None:
        return None
    achieved, notes, stage = _finalize(record, parsed)
    return {"spam": False, "achieved": achieved, "stage": stage, "notes": notes, "latency_s": latency}
