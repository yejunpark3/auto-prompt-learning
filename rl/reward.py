"""통화 한 건의 화자별 보상(자연스러움). 시나리오를 모른다.

화자(caller/callee)마다 따로 점수를 낸다:
    s = 평균(J/2) − DEFECT_W·(결함 수 / 발화 수) − ENDING_W·[끝맺음 부자연]       (J: 심판의 발화별 0~2점)
바닥값 s = FLOOR: 발신자가 지어냄(rl.fabrication), 발신자가 용건 없이 먼저 끊음(리뷰어 단계 0 + hangup_by_caller).
    가산 벌점이 아니라 바닥값이다: 지어내서 얻은 이득이 벌점보다 크면 지어내기를 학습한다.
제외(excluded): 서버 오류·빈 응답, 심판·리뷰 실패. 제외된 통화는 학습 데이터와 평가에서 뺀다.

사용: uv run -m rl.reward --dir tmp/ms3_nolabel_s31 [--no-judge] [--force]
--no-judge는 모델 없이 결정적 결함만 센다(캐시하지 않는다).
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import learner
import reviewer
from agent import _ACTION_RE, _BRACKET_RE, _END_TAG_RE, _LABEL_RE, _TOOL_CALL_RE, LLMConfig, PhoneAgent, _strip_think
from phone_call import NUDGE_RATIO, _similar
from prompts import (
    NATURAL_JUDGE_REQUEST,
    NATURAL_JUDGE_SYSTEM,
    SUBMIT_NATURALNESS_NAME,
    SUBMIT_NATURALNESS_TOOL,
)
from rl import fabrication, judge_config

VERSION = 2  # 2: 심판 요청문에 번호 붙은 발화 수(n)를 명시(점수 개수 불일치로 판정이 비는 것을 줄임)
FLOOR = -1.0
DEFECT_W = 0.2     # 발화당 결함 한 건의 벌점(결함 수/발화 수에 곱한다)
ENDING_W = 0.3     # 끝맺음이 부자연스러울 때의 벌점
SKIP_ENDS = {"error", "empty_response"}
LOOPY_ENDS = {"repetition_loop", "max_turns"}  # 자연스럽게 끝난 통화가 아니다(심판 대신 코드가 끝맺음 부자연으로 본다)
ROLES = ("caller", "callee")

_LATIN_RE = re.compile(r"[A-Za-z]{3,}")
_COLON_RE = re.compile(r"[^\d\s]\s*[:：]")  # 대사 중간의 `이름:` 같은 화자 표시(시각 3:30은 제외)


# ───────────── 결정적 결함 ─────────────

def model_turns(record: dict, role: str) -> list[dict]:
    """모델이 만든 그 화자의 발화(수신자의 고정 인사 turn 0과 오류 turn은 제외)."""
    return [t for t in record["turns"] if t["speaker"] == role and t["turn"] > 0 and "error" not in t]


def _raw_by_idx(record: dict, role: str) -> dict[int, str]:
    """TRAJ_LOG 기록에서 발화 순번(0부터) → 정리 전 원문. 재시도가 있으면 마지막 것이 쓰인 응답이다."""
    return {e["idx"]: e["raw"] for e in record.get(role, {}).get("traj", []) if e["kind"] == "turn"}


def _has_artifact(raw: str) -> bool:
    """정리 전 원문에 화자 표시·대괄호·행동 묘사가 있었는가(clean_utterance가 걷어내 최종 대사에서는 안 보인다)."""
    core = _END_TAG_RE.sub("", _TOOL_CALL_RE.sub("", _strip_think(raw)))
    return bool(_LABEL_RE.match(core) or _BRACKET_RE.search(core) or _ACTION_RE.search(core))


def _persona_grams(record: dict) -> set[str]:
    grams: set[str] = set()
    for v in (record.get("callee", {}).get("persona") or {}).values():
        if isinstance(v, str) and len(v) >= learner.COPY_GRAM:
            grams |= learner._grams(v)
    return grams


def find_defects(record: dict, role: str) -> list[dict]:
    """[{"turn": n, "kind": ...}]. 종류: latin(외국어 단어), script_label(대본식 화자 표시), artifact(원문의 라벨·대괄호·지문),
    too_long(문장 수 제한에 잘림), repeat(자기 직전 말과 비슷), recites_persona(수신자가 페르소나 원문을 12자 이상 낭독),
    unanswered_hangup(상대 질문 직후 말없이 끊음)."""
    raws = _raw_by_idx(record, role)
    grams = _persona_grams(record) if role == "callee" else set()
    out: list[dict] = []
    prev = None
    for k, t in enumerate(model_turns(record, role)):
        text, kinds = t.get("text", ""), []
        if text:
            if _LATIN_RE.search(text):
                kinds.append("latin")
            if _COLON_RE.search(text):
                kinds.append("script_label")
            if k in raws and _has_artifact(raws[k]):
                kinds.append("artifact")
            if "truncated_from" in t:
                kinds.append("too_long")
            if prev and _similar(prev, text, NUDGE_RATIO):
                kinds.append("repeat")
            if grams and learner._grams(text) & grams:
                kinds.append("recites_persona")
            prev = text
        elif t.get("hangup"):
            before = record["turns"][t["turn"] - 1]["text"] if t["turn"] > 0 else ""
            if before.rstrip().endswith("?"):
                kinds.append("unanswered_hangup")
        out += [{"turn": t["turn"], "kind": kind} for kind in kinds]
    return out


# ───────────── 심판 ─────────────

def _utterances(record: dict) -> list[tuple[int, str, str]]:
    """심판이 점수를 매길 발화 [(번호 n, 화자, 글)]. 번호는 1부터, 글이 있는 모델 발화만."""
    out = []
    for t in record["turns"]:
        if t["turn"] > 0 and t.get("text") and "error" not in t:
            out.append((len(out) + 1, t["speaker"], t["text"]))
    return out


def _transcript(record: dict, utts: list[tuple[int, str, str]]) -> str:
    label = {"caller": "발신", "callee": "수신"}
    lines, it = [], iter(utts)
    for t in record["turns"]:
        if t["turn"] == 0:
            lines.append(f"[{label[t['speaker']]}] {t['text']}")  # 고정 인사는 번호 없이 보여 주기만 한다
        elif t.get("text") and "error" not in t:
            n, _, text = next(it)
            lines.append(f"#{n} [{label[t['speaker']]}] {text}")
        elif t.get("hangup"):
            lines.append(f"[{label[t['speaker']]}] (말없이 끊음)")
    return "\n".join(lines)


def _as_bool(v):
    return v if isinstance(v, bool) else {"true": True, "false": False}.get(str(v).strip().lower())


def _parse_judge(calls: list[dict], n: int) -> tuple[list[int], bool, str] | None:
    for c in calls:
        if c["name"] != SUBMIT_NATURALNESS_NAME:
            continue
        try:
            a = json.loads(c["arguments"] or "{}")
        except ValueError:
            continue
        if not isinstance(a, dict):
            continue
        scores = a.get("scores")
        if isinstance(scores, str):  # XML 파싱 경로에서는 배열이 문자열로 온다
            try:
                scores = json.loads(scores)
            except ValueError:
                continue
        ending = _as_bool(a.get("ending_ok"))
        if (isinstance(scores, list) and len(scores) == n and ending is not None
                and all(isinstance(s, int) and not isinstance(s, bool) and 0 <= s <= 2 for s in scores)):
            return scores, ending, str(a.get("notes") or "").strip()
    return None


def judge_call(record: dict, llm: LLMConfig, attempts: int = 2) -> dict | None:
    """발화별 0~2점과 끝맺음을 받는다: {"scores": {n: s}, "ending_ok": bool, "notes": str, "latency_s": float}. 실패는 None."""
    utts = _utterances(record)
    if not utts:
        return {"scores": {}, "ending_ok": False, "notes": "", "latency_s": 0.0}
    agent = PhoneAgent("judge", None, "심판", NATURAL_JUDGE_SYSTEM, llm)
    request = NATURAL_JUDGE_REQUEST.format(end_reason=record["end_reason"], transcript=_transcript(record, utts), n=len(utts))
    latency = 0.0
    for _ in range(attempts):
        content, _, meta = agent._complete(
            messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": request}],
            max_tokens=llm.plan_max_tokens, tools=[SUBMIT_NATURALNESS_TOOL],
        )
        latency += meta["latency_s"]
        if meta["finish_reason"] == "length":
            continue
        parsed = _parse_judge(reviewer.tool_calls_of(content, meta), len(utts))
        if parsed:
            scores, ending, notes = parsed
            return {"scores": {n: s for (n, _, _), s in zip(utts, scores)}, "ending_ok": ending,
                    "notes": notes, "latency_s": round(latency, 2)}
    return None


# ───────────── 점수 ─────────────

def speaker_score(scores: list[int], n_defects: int, ending_ok: bool) -> float | None:
    """판단할 발화가 없으면 None."""
    if not scores:
        return None
    return round(statistics.fmean(s / 2 for s in scores) - DEFECT_W * n_defects / len(scores)
                 - ENDING_W * (not ending_ok), 3)


def score_call(record: dict, judge_llm: LLMConfig | None, judge: bool = True, check_fabrication: bool = True) -> dict:
    """{"version", "excluded": 사유|None, "complete": 심판·검사까지 돌렸는가, "caller": {...}, "callee": {...}, ...}.
    judge=False면 결정적 결함만 센다(complete=False, s는 None).
    check_fabrication=False면 지어내기 검사와 그 바닥값을 건너뛴다(GEPA처럼 성공만 보는 용도; 기본은 검사한다).
    이때 result["fabrication"]은 {"skipped": True, "fabricated": False}이고 용건 없이 끊은 바닥값만 남는다."""
    result: dict = {"version": VERSION, "end_reason": record["end_reason"], "excluded": None, "complete": False,
                    "defects": {r: find_defects(record, r) for r in ROLES}}
    if record["end_reason"] in SKIP_ENDS:
        result["excluded"] = record["end_reason"]
        return result
    if not judge:
        return result

    jr = judge_call(record, judge_llm)
    if jr is None:
        result["excluded"] = "judge_failed"
        return result
    review = record.get("review") or reviewer.review_call(record, judge_llm)
    if review is None:
        result["excluded"] = "review_failed"
        return result
    fab = fabrication.detect(record, judge_llm) if check_fabrication else \
        {"numbers": [], "contacts": [], "claims": None, "fabricated": False, "skipped": True}
    if fab["fabricated"] is None:
        result["excluded"] = "fabrication_check_failed"
        result["fabrication"] = fab
        return result

    loopy = record["end_reason"] in LOOPY_ENDS
    ending_ok = False if loopy else jr["ending_ok"]
    floors = {"caller": "fabricated" if fab["fabricated"]
              else "no_purpose_hangup" if record["end_reason"] == "hangup_by_caller" and review["stage"] == 0
              else None,
              "callee": None}
    result.update(complete=True, stage=review["stage"], achieved=review["achieved"], fabrication=fab,
                  ending_ok=ending_ok, judge_notes=jr["notes"])
    utts = _utterances(record)
    for role in ROLES:
        j = [jr["scores"][n] for n, r, _ in utts if r == role]
        n_def = len(result["defects"][role])
        s = speaker_score(j, n_def, ending_ok)
        result[role] = {"s": FLOOR if floors[role] else s, "raw_s": s, "floor": floors[role], "j": j,
                        "n_utts": len(j), "n_defects": n_def}
    return result


# ───────────── CLI ─────────────

def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="통화 기록의 화자별 자연스러움 보상을 계산한다")
    p.add_argument("--dir", required=True, help="통화 기록(JSON) 폴더")
    p.add_argument("--no-judge", action="store_true", help="모델 없이 결정적 결함만 센다(캐시하지 않음)")
    p.add_argument("--force", action="store_true", help="캐시된 rl 결과를 무시하고 다시 계산")
    p.add_argument("--limit", type=int, help="앞에서 N건만")
    args = p.parse_args()

    files = sorted(Path(args.dir).glob("*.json"))[: args.limit]
    llm = None if args.no_judge else judge_config()
    rows = []
    for f in files:
        rec = _load(f)
        if rec is None or "turns" not in rec:
            continue
        cached = rec.get("rl")
        if cached and cached.get("complete") and cached.get("version") == VERSION and not args.force and not args.no_judge:
            result = cached
        else:
            try:
                result = score_call(rec, llm, judge=not args.no_judge)
            except Exception as e:  # 서버 오류 한 번이 배치 전체를 멈추지 않게 한다
                print(f"[경고] {f.name}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            if result["complete"]:
                rec["rl"] = result
                f.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append((f, rec, result))
        kinds = Counter(d["kind"] for r in ROLES for d in result["defects"][r])
        s = " ".join(f"{r[:2]}={result[r]['s']}" for r in ROLES if r in result) if result["complete"] else ""
        floor = next((result[r]["floor"] for r in ROLES if r in result and result[r]["floor"]), None)
        print(f"{f.name[:40]:40} {rec['end_reason']:18} {rec['num_turns']:>2}턴 {s} {('바닥:' + floor) if floor else ''}"
              f"{('제외:' + result['excluded']) if result['excluded'] else ''} {dict(kinds) or ''}")

    print(f"\n통화 {len(rows)}건")
    kinds = Counter(d["kind"] for _, _, r in rows for role in ROLES for d in r["defects"][role])
    print("결함:", dict(kinds) or "없음")
    done = [r for _, _, r in rows if r["complete"]]
    for role in ROLES:
        ss = [r[role]["s"] for r in done if r[role]["s"] is not None]
        if ss:
            print(f"{role}: 평균 s={statistics.fmean(ss):.3f} (n={len(ss)}, 바닥값 {sum(r[role]['floor'] is not None for r in done)}건)")
    print(f"제외: {dict(Counter(r['excluded'] for _, _, r in rows if r['excluded'])) or '없음'}")


if __name__ == "__main__":
    main()
