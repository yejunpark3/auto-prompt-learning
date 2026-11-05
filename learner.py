import argparse
import json
import os
import random
import re
import secrets
import shutil
import string
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import reviewer
import wiki_store
from agent import LEARN_TOOL_ROUNDS, LLMConfig, PhoneAgent, _strip_think
from prompts import (
    WIKI_RULE_OFF,
    WIKI_RULE_ON,
    FACT_CHECK_NAME,
    FACT_CHECK_REQUEST,
    FACT_CHECK_TOOL,
    LEARN_REQUEST,
    SYSTEM_PROMPT_PATH,
    load_caller_profile,
)

LEARN_DIR = Path("agent_workspace/learning")
PENDING = LEARN_DIR / "pending.jsonl"
# 학습자의 위키 쓰기는 기본 꺼짐. 5회에 걸친 실측에서 14B 학습자가 위키에 쓴 내용은 매번 지어낸 사실·중복·변경 이력 문장이었고,
# 사실 검사기(재현율 한계)가 막지 못한 문장을 발신자가 통화에서 그대로 따라 말했다. 켜려면 .env에 LEARN_WIKI_WRITES=1.
LEARN_WIKI_WRITES = os.environ.get("LEARN_WIKI_WRITES", "0") == "1"
PROMPT_MAX_CHARS = 600
PROMPT_MIN_CHARS = 20
WRITE_TOOLS = {"wiki_create", "wiki_update", "wiki_delete"}

PROMPT_EDIT_TOOL = wiki_store._tool(
    "prompt_edit",
    f"발신자 system.md를 줄 ID([k7x2] 같은 영문 소문자+숫자 4자리 코드)로 고친다. replace는 그 줄을 new_text로 바꾸고, insert_after는 그 줄 뒤에 new_text를 넣고(line_id가 START면 맨 앞), delete는 그 줄을 지운다. "
    f"new_text에는 [ID] 표시 없이 내용만 쓴다. 고치지 않은 줄의 ID는 그대로이고 새로 생긴 줄만 새 ID를 받는다 (결과는 최대 {PROMPT_MAX_CHARS}자)",
    {"action": {"type": "string", "enum": ["replace", "insert_after", "delete"]},
     "line_id": {"type": "string"}, "new_text": {"type": "string"}},
    ["action", "line_id"],
)
LINE_TAG = re.compile(r"^\[[a-z0-9]{4}\]\s*")
START = "START"
ID_CHARS = string.ascii_lowercase + string.digits


class PromptLines:
    """system.md의 줄마다 무작위 ID를 붙여 학습자에게 보여 준다 (ID는 파일에 쓰지 않고 한 회차 동안만 유지)."""

    def __init__(self, text: str):
        self._used: set[str] = set()
        self.rows = self._assign(text)

    def _new_id(self) -> str:
        while (i := "".join(secrets.choice(ID_CHARS) for _ in range(4))) in self._used or i.isalpha() or i.isdigit():
            pass
        self._used.add(i)
        return i

    def _assign(self, text: str) -> list[tuple[str, str]]:
        return [(self._new_id(), l) for l in text.strip().splitlines()]

    def text(self) -> str:
        return "\n".join(l for _, l in self.rows)

    def show(self) -> str:
        return "\n".join(f"[{i}] {l}" for i, l in self.rows)

    def sync(self, text: str) -> None:
        if text.strip() != self.text():  # 파일이 밖에서 바뀐 경우에만 새로 매긴다
            self.rows = self._assign(text)

    def edit(self, action: str, line_id: str, new_text: str) -> str | None:
        """성공하면 None, 실패하면 오류 문장."""
        if action not in ("replace", "insert_after", "delete"):
            return "오류: action은 replace, insert_after, delete 중 하나여야 합니다"
        line_id = line_id.strip().strip("[]").lower()
        ids = [i for i, _ in self.rows]
        if action == "insert_after" and line_id == START.lower():
            pos = 0
        elif line_id in ids:
            pos = ids.index(line_id) + (action == "insert_after")
        else:
            return f"오류: 없는 line_id입니다: {line_id}"
        new = [(self._new_id(), LINE_TAG.sub("", l)) for l in new_text.strip().splitlines()]
        if action != "delete" and not new:
            return f"오류: {action}에는 new_text가 필요합니다 (지우려면 delete)"
        if action == "insert_after":
            self.rows[pos:pos] = new
        else:
            self.rows[pos:pos + 1] = new if action == "replace" else []
        return None


def enqueue(record: dict, path: Path) -> int:
    LEARN_DIR.mkdir(parents=True, exist_ok=True)
    review = record.get("review") or {}
    row = {"call_id": record["call_id"], "path": str(path), "end_reason": record["end_reason"], "purpose": record.get("purpose", ""),
           "spam": review.get("spam"), "achieved": review.get("achieved"), "stage": review.get("stage"),
           "notes": review.get("notes", "")}
    with PENDING.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(_pending())


def _pending() -> list[dict]:
    if not PENDING.exists():
        return []
    rows = [json.loads(l) for l in PENDING.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if "achieved" in r and "spam" in r]
    if len(kept) < len(rows):  # 점수(score) 형식의 옛 행은 신고·달성 여부를 알 수 없다
        print(f"[경고] 옛 형식 대기열 {len(rows) - len(kept)}건을 버립니다", flush=True)
        PENDING.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    return kept


def _rate(rows: list[dict], key: str) -> float | None:
    return round(sum(bool(r.get(key)) for r in rows) / len(rows), 2) if rows else None


def _stage_mean(rows: list[dict]) -> float | None:
    """평균 진행 단계(0~4). 달성이 0건이어도 어디까지 갔는지 보여 준다. 단계가 없는 옛 행은 뺀다."""
    stages = [r["stage"] for r in rows if isinstance(r.get("stage"), int)]
    return round(sum(stages) / len(stages), 2) if stages else None


def _invented_rate(rows: list[dict]) -> float | None:
    """발신자가 근거 없는 사실을 말한 통화 비율. 검사하지 못한 통화(invented가 None이거나 없음)는 뺀다."""
    checked = [r for r in rows if isinstance(r.get("invented"), list)]
    return round(sum(bool(r["invented"]) for r in checked) / len(checked), 2) if checked else None


def _rates(rows: list[dict]) -> dict:
    return {"achieved_rate": _rate(rows, "achieved"), "spam_rate": _rate(rows, "spam"), "stage_mean": _stage_mean(rows),
            "invented_rate": _invented_rate(rows)}


def _copy_wiki(dst: Path) -> None:
    shutil.copytree(wiki_store.WIKI_ROOT, dst, ignore=shutil.ignore_patterns(".*"))


MAX_EDITS = 3  # 한 회차에서 고칠 수 있는 횟수(prompt_edit·위키 쓰기 합산). 같은 문서에 비슷한 내용을 11번 덧붙인 일이 있었다
QUOTE_RE = re.compile(r"[\"“'‘]([^\"”'’\n]{8,}?)[\"”'’]")  # 따옴표로 묶은 문구(통화에서 그대로 말하게 될 대사)
CONTACT_RE = re.compile(r"(?:https?://|www\.)[^\s'\"),]+|[\w.+-]+@[\w-]+\.[\w.-]+")  # URL·이메일
COPY_GRAM = 12  # 실패 통화의 발신자 말을 이 글자 수 이상 이어서 옮기면 거절


def _numbers(text: str, strict: bool) -> set[str]:
    """글 속 숫자(쉼표 제거). strict면 줄 머리 번호("1.", "2)")는 뺀다. 단위·표기(원, 세 등)는 보지 않는다: 시나리오마다 다르다."""
    if strict:
        text = re.sub(r"(?m)^\s*\d+[.)]\s*", "", text)
    return {m.group(0).replace(",", "") for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text)}


def _grams(text: str, k: int = COPY_GRAM) -> set[str]:
    s = reviewer._norm(text)
    return {s[i:i + k] for i in range(len(s) - k + 1)}


FACT_CHECK_MAX_TOKENS = 2500


def _readonly_docs() -> list[str]:
    docs = []
    for rel in wiki_store.list_pages():
        if wiki_store.is_readonly(rel):
            try:
                docs.append(wiki_store.read(rel))
            except wiki_store.WikiError:
                pass
    return docs


def _fact_source(rows: list[dict], prompt_text: str, docs: list[str] | None = None) -> str:
    """사실 근거로 인정하는 글: 현재 system.md, 통화 목적, 읽기 전용 문서(사람이 정한 지식).
    docs를 주면 위키 대신 그 문서들을 쓴다(벤치가 시나리오마다 근거를 따로 줄 때)."""
    docs = _readonly_docs() if docs is None else docs
    return "\n\n".join([prompt_text, *dict.fromkeys(r.get("purpose", "") for r in rows), *docs])


def _decoys(rows: list[dict], k: int = 3, docs: list[str] | None = None) -> list[str]:
    """검사 글에 섞을 '근거가 확실한' 문장: 읽기 전용 문서와 통화 목적의 문장. 실측에서 14B 검사기는 문장 하나만 주면
    정상 문장까지 전부 지적했고(정상 4/4 오차단), 여러 문장이 섞이면 구분했다(정상 0/8 오차단)."""
    # 사실을 서술한 읽기 전용 문서를 먼저, 통화 목적(사실 서술이 아님)은 뒤에 둔다. 목적 문구를 미끼로 쓰면 검사기가 지적했다
    parts = list(_readonly_docs() if docs is None else docs)
    parts += [r.get("purpose", "") for r in rows]
    out: list[str] = []
    for part in parts:
        for raw in re.split(r"\n+|(?<=[.!?])\s+", part):
            s = raw.strip().lstrip("-# ").strip()
            if len(s) >= 12 and s not in out:
                out.append(s)
    return out[:k]


def _fact_check(text: str, source: str, llm: LLMConfig, attempts: int = 2, decoys: list[str] | None = None, lenient: bool = False, request: str = FACT_CHECK_REQUEST) -> list[str] | None:
    """글 속의, 근거 문서에 없는 사실 주장 문장 목록. 숫자 없는 지어내기까지 잡는다. 검사 실패는 None(막지 않는다).

    decoys(근거가 확실한 문장)를 글에 섞어 검사하고, 검사기가 그 문장을 지적하면 전부 지적하는 상태이므로 결과를 버린다(None).
    lenient(감사용)면 미끼 지적분만 빼고 쓰되, 남은 지적이 글 줄 수의 30%(최소 2건)를 넘으면 그때 버린다."""
    if not text.strip():
        return []
    mixed = text
    if decoys:
        items = [l for l in text.splitlines() if l.strip()] + list(decoys)
        random.Random(len(text)).shuffle(items)
        mixed = "\n".join(items)
    agent = PhoneAgent("checker", None, "검사자", "글의 사실 주장이 근거 문서에 있는지 검사한다.", llm)
    request = request.format(source=source, text=mixed)
    for _ in range(attempts):
        content, _, meta = agent._complete(
            messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": request}],
            max_tokens=min(llm.plan_max_tokens, FACT_CHECK_MAX_TOKENS), tools=[FACT_CHECK_TOOL])
        if meta["finish_reason"] == "length":
            continue
        for c in [*meta["tool_calls"], *reviewer._content_calls(content)]:
            if c["name"] != FACT_CHECK_NAME:
                continue
            try:
                a = json.loads(c["arguments"] or "{}")
            except ValueError:
                continue
            claims = a.get("ungrounded") if isinstance(a, dict) else None
            if isinstance(claims, str):  # XML 파싱 경로에서는 배열이 문자열로 온다
                try:
                    claims = json.loads(claims)
                except ValueError:
                    claims = None
            if isinstance(claims, list) and all(isinstance(x, str) for x in claims):
                claims = [x.strip() for x in claims if x.strip()]
                if decoys and any(_same_line(d, c) for d in decoys for c in claims):
                    if lenient:
                        claims = [c for c in claims if not any(_same_line(d, c) for d in decoys)]
                        if len(claims) <= max(2, 0.3 * sum(1 for l in text.splitlines() if l.strip())):
                            return claims
                    print("[경고] 사실 검사기가 근거 있는 문장까지 지적해 믿을 수 없어 검사 없이 통과시킵니다", flush=True)
                    return None
                return claims
    print("[경고] 사실 검사에 실패해 검사 없이 통과시킵니다", flush=True)
    return None


def _fact_check_vote(text: str, source: str, llm: LLMConfig, decoys: list[str] | None = None, runs: int = 2, tries: int = 1, lenient: bool = False, parallel: bool = False, combine: str = "all", request: str = FACT_CHECK_REQUEST) -> list[str] | None:
    """_fact_check를 runs번 돌려 모든 실행이 함께 지적한 문장만 돌려준다. 한 번의 지적은 정상 요령 문장을 막는 일이 있었다
    (실측: 행동 지시 한 줄 오차단). 실행 중 믿을 수 없는 결과(None)가 나오면 처음부터 tries번까지 다시 해 보고, 끝내 안 되면 None.

    combine="any"면 한 번이라도 지적한 문장을 모두 돌려준다(합집합). 강화학습 보상용: 놓친 지어내기는 정책이 그 빈틈을 배우므로
    오지적을 감수하고 재현율을 우선한다. 학습자 가드의 기본은 "all"(정밀도 우선)이고 request로 검사 문구를 바꿀 수 있다."""
    for _ in range(max(1, tries)):
        check = lambda _: _fact_check(text, source, llm, decoys=decoys, lenient=lenient, request=request)
        if parallel and runs > 1:  # 서버가 요청 2개를 동시에 처리하므로 같은 검사를 동시에 돌려 시간을 줄인다
            with ThreadPoolExecutor(runs) as ex:
                results = list(ex.map(check, range(runs)))
        else:
            results = [check(i) for i in range(runs)]
        if any(r is None for r in results):
            continue
        if combine == "any":
            merged: list[str] = []
            for r in results:
                merged += [c for c in r if not any(_same_line(c, m) or _same_line(m, c) for m in merged)]
            return merged
        first, rest = results[0], results[1:]
        return [c for c in first if all(any(_same_line(c, o) or _same_line(o, c) for o in r) for r in rest)]
    return None


def _same_line(line: str, claim: str) -> bool:
    a, b = reviewer._norm(line), reviewer._norm(claim)
    return min(len(a), len(b)) >= 6 and (a in b or b in a)


def audit(llm: LLMConfig, fix: bool = False, prompt_path: str = SYSTEM_PROMPT_PATH, rows: list[dict] | None = None) -> list[dict]:
    """고칠 수 있는 위키 문서에서 근거 없는 사실 주장 줄을 찾는다. fix면 그 줄을 지운다(과거 회차가 남긴 지어내기 정리)."""
    source = _fact_source(rows or [], load_caller_profile(prompt_path))
    found = []
    for rel in wiki_store.list_pages():
        if wiki_store.is_readonly(rel):
            continue
        text = wiki_store.read(rel)
        claims = _fact_check_vote(text, source, llm, decoys=_decoys(rows or []), tries=2, lenient=True, parallel=True)
        if not claims:
            continue
        lines = text.splitlines()
        bad = {i for i, l in enumerate(lines) if any(_same_line(l, c) for c in claims)}
        found.append({"path": rel, "claims": claims, "removed_lines": [lines[i] for i in sorted(bad)] if fix else []})
        if fix and bad:
            wiki_store.update(rel, "\n".join(l for i, l in enumerate(lines) if i not in bad), False)
    return found


def _invented_claims(rows: list[dict], prompt_text: str, llm: LLMConfig, workers: int = 2) -> None:
    """각 통화의 발신자 발화에서 근거 문서에 없는 사실 주장을 찾아 row["invented"]에 넣는다(검사 불가면 None).
    발신자는 위키를 조회하지 않고도 구체적 조건·수치를 지어내 말하는 일이 있었다."""
    source, decoys = _fact_source(rows, prompt_text), _decoys(rows)

    def one(row: dict) -> list[str] | None:
        if isinstance(row.get("invented"), list):  # 다시 대기열에 넣은 통화는 검사 결과를 재사용한다
            return row["invented"]
        try:
            rec = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
            said = [t["text"] for t in rec["turns"] if t["speaker"] == "caller" and t["text"]]
            return _fact_check_vote("\n".join(said), source, llm, decoys=decoys, tries=2, lenient=True) if said else []
        except Exception as e:  # 한 통화의 검사 실패가 학습을 막지 않게 한다
            print(f"[경고] 발언 검사 실패({row.get('call_id')}): {type(e).__name__}: {e}", flush=True)
            return None

    with ThreadPoolExecutor(workers) as ex:
        for row, claims in zip(rows, ex.map(one, rows)):
            row["invented"] = claims


def _make_guard(rows: list[dict], prompt_text: str, llm: LLMConfig | None = None):
    """학습자가 새로 쓰는 글(system.md 줄, 위키 내용)에 대한 검사. 문제가 있으면 오류 문장, 없으면 None.

    1) 근거 없는 수치: 읽기 전용 문서·통화 목적·현재 system.md와 위키에 없는 숫자는 쓸 수 없다(14B 학습자가 문서에 없는 조건을 지어 썼다).
    2) 베끼기: 달성하지 못한 통화에서 발신자가 한 말을 그대로 규칙에 옮길 수 없다(비문을 모범 문구로 굳혔다).
       이미 system.md·위키에 있던 문구는 예외."""
    docs = []
    for rel in wiki_store.list_pages():
        try:
            docs.append((rel, wiki_store.read(rel)))
        except wiki_store.WikiError:
            pass
    existing = prompt_text + "\n" + "\n".join(t for _, t in docs)
    source = "\n".join([prompt_text, *(r.get("purpose", "") for r in rows),
                        *(t for rel, t in docs if wiki_store.is_readonly(rel))])
    fact_source = _fact_source(rows, prompt_text)
    decoys = _decoys(rows)
    known_numbers = _numbers(source, strict=False)
    known_grams = _grams(existing)
    known_norm = reviewer._norm(existing)
    bad_caller = set()
    for r in rows:
        if r.get("achieved"):
            continue
        try:
            rec = json.loads(Path(r["path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            continue
        for t in rec.get("turns", []):
            if t["speaker"] == "caller" and t["text"]:
                bad_caller |= _grams(t["text"])
    bad_caller -= known_grams

    def guard(text: str, strict: bool = False) -> str | None:
        if unknown := sorted(_numbers(text, strict=True) - known_numbers):
            return (f"숫자 {', '.join(unknown)}은(는) 통화 목적·읽기 전용 문서·현재 문서에 없는 값입니다. "
                    "근거 없는 수치·조건은 쓸 수 없습니다. 숫자 없이 다시 쓰세요")
        if unknown := [c for c in CONTACT_RE.findall(text) if c.lower() not in source.lower()]:
            return f"주소·연락처 {', '.join(unknown)}은(는) 근거 문서에 없습니다. 지어낸 주소·연락처는 쓸 수 없습니다"
        if copied := sorted(_grams(text) & bad_caller):
            # 비교는 공백·문장부호를 뺀 글로 했으므로, 학습자에게는 원문에서 같은 구간을 찾아 보여 준다.
            m = re.search(r"[\s.,!?~…'\"“”‘’]*".join(map(re.escape, copied[0])), text)
            return (f"달성하지 못한 통화에서 발신자가 한 말('{m.group(0) if m else copied[0]}…')을 그대로 옮겼습니다. "
                    "기록 속 말은 문제의 증거일 뿐 모범 문구가 아닙니다. 규칙은 직접 새로 쓰세요")
        if llm is not None:
            blocked = lambda claims: (f"근거 문서에 없는 사실 주장입니다: {' / '.join(claims[:2])}. "
                                      "읽기 전용 문서에 있는 내용만 사실로 쓰고, 모르는 것은 쓰지 말고 말하는 방법만 쓰세요")
            claims = _fact_check_vote(text, fact_source, llm, decoys=decoys, tries=1, parallel=True)
            if claims:
                return blocked(claims)
            if claims is None and strict:
                # 위키 내용은 발신자가 통화 중에 그대로 읽어 말한다(실측: 학습자가 쓴 지어낸 문장을 발신자가 따라 말했다).
                # 검사기가 믿을 수 없는 결과를 내면 검증하지 못한 것이므로 사실처럼 남기지 않는다.
                return ("사실 검사를 믿을 수 있게 하지 못해 위키에는 쓸 수 없습니다. 검증하지 못한 내용은 사실처럼 남기지 않습니다. "
                        "행동 규칙이면 system.md에 쓰세요")
            # 따옴표로 묶은 새 문구는 발신자가 그대로 말하게 되는 대사다(이미 문서에 있던 문구는 제외). 실측: 학습자가 지어낸 주장을 따옴표 대사로 박았다.
            # 전체 글 검사가 통과해도(또는 검사 불가여도) 따옴표 문구는 따로, 믿을 수 있는 결과가 나올 때만 통과시킨다.
            quoted_new = [q for q in QUOTE_RE.findall(text) if reviewer._norm(q) not in known_norm]
            if quoted_new and not strict:
                qclaims = _fact_check_vote("\n".join(quoted_new), fact_source, llm, decoys=decoys, tries=1, parallel=True)
                if qclaims:
                    return blocked(qclaims)
                if qclaims is None:
                    return ("따옴표로 묶은 문구는 통화에서 그대로 말하게 됩니다. 사실 검사를 믿을 수 있게 하지 못해 쓸 수 없습니다. "
                            "따옴표 없이 행동 규칙으로 쓰세요")
        return None

    return guard


def _tool_runner(prompt_path: str, lines: PromptLines, guard=None, max_edits: int = MAX_EDITS):
    edits = [0]  # 이번 회차에서 적용된 편집 수
    limit_msg = f"오류: 이번 회차에 이미 {max_edits}번 고쳤습니다. 더 고치지 말고 바꾼 내용을 한두 줄로 요약하며 끝내세요"

    def run(name: str, arguments: str) -> str:
        if (name == "prompt_edit" or name in WRITE_TOOLS) and edits[0] >= max_edits:
            return limit_msg
        if name == "prompt_edit":
            try:
                a = json.loads(arguments)
                if "line_id" not in a:  # 14B가 줄 ID 자체를 키로 보내는 실수를 했다({"[kya3]": "...", ...}): 키가 하나뿐이면 그것을 line_id로 본다
                    id_keys = [k for k in a if re.fullmatch(r"\[?[a-z0-9]{4}\]?", str(k))]
                    if len(id_keys) == 1:
                        a["line_id"] = id_keys[0]
                action, line_id, new = a["action"], str(a["line_id"]), str(a.get("new_text") or "")
                text = Path(prompt_path).read_text(encoding="utf-8")
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                return ('오류: 잘못된 인자입니다. 키는 action, line_id, new_text여야 합니다. '
                        '예: {"action": "replace", "line_id": "k7x2", "new_text": "새 내용"}')
            lines.sync(text)
            before = list(lines.rows)
            err = lines.edit(action, line_id, new)
            if err is None and action != "delete" and guard and (g := guard(new)):
                err = f"오류: {g}"
            edited = lines.text().strip()
            if err is None and len(edited) < PROMPT_MIN_CHARS:
                err = f"오류: 결과가 {len(edited)}자입니다. 최소 {PROMPT_MIN_CHARS}자여야 합니다"
            if err is None and len(edited) > PROMPT_MAX_CHARS:
                err = f"오류: 결과가 {len(edited)}자입니다. {PROMPT_MAX_CHARS}자 이하로 줄이세요"
            if err:
                lines.rows = before
                return f"{err}\n현재 system.md:\n{lines.show()}"
            Path(prompt_path).write_text(edited + "\n", encoding="utf-8")
            edits[0] += 1
            return f"system.md를 고쳤습니다. 최신 줄 ID:\n{lines.show()}"
        if name in WRITE_TOOLS:
            try:
                a = json.loads(arguments)
                rel, _ = wiki_store._resolve(a.get("path", ""))
            except (ValueError, TypeError, AttributeError, wiki_store.WikiError) as e:
                return f"오류: {e}"
            if wiki_store.is_readonly(rel):
                return f"오류: {rel}은 고칠 수 없는 문서입니다"
            if guard and name in ("wiki_create", "wiki_update"):
                content = str(a.get("content") or "")
                if name == "wiki_update" and not a.get("append"):  # 통째로 다시 쓸 때는 새로 들어오는 줄만 검사한다
                    try:
                        old = set(wiki_store.read(rel).splitlines())
                    except wiki_store.WikiError:
                        old = set()
                    content = "\n".join(l for l in content.splitlines() if l not in old)
                if g := guard(content, strict=True):
                    return f"오류: {g}"
        result = wiki_store.run_tool(name, arguments)
        if name in WRITE_TOOLS and result.startswith(("만듦", "고침", "지움")):
            edits[0] += 1
            if name != "wiki_delete":  # 학습자가 자기가 쓴 내용을 볼 수 있어야 같은 말을 되풀이해 덧붙이지 않는다
                try:
                    result += f"\n현재 문서:\n{wiki_store.read(rel)}"
                except wiki_store.WikiError:
                    pass
        return result
    return run


# 14B 리뷰어의 노트는 "수신자의 거절을 무시했다"처럼 틀린 진단이 섞여 학습자를 일찍 끊는 쪽으로 끌었다(2회차).
# 켜면 통화 결과 줄에 노트를 붙인다. 기본은 단계·달성·종료 사유만 보여 주고 원문 발췌로 판단하게 한다.
SHOW_NOTES = False
TRANSCRIPT_BUDGET = 9000  # 발췌 전체 글자 수 상한. 통화 수로 나눠 쓴다(컨텍스트·KV 캐시 절약)
LINE_MAX_CHARS = 80


def _excerpts(rows: list[dict]) -> str:
    """리뷰 대기열 각 통화의 실제 대화 앞부분. 14B 리뷰어의 요약이 틀려도 학습자가 원문을 볼 수 있게 한다."""
    per_call = max(300, TRANSCRIPT_BUDGET // max(1, len(rows)))
    out = []
    for i, r in enumerate(rows, 1):
        try:
            rec = json.loads(Path(r["path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            continue
        shown, used = [], 0
        # reviewer._transcript는 빈 발화를 뺀다. 학습자에게는 마지막의 말없는 끊김이 보여야 "질문에 답하지 않고 끊음"을 고칠 수 있다.
        transcript = "\n".join(f"[{reviewer._SPEAKER_LABELS[t['speaker']]}] {t['text'] or '(말없이 끊음)'}" for t in rec["turns"])
        for line in transcript.splitlines():
            line = line if len(line) <= LINE_MAX_CHARS else line[:LINE_MAX_CHARS] + "…"
            if used + len(line) > per_call:
                shown.append("…")
                break
            shown.append(line)
            used += len(line)
        out.append(f"<통화 {i}: {r['end_reason']}, 달성 {'예' if r['achieved'] else '아니오'}>\n" + "\n".join(shown))
    return "\n\n".join(out) or "(없음)"


def _history(n: int, current: dict, k: int = 3) -> str:
    """이전 k회차의 변경과, 그 변경 전후 묶음의 신고율·달성률.

    round_i의 입력 묶음은 round_{i-1}이 만든 프롬프트로 돌았으므로, round_i 변경의 효과는
    다음 묶음(round_{i+1}의 입력, 마지막 회차는 지금 묶음)에서 보인다."""
    def rates_of(d: Path) -> dict | None:
        try:
            return json.loads((d / "reviews.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def fmt(r: dict | None) -> str:
        if not r:
            return "알 수 없음"
        stage = f", 평균 단계 {r['stage_mean']}/4" if r.get("stage_mean") is not None else ""
        invented = f", 지어낸 말 한 통화 {r['invented_rate']}" if r.get("invented_rate") is not None else ""
        return f"신고율 {r.get('spam_rate')}, 달성률 {r.get('achieved_rate')}{stage}{invented}"

    lines = []
    for i in range(max(1, n - k), n):
        d = LEARN_DIR / f"round_{i:03d}"
        try:
            summary = json.loads((d / "learner.json").read_text(encoding="utf-8")).get("summary", "")
        except (OSError, ValueError):
            continue
        rolled_back = (d / "rolled_back").exists()
        auto = rolled_back and (d / "rolled_back").read_text(encoding="utf-8").startswith(AUTO_ROLLBACK_TAG)
        if i + 1 == n:
            after = f"{fmt(current)} (이 결과가 나빠 자동 롤백됨)" if auto else "적용된 적 없음" if rolled_back else fmt(current)
        else:
            after = fmt(rates_of(LEARN_DIR / f"round_{i + 1:03d}"))
        note = " [롤백됨: 현재 프롬프트에 없음]" if rolled_back else ""
        for part, name in (("system", "system.md"), ("wiki", "위키")):
            if (d / f"rolled_back_{part}").exists():
                note += f" [{name} 변경만 롤백됨]"
        # 요약 속 따옴표 문구는 학습자가 그대로 다시 쓴다(1회차 문구가 2회차에, 롤백되지 않은 5회차의 문구가 7회차의 지어낸 대사로 재등장) → 모든 회차에서 지운다
        text = re.sub(r"['\"‘“][^'\"’”]*['\"’”]", "…", " ".join(summary.split()))
        lines.append(
            f"- round_{i:03d} 변경{note}: {text[:300]}\n"
            f"  변경 전 묶음: {fmt(rates_of(d))} → 변경 후 묶음: {after}")
    return "\n".join(lines) or "(없음)"


AUTO_ROLLBACK_TAG = "자동:"
ROLLBACK_ACHIEVED_DROP = 0.2   # 달성률이 이만큼 이상 떨어지면 직전 변경을 되돌린다
# 달성률이 늘지 않은 채 평균 단계가 이만큼 이상 떨어져도 되돌린다(0~4 척도). 10통화 묶음의 평균 단계는 사실상 같은 설정에서도 0.9
# (1.6 대 0.7)나 흔들렸으므로, 잡음 밖인 1.0으로 둔다. 더 믿을 만한 판단은 달성률(수락 수)이다.
ROLLBACK_STAGE_DROP = 1.0


def _wiki_files(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): p.read_text(encoding="utf-8") for p in sorted(d.rglob("*")) if p.is_file()}


def _round_changed(d: Path) -> bool:
    """그 회차가 지금도 살아 있는 변경을 system.md나 위키에 남겼는가. 일부 롤백(rolled_back_system/wiki)된 부분은 세지 않는다."""
    try:
        if not (d / "rolled_back_system").exists() and \
                (d / "system_before.md").read_text(encoding="utf-8") != (d / "system_after.md").read_text(encoding="utf-8"):
            return True
        if (d / "rolled_back_wiki").exists():
            return False
        return _wiki_files(d / "wiki_before") != _wiki_files(d / "wiki_after")
    except OSError:
        return False


def _rollback_reason(before: dict | None, cur: dict) -> str | None:
    """직전 회차 변경 전 묶음(before)보다 변경 후 묶음(cur)이 뚜렷이 나쁘면 이유를 돌려준다."""
    if not before or before.get("achieved_rate") is None or cur.get("achieved_rate") is None:
        return None
    if before["achieved_rate"] - cur["achieved_rate"] >= ROLLBACK_ACHIEVED_DROP:
        return f"달성률 {before['achieved_rate']}→{cur['achieved_rate']}"
    bs, cs = before.get("stage_mean"), cur.get("stage_mean")
    if bs is not None and cs is not None and bs - cs >= ROLLBACK_STAGE_DROP and cur["achieved_rate"] <= before["achieved_rate"]:
        return f"평균 단계 {bs}→{cs}"
    return None


def _restore_state(system_src: Path, wiki_src: Path, prompt_path: str, system: bool = True, wiki: bool = True) -> None:
    """system.md와 위키를 주어진 스냅샷으로 되돌린다(숨김 경로 .readonly·.index는 건드리지 않는다). system·wiki로 일부만 고를 수 있다."""
    root = wiki_store.WIKI_ROOT
    if system:
        shutil.copy(system_src, prompt_path)
    if wiki:
        for p in list(root.rglob("*")):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
                p.unlink()
        for rel, text in _wiki_files(wiki_src).items():
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")


def _restore(d: Path, prompt_path: str, reason: str, tag: str = AUTO_ROLLBACK_TAG, only: str | None = None) -> None:
    """round의 *_before로 되돌린다. only("system"|"wiki")면 그 부분만 되돌리고 rolled_back_<부분> 표시를, 아니면 rolled_back 표시를 남긴다.
    rolled_back은 이력에서 '적용된 적 없음'/'자동 롤백'으로 읽힌다."""
    _restore_state(d / "system_before.md", d / "wiki_before", prompt_path, system=only in (None, "system"), wiki=only in (None, "wiki"))
    (d / ("rolled_back" if only is None else f"rolled_back_{only}")).write_text(f"{tag} {reason}", encoding="utf-8")


def _auto_rollback(n: int, current: dict, prompt_path: str) -> dict | None:
    """직전 회차(n-1)의 변경으로 돌린 묶음이 지금 대기열이다. 그 변경 전 묶음보다 뚜렷이 나쁘면 되돌린다.

    20통화도 변동이 커서 작은 차이는 우연일 수 있으므로 임계값을 크게 잡았다. 같은 수신자 시드로 돌린 묶음끼리일 때 믿을 만하다."""
    d = LEARN_DIR / f"round_{n - 1:03d}"
    if n < 2 or (d / "rolled_back").exists() or not _round_changed(d):
        return None
    try:
        before = json.loads((d / "reviews.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    reason = _rollback_reason(before, current)
    if reason is None:
        return None
    _restore(d, prompt_path, reason)
    return {"round": n - 1, "reason": reason}


def _logged(run, log: list[dict]):
    """도구 실행기를 감싸 호출과 결과를 log에 남긴다(학습자 호출이 중간에 끊겨도 적용된 편집을 알 수 있게)."""
    def wrapped(name: str, arguments: str) -> str:
        result = run(name, arguments)
        log.append({"name": name, "arguments": arguments, "result": result[:300]})
        return result
    return wrapped


def learn(llm: LLMConfig, prompt_path: str = SYSTEM_PROMPT_PATH, keep_on_fail: bool = False) -> Path:
    """keep_on_fail면 학습자 호출이 실패했을 때 대기열을 남겨 다시 돌릴 수 있게 한다(수동 실행용; 자동 실행은 재시도 폭주를 막으려 비운다)."""
    rows = _pending()
    if not wiki_store.READONLY_FILE.exists():
        print(f"[경고] {wiki_store.READONLY_FILE}이 없어 모든 위키 문서를 고칠 수 있습니다", flush=True)
    n = len(list(LEARN_DIR.glob("round_*"))) + 1
    out = LEARN_DIR / f"round_{n:03d}"
    result: dict = {}
    calls_log: list[dict] = []  # 학습자 호출이 중간에 실패해도 어떤 편집이 적용됐는지 남긴다
    learner_failed = False
    try:
        try:
            _invented_claims(rows, load_caller_profile(prompt_path), llm)
        except Exception as e:
            print(f"[경고] 발언 검사 실패: {type(e).__name__}: {e}", flush=True)
        rates = _rates(rows)
        if rolled := _auto_rollback(n, rates, prompt_path):  # 새 회차의 *_before를 만들기 전에 되돌려야 한다
            print(f"[롤백] round_{rolled['round']:03d} 변경이 나빠 되돌립니다: {rolled['reason']}", flush=True)
            result["auto_rollback"] = rolled
        out.mkdir(parents=True)
        shutil.copy(prompt_path, out / "system_before.md")
        _copy_wiki(out / "wiki_before")
        try:  # 과거 회차가 남긴 근거 없는 사실 주장은 학습자가 일하기 전에 정리한다(wiki_before/after 차이에 남는다)
            if audited := audit(llm, fix=True, prompt_path=prompt_path, rows=rows):
                result["audit"] = audited
                print(f"[감사] 근거 없는 사실 주장 {sum(len(a['claims']) for a in audited)}건 정리", flush=True)
        except Exception as e:
            result["audit_error"] = f"{type(e).__name__}: {e}"
            print(f"[경고] 위키 감사 실패: {result['audit_error']}", flush=True)
        _copy_wiki(out / "wiki_audited")  # 감사 정리가 끝난 상태. 학습자 호출이 실패하면 여기로 되돌린다
        (out / "reviews.json").write_text(
            json.dumps({**rates, "reviews": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        lines = PromptLines(load_caller_profile(prompt_path))
        reviews = "\n".join(
            f"- 통화 {i}: 목적 {r.get('purpose', '')}, {r['end_reason']}, 단계 {r['stage'] if r.get('stage') is not None else '?'}/4, "
            f"달성 {'예' if r['achieved'] else '아니오'}, 신고 {'예' if r['spam'] else '아니오'}"
            + (f"\n  근거 없는 발언: {' / '.join(c[:70] for c in r['invented'][:2])}" if r.get("invented") else "")
            + (f"\n  리뷰 노트: {r['notes']}" if SHOW_NOTES else "")
            for i, r in enumerate(rows, 1))
        request = LEARN_REQUEST.format(
            n=len(rows), achieved_rate=rates["achieved_rate"], spam_rate=rates["spam_rate"], stage_mean=rates["stage_mean"],
            reviews=reviews, transcripts=_excerpts(rows), history=_history(n, rates), prompt=lines.show(),
            pages=("\n".join(p for p in wiki_store.list_pages() if not wiki_store.is_readonly(p)) or "(없음)") if LEARN_WIKI_WRITES
            else "(없음: 위키는 읽기만 할 수 있다)",
            wiki_rule=WIKI_RULE_ON if LEARN_WIKI_WRITES else WIKI_RULE_OFF.format(max_chars=PROMPT_MAX_CHARS),
            max_chars=PROMPT_MAX_CHARS,
            readonly="\n".join(wiki_store.readonly_patterns()) or "(없음)")
        learner = PhoneAgent(
            "learner", None, "학습자", "전화 발신자의 프롬프트와 위키를 개선한다.", llm,
            tools=[*(wiki_store.WIKI_TOOLS if LEARN_WIKI_WRITES else wiki_store.WIKI_READ_TOOLS), PROMPT_EDIT_TOOL], tool_runner=_logged(_tool_runner(prompt_path, lines, _make_guard(rows, lines.text(), llm)), calls_log))
        raw, _, meta, lookups = learner._generate(
            messages=[{"role": "system", "content": learner.system_prompt},
                      {"role": "user", "content": request}],
            max_tokens=llm.plan_max_tokens, tools=learner.tools, final_tools=[],
            max_rounds=LEARN_TOOL_ROUNDS)
        result.update({"summary": _strip_think(raw).strip(), **meta, "tool_calls": lookups})
    except Exception as e:
        learner_failed = True
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[경고] 학습 실패: {result['error']}", flush=True)
        if (out / "wiki_audited").exists():  # 끝까지 못 간 회차의 반쯤 적용된 편집은 남기지 않는다
            try:
                _restore_state(out / "system_before.md", out / "wiki_audited", prompt_path)
                result["reverted_partial_edits"] = True
                print("[롤백] 학습이 끝나지 않아 이 회차의 편집을 되돌렸습니다", flush=True)
            except OSError as e2:
                result["revert_error"] = str(e2)
    finally:
        result["tool_calls_logged"] = calls_log
        if not (learner_failed and keep_on_fail):
            PENDING.unlink(missing_ok=True)  # 자동 실행은 어떤 실패에도 비운다 (매 통화 재시도 방지)

    try:
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(prompt_path, out / "system_after.md")
        _copy_wiki(out / "wiki_after")
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        (out / "learner.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[경고] 학습 결과 저장 실패: {e}", flush=True)
    return out


def maybe_learn(llm: LLMConfig, prompt_path: str = SYSTEM_PROMPT_PATH, every: int = 20) -> Path | None:
    if every > 0 and len(_pending()) >= every:
        return learn(llm, prompt_path)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="대기열 상태를 보거나 학습을 바로 실행")
    p.add_argument("--now", action="store_true", help="개수와 무관하게 지금 학습")
    p.add_argument("--caller-prompt", default=SYSTEM_PROMPT_PATH)
    p.add_argument("--plan-max-tokens", type=int, default=LLMConfig.plan_max_tokens, help="학습자 응답 max_tokens (사고 포함)")
    p.add_argument("--rollback", type=int, metavar="N", help="round N의 변경을 되돌린다(*_before로 복원, rolled_back 표시)")
    p.add_argument("--only", choices=["system", "wiki"], help="--rollback과 함께: 그 부분만 되돌린다")
    p.add_argument("--requeue", type=int, metavar="N", help="round N의 리뷰를 대기열에 다시 넣는다(실패한 회차를 다시 돌릴 때)")
    p.add_argument("--audit", action="store_true", help="고칠 수 있는 위키 문서에서 근거 없는 사실 주장을 찾는다(학습은 하지 않음)")
    p.add_argument("--fix", action="store_true", help="--audit와 함께: 찾은 줄을 지운다")
    args = p.parse_args()
    if args.rollback is not None:
        d = LEARN_DIR / f"round_{args.rollback:03d}"
        if not (d / "system_before.md").exists():
            raise SystemExit(f"{d}에 되돌릴 스냅샷이 없습니다")
        _restore(d, args.caller_prompt, "수동 롤백", tag="수동:", only=args.only)
        print(f"round_{args.rollback:03d}의 *_before로 복원했습니다" + (f" ({args.only}만)" if args.only else ""))
        return
    if args.requeue is not None:
        d = LEARN_DIR / f"round_{args.requeue:03d}"
        rows = json.loads((d / "reviews.json").read_text(encoding="utf-8"))["reviews"]
        LEARN_DIR.mkdir(parents=True, exist_ok=True)
        with PENDING.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"round_{args.requeue:03d}의 리뷰 {len(rows)}건을 대기열에 넣었습니다(대기열 {len(_pending())}건)")
        return
    if args.audit:
        llm = LLMConfig(plan_max_tokens=args.plan_max_tokens)
        for a in audit(llm, fix=args.fix, prompt_path=args.caller_prompt, rows=_pending()):
            print(f"{a['path']}: 근거 없는 주장 {len(a['claims'])}건")
            for c in a["claims"]:
                print(f"  - {c}")
            if a["removed_lines"]:
                print(f"  지운 줄 {len(a['removed_lines'])}개")
        return
    rows = _pending()
    print(f"대기열 {len(rows)}건")
    if args.now and rows:
        print(f"→ {learn(LLMConfig(plan_max_tokens=args.plan_max_tokens), args.caller_prompt, keep_on_fail=True)}")


if __name__ == "__main__":
    main()
