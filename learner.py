import argparse
import json
import random
import re
import secrets
import shutil
import string
from datetime import datetime
from pathlib import Path

import wiki_store
from agent import LEARN_TOOL_ROUNDS, LLMConfig, PhoneAgent, _strip_think
from prompts import (
    LEARN_REQUEST,
    SYSTEM_PROMPT_PATH,
)

LEARN_DIR = Path("agent_workspace/learning")
PENDING = LEARN_DIR / "pending.jsonl"
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
           "spam": review.get("spam"), "achieved": review.get("achieved"), "notes": review.get("notes", "")}
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


def _copy_wiki(dst: Path) -> None:
    shutil.copytree(wiki_store.WIKI_ROOT, dst, ignore=shutil.ignore_patterns(".*"))


def _tool_runner(prompt_path: str, lines: PromptLines):
    def run(name: str, arguments: str) -> str:
        if name == "prompt_edit":
            try:
                a = json.loads(arguments)
                action, line_id, new = a["action"], str(a["line_id"]), str(a.get("new_text") or "")
                text = Path(prompt_path).read_text(encoding="utf-8")
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                return "오류: 잘못된 인자"
            lines.sync(text)
            before = list(lines.rows)
            err = lines.edit(action, line_id, new)
            edited = lines.text().strip()
            if err is None and len(edited) < PROMPT_MIN_CHARS:
                err = f"오류: 결과가 {len(edited)}자입니다. 최소 {PROMPT_MIN_CHARS}자여야 합니다"
            if err is None and len(edited) > PROMPT_MAX_CHARS:
                err = f"오류: 결과가 {len(edited)}자입니다. {PROMPT_MAX_CHARS}자 이하로 줄이세요"
            if err:
                lines.rows = before
                return f"{err}\n현재 system.md:\n{lines.show()}"
            Path(prompt_path).write_text(edited + "\n", encoding="utf-8")
            return f"system.md를 고쳤습니다. 최신 줄 ID:\n{lines.show()}"
        if name in WRITE_TOOLS:
            try:
                a = json.loads(arguments)
                rel, _ = wiki_store._resolve(a.get("path", ""))
            except (ValueError, TypeError, AttributeError, wiki_store.WikiError) as e:
                return f"오류: {e}"
            if wiki_store.is_readonly(rel):
                return f"오류: {rel}은 고칠 수 없는 문서입니다"
        return wiki_store.run_tool(name, arguments)
    return run


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
        return f"신고율 {r.get('spam_rate')}, 달성률 {r.get('achieved_rate')}" if r else "알 수 없음"

    lines = []
    for i in range(max(1, n - k), n):
        d = LEARN_DIR / f"round_{i:03d}"
        try:
            summary = json.loads((d / "learner.json").read_text(encoding="utf-8")).get("summary", "")
        except (OSError, ValueError):
            continue
        rolled_back = (d / "rolled_back").exists()
        if i + 1 == n:
            after = "적용된 적 없음" if rolled_back else fmt(current)
        else:
            after = fmt(rates_of(LEARN_DIR / f"round_{i + 1:03d}"))
        note = " [롤백됨: 현재 프롬프트에 없음]" if rolled_back else ""
        lines.append(
            f"- round_{i:03d} 변경{note}: {' '.join(summary.split())[:300]}\n"
            f"  변경 전 묶음: {fmt(rates_of(d))} → 변경 후 묶음: {after}")
    return "\n".join(lines) or "(없음)"


def learn(llm: LLMConfig, prompt_path: str = SYSTEM_PROMPT_PATH) -> Path:
    rows = _pending()
    if not wiki_store.READONLY_FILE.exists():
        print(f"[경고] {wiki_store.READONLY_FILE}이 없어 모든 위키 문서를 고칠 수 있습니다", flush=True)
    n = len(list(LEARN_DIR.glob("round_*"))) + 1
    out = LEARN_DIR / f"round_{n:03d}"
    result: dict = {}
    try:
        out.mkdir(parents=True)
        shutil.copy(prompt_path, out / "system_before.md")
        _copy_wiki(out / "wiki_before")
        rates = {"achieved_rate": _rate(rows, "achieved"), "spam_rate": _rate(rows, "spam")}
        (out / "reviews.json").write_text(
            json.dumps({**rates, "reviews": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        lines = PromptLines(prompt_path)
        lines = PromptLines(load_caller_profile(prompt_path))
        reviews = "\n\n".join(f"- (목적: {r.get('purpose', '')}, {r['end_reason']}, 달성 {'예' if r['achieved'] else '아니오'}, 신고 {'예' if r['spam'] else '아니오'}) {r['notes']}" for r in rows)
        request = LEARN_REQUEST.format(
            n=len(rows), achieved_rate=rates["achieved_rate"], spam_rate=rates["spam_rate"], reviews=reviews, history=_history(n, rates), prompt=lines.render(),
            pages="\n".join(wiki_store.list_pages()) or "(없음)", max_chars=PROMPT_MAX_CHARS,
            readonly="\n".join(wiki_store.readonly_patterns()) or "(없음)")
        learner = PhoneAgent(
            "learner", None, "학습자", "전화 발신자의 프롬프트와 위키를 개선한다.", llm,
            tools=[*wiki_store.WIKI_TOOLS, PROMPT_EDIT_TOOL], tool_runner=_tool_runner(lines))
        raw, _, meta, lookups = learner._generate(
            messages=[{"role": "system", "content": learner.system_prompt},
                      {"role": "user", "content": request}],
            max_tokens=llm.plan_max_tokens, tools=learner.tools, final_tools=[],
            max_rounds=LEARN_TOOL_ROUNDS)
        result = {"summary": _strip_think(raw).strip(), **meta, "tool_calls": lookups}
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
        print(f"[경고] 학습 실패: {result['error']}", flush=True)
    finally:
        PENDING.unlink(missing_ok=True)  # 어떤 실패에도 비운다 (매 통화 재시도 방지)

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
    args = p.parse_args()
    rows = _pending()
    print(f"대기열 {len(rows)}건")
    if args.now and rows:
        print(f"→ {learn(LLMConfig(), args.caller_prompt)}")


if __name__ == "__main__":
    main()
