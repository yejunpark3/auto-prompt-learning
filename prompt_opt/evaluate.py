"""후보 하나를 수신자 한 명에게 걸어 보고(통화) 점수를 낸다(채점). 서버별 동시 요청 한도(슬롯)도 여기서 지킨다.

한 칸(cell) = 후보 × 수신자: 자연스러움(발신자 점수 s), 과제 진행(단계/4, 달성이면 1), 지어내기 기록(C1), 부자연스러운 발화·결함.
지어내기는 점수에서 깎지 않는다(성공만 본다): C1은 기록만 하고 C2(모델 검사)는 돌리지 않는다.
"""
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import personas
import wiki_store
from agent import LLMConfig
from phone_call import build_agents, save_record, simulate_call
from prompt_opt.candidates import SYSTEM, Candidate, materialize_wiki
from rl import fabrication
from rl.reward import _utterances, score_call

SERVER_ERRORS = {"error", "empty_response"}  # 서버 문제로 끊긴 통화: 점수를 낼 수 없고 같은 수신자로 다시 돌린다


class SlotPool:
    """서버(api_base)별 동시 요청 한도. 통화는 한 번에 요청 하나만 보내므로 통화 하나가 슬롯 하나를 쓴다."""

    def __init__(self, capacity: dict[str | None, int], default: int = 1):
        self.capacity, self.default = dict(capacity), default
        self._used: dict[str | None, int] = {}
        self._busy: dict[str | None, float] = {}  # 슬롯·초
        self._cond = threading.Condition()
        self._t0 = time.monotonic()

    def cap(self, base: str | None) -> int:
        return self.capacity.get(base, self.default)

    @contextmanager
    def hold(self, base: str | None, n: int = 1):
        n = max(1, min(n, self.cap(base)))
        with self._cond:
            while self._used.get(base, 0) + n > self.cap(base):
                self._cond.wait()
            self._used[base] = self._used.get(base, 0) + n
        t = time.monotonic()
        try:
            yield
        finally:
            with self._cond:
                self._used[base] -= n
                self._busy[base] = self._busy.get(base, 0.0) + n * (time.monotonic() - t)
                self._cond.notify_all()

    def utilization(self) -> dict[str, float]:
        elapsed = max(1e-9, time.monotonic() - self._t0)
        return {str(b): round(s / (self.cap(b) * elapsed), 3) for b, s in self._busy.items()}


def _quiet(*_):
    return None


class Evaluator:
    def __init__(self, run_dir: Path, purpose: str, call_llm: LLMConfig, judge_llm: LLMConfig, slots: SlotPool,
                 readonly_docs: dict[str, str], patterns: list[str], max_turns: int = 14, retries: int = 2):
        self.run_dir, self.purpose = run_dir, purpose
        self.call_llm, self.judge_llm, self.slots = call_llm, judge_llm, slots
        self.readonly_docs, self.patterns = readonly_docs, patterns
        self.max_turns, self.retries = max_turns, retries
        self._lock = threading.Lock()

    def wiki_dir(self, cand: Candidate) -> Path:
        with self._lock:
            return materialize_wiki(cand, self.run_dir / "wiki" / cand.cid, self.readonly_docs, self.patterns)

    def _call(self, cand: Candidate, uuid: str) -> tuple[dict, int]:
        """통화를 한다. 서버 오류로 끊기면 같은 수신자로 retries번까지 다시 한다. (기록, 시도 수)"""
        wdir = self.wiki_dir(cand)
        for attempt in range(1, self.retries + 2):
            persona = personas.get_persona(uuid)
            with self.slots.hold(self.call_llm.api_base):
                with wiki_store.using_root(wdir):  # 이 스레드의 위키 = 이 후보의 위키(읽기 전용 문서 + 후보의 문서)
                    caller, callee = build_agents(cand.text(SYSTEM), persona, self.purpose, self.call_llm, wiki=True)
                    record = simulate_call(caller, callee, self.purpose, max_turns=self.max_turns,
                                           on_turn=_quiet, on_plan=_quiet)
            if record["end_reason"] not in SERVER_ERRORS:
                break
        return record, attempt

    def run_cell(self, cand: Candidate, recipient: dict) -> dict:
        t0 = time.monotonic()
        record, attempts = self._call(cand, recipient["uuid"])
        cell: dict = {"cid": cand.cid, "uuid": recipient["uuid"], "end_reason": record["end_reason"],
                      "excluded": None, "attempts": attempts, "n_turns": record["num_turns"],
                      "natural": None, "task": None, "stage": None, "achieved": None,
                      "c1": None, "c1_items": [], "zero_utts": [], "defects": []}
        try:
            with self.slots.hold(self.judge_llm.api_base):
                result = score_call(record, self.judge_llm, judge=True, check_fabrication=False)
            cell["excluded"] = result["excluded"]
            if result["complete"]:
                cell["stage"], cell["achieved"] = result["stage"], bool(result["achieved"])
                cell["task"] = 1.0 if cell["achieved"] else result["stage"] / 4
                cell["natural"] = result["caller"]["s"]
                said = [t for _, r, t in _utterances(record) if r == "caller"]
                cell["zero_utts"] = [u for u, j in zip(said, result["caller"]["j"]) if j == 0][:3]
                cell["defects"] = sorted({d["kind"] for d in result["defects"]["caller"]})
        except Exception as e:  # 한 통화의 채점 실패가 실행을 멈추지 않게 한다(그 칸은 빈칸)
            cell["excluded"] = f"score_error: {type(e).__name__}: {e}"
        try:  # 기록용(점수에는 쓰지 않는다): 근거 문서에 없는 숫자·URL·이메일을 말했는가
            numbers, contacts = fabrication.ungrounded(record, list(self.readonly_docs.values()))
            cell["c1"], cell["c1_items"] = bool(numbers or contacts), [*numbers, *contacts][:5]
        except Exception:
            pass
        cell["latency_s"] = round(time.monotonic() - t0, 1)
        record["gepa"] = {k: v for k, v in cell.items()}
        cell["call_file"] = str(save_record(record, str(self.run_dir / "calls" / cand.cid)))
        return cell
