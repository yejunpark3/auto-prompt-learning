"""GEPA 실행기: 세 전략(hillclimb | gepa | ours)을 같은 변이 연산자·평가·예산으로 돌려 비교한다. 시나리오를 모른다.

    uv run -m prompt_opt.runner --run NAME --strategy ours --purpose "..." --budget 60 [--metric natural|task]
    uv run -m prompt_opt.runner --run NAME --report                 # 결과표(후보별 D 평균, 파레토 프런티어, best)
    uv run -m prompt_opt.runner --run NAME --apply c007 [--yes]     # diff를 보여 주고, --yes면 prompt/·wiki/에 쓴다

실행 중에는 저장소의 prompt/system.md와 wiki/를 건드리지 않는다. 모든 상태는 agent_workspace/gepa/<run>/에 있고,
같은 --run으로 다시 실행하면 이어서 돈다(Ctrl-C·서버 중단 후에도 통화 단위로 재개).

전략:
  hillclimb  지금 러너(learner.py)의 방식: 제안 → D 전체 평가 → 주 점수 평균이 hill_tol 넘게 떨어지면 되돌림.
  gepa       순수 GEPA: 주 점수 하나, 수신자별 파레토 후보 풀, 미니배치 합이 오를 때만 채택, 문서 통째 병합.
  ours       ① 짝지은 순차 검정 채택 ② 다목적(자연스러움·과제) 파레토 ③ 줄 단위 병합 ④ 슬롯 스케줄러.
점수: 통화마다 자연스러움(natural, 발신자 점수)과 과제 진행(task, 단계/4)을 낸다. --metric이 두 목표의 우선순위를 정한다.
지어내기는 점수에서 깎지 않는다(성공만 본다). C1 비율만 기록한다.
"""
import argparse
import difflib
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import learner
from prompt_opt import candidates as C
from prompt_opt import choose as S
from prompt_opt.evaluate import SERVER_ERRORS
from prompt_opt.reflect import pick_target, strip_quotes

RUN_ROOT = Path("agent_workspace/gepa")
STRATEGIES = ("hillclimb", "gepa", "ours")
LABEL = {"natural": "자연스러움", "task": "단계"}
MAX_IDLE = 6  # 변경 없는 반성이 이만큼 이어지면 멈춘다(예산을 쓰지 않아 끝나지 않는 루프 방지)


class BudgetExhausted(Exception):
    pass


class ServerDown(Exception):
    pass


@dataclass
class Config:
    strategy: str
    purpose: str
    metric: str = "natural"
    seeds: list[int] = field(default_factory=lambda: [41, 42])
    per_seed: int = 5
    budget: int = 60
    minibatch: int = 3
    max_turns: int = 14
    rng_seed: int = 0
    system_max: int = learner.PROMPT_MAX_CHARS
    wiki_max: int = 2500
    workers: int = 1
    max_merges: int = 5
    hill_tol: float = 0.1
    seq_start: int = 2
    seq_step: int = 2
    seq_max: int = 6
    allow_new_docs: bool = True
    prefetch: bool = False  # ours 전용: 다음 반성을 현재 후보를 시험하는 동안 미리 돌린다(느린 로컬 모델의 대기 시간을 겹친다)

    # 이어 돌릴 때 달라지면 안 되는 값(알고리즘·평가 대상을 바꾼다). 예산·병렬 수는 바꿔도 된다.
    FIXED = ("strategy", "purpose", "metric", "seeds", "per_seed", "max_turns", "system_max", "wiki_max", "minibatch",
             "rng_seed", "hill_tol", "seq_start", "seq_step", "seq_max", "allow_new_docs", "prefetch")

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _dump_rng(state) -> list:
    v, internal, gauss = state
    return [v, list(internal), gauss]


def _load_rng(d: list):
    return (d[0], tuple(d[1]), d[2])


def _fmt(v: float | None) -> str:
    return "?" if v is None else f"{v:.2f}"


class _Prefetch:
    """반성 하나를 백그라운드에서 돌린다. 데몬 스레드라 Ctrl-C로 나갈 때 끝나길 기다리지 않는다(결과는 버려지고,
    저장된 선택으로 다음 실행에서 다시 돌린다). 예외는 result에 담아 두었다가 받는 쪽에서 다시 던진다."""

    def __init__(self, fn):
        self.done, self.result, self.seconds = threading.Event(), None, 0.0

        def run() -> None:
            t = time.monotonic()
            try:
                self.result = fn()
            except BaseException as e:  # noqa: BLE001
                self.result = e
            finally:
                self.seconds = time.monotonic() - t
                self.done.set()

        threading.Thread(target=run, daemon=True, name="gepa-prefetch").start()


class Run:
    def __init__(self, run_dir: Path, cfg: Config, state: dict, evaluator, reflect_fn):
        """evaluator.run_cell(cand, recipient) -> cell, evaluator.wiki_dir(cand) -> Path,
        reflect_fn(parent, calls, *, cid, iteration, target, history, wiki_dir, used_ids, readonly) -> Reflection."""
        self.dir, self.cfg, self.state = Path(run_dir), cfg, state
        self.evaluator, self.reflect_fn = evaluator, reflect_fn
        self.cands = {cid: C.Candidate.from_json(d) for cid, d in state["candidates"].items()}
        self.recip = {r["uuid"]: r for r in state["recipients"]}
        self.D = [r["uuid"] for r in state["recipients"]]
        self.rng = random.Random()
        self.rng.setstate(_load_rng(state["rng"]))
        self.used_ids: set[str] = set().union(*(c.ids() for c in self.cands.values()))
        self.objectives = ["natural", "task"] if cfg.metric == "natural" else ["task", "natural"]
        self.primary = self.objectives[0]
        self.lock = threading.RLock()
        self.in_flight = 0
        self.consec_err = 0
        self._t0, self._base = time.monotonic(), state.get("elapsed_s", 0.0)
        state.setdefault("next_cid", len(self.cands))   # 후보 번호 예약 카운터(선행 반성이 번호를 미리 잡는다)
        state.setdefault("prefetch", None)             # 선행 반성의 선택(부모·순서·대상·번호). 결과는 메모리에만 있다
        state.setdefault("prefetch_stats", {"used": 0, "reflect_s": 0.0, "wait_s": 0.0})
        self._pf: _Prefetch | None = None
        # 서버 오류로 점수를 못 낸 칸은 다시 돌릴 수 있게 비운다
        for cells in state["scores"].values():
            for u in [u for u, c in cells.items() if c.get("end_reason") in SERVER_ERRORS]:
                del cells[u]

    # ───────────── 생성·저장·불러오기 ─────────────

    @classmethod
    def create(cls, run_dir, cfg: Config, system_text: str, editable_wiki: dict[str, str], readonly_docs: dict[str, str],
               patterns: list[str], recipients: list[dict], evaluator, reflect_fn) -> "Run":
        used: set[str] = set()
        c0 = C.seed_candidate("c000", system_text, editable_wiki, used)
        if err := C.size_error(c0, cfg.system_max, cfg.wiki_max):
            raise SystemExit(f"첫 후보가 크기 상한을 넘습니다: {err} (--system-max/--wiki-max로 조정)")
        state = {"version": 1, "config": asdict(cfg), "recipients": recipients, "readonly_docs": readonly_docs,
                 "patterns": patterns, "candidates": {"c000": c0.to_json()}, "scores": {}, "pool": [], "current": "c000", "next_cid": 1,
                 "calls_used": 0, "iter": 0, "rng": _dump_rng(random.Random(cfg.rng_seed).getstate()), "pending": None,
                 "history": [], "merges_done": 0, "merge_tried": [], "want_merge": False, "idle": 0, "elapsed_s": 0.0,
                 "stopped": None, "started_at": datetime.now().isoformat(timespec="seconds")}
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        run = cls(Path(run_dir), cfg, state, evaluator, reflect_fn)
        run.save()
        return run

    @staticmethod
    def load_state(run_dir) -> dict:
        return json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8"))

    def save(self) -> None:
        with self.lock:
            st = self.state
            st["config"] = asdict(self.cfg)  # 이어 돌릴 때 늘린 예산 등이 반영되도록(비교 스크립트가 이 값을 읽는다)
            st["candidates"] = {cid: c.to_json() for cid, c in self.cands.items()}
            st["rng"] = _dump_rng(self.rng.getstate())
            st["elapsed_s"] = round(self._base + time.monotonic() - self._t0, 1)
            tmp = self.dir / "state.json.tmp"
            tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.dir / "state.json")

    # ───────────── 점수표 ─────────────

    def new_cid(self) -> str:
        """다음 후보 번호(예약하지 않는다)."""
        return f"c{self.state['next_cid']:03d}"

    def reserve_cid(self) -> str:
        cid = self.new_cid()
        self.state["next_cid"] += 1
        return cid

    def values(self, cid: str, obj: str) -> dict[str, float | None]:
        cells = self.state["scores"].get(cid, {})
        return {u: (cells.get(u) or {}).get(obj) for u in self.D}

    def mean(self, cid: str, obj: str, over: list[str] | None = None) -> float | None:
        v = self.values(cid, obj)
        xs = [v[u] for u in (over or self.D) if v.get(u) is not None]
        return statistics.fmean(xs) if xs else None

    def means(self, cid: str) -> dict[str, float | None]:
        return {o: self.mean(cid, o) for o in self.objectives}

    def rank_key(self, cid: str) -> tuple:
        return S.lexi(self.means(cid), self.objectives)

    def complete(self, cid: str) -> bool:
        cells = self.state["scores"].get(cid, {})
        return all(u in cells for u in self.D)

    def budget_left(self) -> int:
        return self.cfg.budget - self.state["calls_used"]

    def min_cost(self) -> int:
        c = self.cfg
        return {"hillclimb": len(self.D), "gepa": c.minibatch, "ours": c.seq_start}[c.strategy]

    def say(self, msg: str) -> None:
        print(f"[{self.cfg.strategy}] {msg}", flush=True)

    # ───────────── 평가 ─────────────

    def eval_cells(self, cid: str, uuids: list[str]) -> None:
        """cid를 uuids에서 아직 평가하지 않은 수신자에게 건다(예산·서버 오류 처리, workers만큼 병렬)."""
        todo = [u for u in uuids if u not in self.state["scores"].get(cid, {})]
        if not todo:
            return
        cand = self.cands[cid]

        def one(u: str) -> None:
            with self.lock:
                if self.state["calls_used"] + self.in_flight >= self.cfg.budget:
                    raise BudgetExhausted()
                self.in_flight += 1
            try:
                cell = self.evaluator.run_cell(cand, self.recip[u])
            finally:
                with self.lock:
                    self.in_flight -= 1
            with self.lock:
                self.state["scores"].setdefault(cid, {})[u] = cell
                if cell["end_reason"] in SERVER_ERRORS:  # 점수를 낼 수 없는 통화: 예산에 넣지 않고 연속 오류만 센다
                    self.consec_err += 1
                else:
                    self.consec_err = 0
                    self.state["calls_used"] += 1
                self.save()
                self.say(f"{cid} × {self.recip[u]['name']}: 자연스러움 {_fmt(cell.get('natural'))}, 단계 {cell.get('stage')}, "
                         f"{cell['end_reason']} (통화 {self.state['calls_used']}/{self.cfg.budget})")
                if self.consec_err >= 3:
                    raise ServerDown("서버 오류로 끊긴 통화가 3번 연속입니다. 서버를 확인하고 같은 --run으로 다시 실행하세요")

        if self.cfg.workers > 1 and len(todo) > 1:
            with ThreadPoolExecutor(self.cfg.workers) as ex:
                for f in [ex.submit(one, u) for u in todo]:
                    f.result()
        else:
            for u in todo:
                one(u)

    def call_info(self, cid: str, u: str) -> dict | None:
        cell = self.state["scores"].get(cid, {}).get(u)
        if not cell or not cell.get("call_file") or cell["end_reason"] in SERVER_ERRORS:
            return None
        return {"path": cell["call_file"], "end_reason": cell["end_reason"], "achieved": cell.get("achieved"),
                "purpose": self.cfg.purpose, "cell": cell, "name": self.recip[u]["name"]}

    # ───────────── 선택 ─────────────

    def front(self) -> dict[str, int]:
        """파레토 프런티어 {후보: 가중치}. gepa는 수신자 키(주 점수), ours는 수신자 × 목표 키."""
        pool = self.state["pool"]
        if self.cfg.strategy == "gepa":
            keys = list(self.D)
            table = {c: self.values(c, self.primary) for c in pool}
        else:
            keys = [f"{u}|{o}" for u in self.D for o in self.objectives]
            table = {c: {f"{u}|{o}": v for o in self.objectives for u, v in self.values(c, o).items()} for c in pool}
        return S.pareto_front(table, keys)

    def choose_parent(self) -> str:
        if self.cfg.strategy == "hillclimb":
            return self.state["current"]
        return S.sample_parent(self.rng, self.front())

    def history_for(self, cid: str) -> str:
        """부모의 조상 사슬(변경 요약과 전체 평균)과 이 부모에서 기각된 시도. 따옴표 문구는 지운다."""
        chain, c = [], cid
        while c:
            chain.append(c)
            ps = self.cands[c].parents
            c = ps[0] if ps else None
        lines = []
        for c in reversed(chain):
            cand, m = self.cands[c], self.means(c)
            avg = ", ".join(f"{LABEL[o]} {_fmt(m[o])}" for o in ("natural", "task"))
            if cand.kind == "seed":
                lines.append(f"- {c} (처음 문서): {avg}")
            else:
                lines.append(f"- {c} ({'병합' if cand.kind == 'merge' else '반성'}, 대상 {cand.target}): "
                             f"{strip_quotes(cand.summary)[:250]} → 전체 평균 {avg}")
        rejected = [h for h in self.state["history"] if h.get("parent") == cid and h["decision"] == "rejected" and h.get("child")]
        for h in rejected[-3:]:
            d = h.get("delta") or {}
            ch = ", ".join(f"{LABEL[o]} {d[o]:+.2f}" for o in ("natural", "task") if d.get(o) is not None)
            lines.append(f"- 기각됨 {h['child']} (대상 {h.get('target')}): {strip_quotes(self.cands[h['child']].summary)[:200]} → 시험한 수신자 평균 변화 {ch or '?'}")
        return "\n".join(lines)

    # ───────────── 제안 ─────────────

    def _write_iter(self, n: int, data: dict) -> None:
        (self.dir / f"iter_{n:03d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def propose(self) -> None:
        """다음 후보를 하나 제안한다. 원자적이다: 도중에 예외가 나면(서버 중단·Ctrl-C) 난수와 상태를 되돌려, 이어 돌릴 때
        중단 없는 실행과 같은 부모·수신자 순서를 다시 뽑는다(반쯤 쓴 난수가 저장되면 결과가 갈라진다).

        병합 후보(LLM 없음)가 있으면 반성을 기다리지 않고 먼저 시험한다. 선행 반성이 있으면 그 결과를 받아 쓴다."""
        st = self.state
        snap_rng = self.rng.getstate()
        snap = {k: json.loads(json.dumps(st.get(k))) for k in
                ("want_merge", "merge_tried", "merges_done", "iter", "idle", "next_cid", "prefetch", "prefetch_stats")}
        try:
            if st.get("want_merge") and self.cfg.strategy in ("gepa", "ours"):
                st["want_merge"] = False
                if st["merges_done"] < self.cfg.max_merges and self.propose_merge():
                    self._maybe_prefetch()
                    return
            if self._pf is not None or st["prefetch"]:
                self._take_prefetch()
            else:
                plan = self._select_reflect()
                self._finish_reflect(plan, self._do_reflect(self._reflect_inputs(plan)))
            if st["pending"]:
                self._maybe_prefetch()
        except BaseException:
            self.rng.setstate(snap_rng)
            st.update(snap)
            raise

    # 반성은 세 단계다. 선택(난수·번호 예약, 메인 스레드) → 반성(LLM, 메인 또는 선행 스레드) → 마무리(상태 반영, 메인 스레드).
    # 난수와 상태를 건드리는 것은 메인 스레드뿐이라 스레드 타이밍이 결과를 바꾸지 않는다.
    def _select_reflect(self) -> dict:
        st, cfg = self.state, self.cfg
        parent = self.choose_parent()
        order = self.rng.sample(self.D, len(self.D))
        st["iter"] += 1  # 반복 번호와 후보 번호를 예약한다(선행 반성이 있어도 겹치지 않게)
        n = st["iter"]
        return {"iter": n, "parent": parent, "order": order, "cid": self.reserve_cid(),
                "target": pick_target(self.cands[parent], n, cfg.allow_new_docs)}

    def _reflect_inputs(self, plan: dict) -> dict:
        """반성에 넘길 자료를 메인 스레드에서 모은다(반성 스레드는 상태를 읽지 않는다)."""
        st, parent = self.state, plan["parent"]
        pc = self.cands[parent]
        calls = [ci for u in plan["order"][:self.cfg.minibatch] if (ci := self.call_info(parent, u))]
        self.say(f"반복 {plan['iter']}: 부모 {parent}, 대상 {plan['target']}, 미니배치 {len(calls)}통화 → 반성")
        return {"parent": pc, "calls": calls, "kw": dict(
            cid=plan["cid"], iteration=plan["iter"], target=plan["target"], history=self.history_for(parent),
            wiki_dir=self.evaluator.wiki_dir(pc), used_ids=self.used_ids,
            readonly="\n".join([*st["patterns"], *st["readonly_docs"]]))}

    def _do_reflect(self, inputs: dict):
        return self.reflect_fn(inputs["parent"], inputs["calls"], **inputs["kw"])

    def _finish_reflect(self, plan: dict, refl) -> None:
        st = self.state
        n, parent, cid, target = plan["iter"], plan["parent"], plan["cid"], plan["target"]
        info = {"summary": refl.summary, "edits": refl.edits, "log": refl.log, "error": refl.error, "note": refl.note}
        if refl.child is None:
            if st["next_cid"] == C.num(cid) + 1:  # 가장 최근에 예약한 번호면 반납한다(번호에 빈틈을 만들지 않는다)
                st["next_cid"] -= 1
            st["idle"] = st.get("idle", 0) + 1
            st["history"].append({"iter": n, "kind": "reflect", "parent": parent, "child": None, "target": target,
                                  "decision": "no_child", "note": refl.note or refl.error, "calls_used": st["calls_used"]})
            self._write_iter(n, {"iter": n, "parent": parent, "target": target, "reflect": info, "decision": "no_child"})
            self.say(f"반복 {n}: 새 후보 없음 ({refl.note or refl.error})")
            self.save()
            return
        st["idle"] = 0
        self.cands[cid] = refl.child
        self.used_ids |= refl.child.ids()
        st["pending"] = {"iter": n, "kind": "reflect", "parent": parent, "parents": [parent], "ref": parent, "child": cid,
                         "order": plan["order"], "target": target, "reflect": info}
        self.save()
        self.say(f"반복 {n}: {cid} 제안 — {strip_quotes(refl.summary)[:120]}")

    # ───────────── 선행 반성 (④) ─────────────

    def _start_prefetch(self, plan: dict) -> None:
        inputs = self._reflect_inputs(plan)
        self._pf = _Prefetch(lambda: self._do_reflect(inputs))

    def _maybe_prefetch(self) -> None:
        """현재 후보를 시험하는 동안 다음 반성을 미리 돌린다(통화 서버가 반성을 기다리며 놀지 않게). 한 번에 하나만 앞서 간다.
        부모는 지금의 프런티어에서 고른다(시험 중인 후보는 아직 풀에 없다). 선택은 여기서 끝내 상태에 저장한다."""
        st, cfg = self.state, self.cfg
        if not (cfg.prefetch and cfg.strategy == "ours") or self._pf is not None or st["prefetch"]:
            return
        if st.get("idle", 0) >= MAX_IDLE or self.budget_left() < 2 * self.min_cost():
            return
        plan = self._select_reflect()
        st["prefetch"] = plan
        self._start_prefetch(plan)
        self.save()
        self.say(f"선행 반성 시작: 반복 {plan['iter']}, 부모 {plan['parent']}, 대상 {plan['target']}")

    def _take_prefetch(self) -> None:
        st = self.state
        plan = st["prefetch"]
        if self._pf is None:  # 이어 돌리는 중이면 저장된 선택으로 반성을 다시 시작한다
            self._start_prefetch(plan)
        pf, t = self._pf, time.monotonic()
        pf.done.wait()
        stats = st["prefetch_stats"]
        stats["used"] += 1
        stats["reflect_s"] = round(stats["reflect_s"] + pf.seconds, 1)
        stats["wait_s"] = round(stats["wait_s"] + (time.monotonic() - t), 1)
        self._pf, st["prefetch"] = None, None
        if isinstance(pf.result, BaseException):
            raise pf.result
        self._finish_reflect(plan, pf.result)

    def propose_merge(self) -> bool:
        """공통 조상이 있는 프런티어 쌍을 합친 후보를 만든다(LLM 호출 없음). 만들 수 있는 쌍이 없으면 False."""
        st, cfg = self.state, self.cfg
        front = sorted(self.front(), key=C.num)
        pairs = [(a, b) for i, a in enumerate(front) for b in front[i + 1:] if f"{a}+{b}" not in st["merge_tried"]]
        self.rng.shuffle(pairs)
        mode = "module" if cfg.strategy == "gepa" else "line"
        n = st["iter"] + 1  # 병합을 만들 때만 번호를 가져간다(st["iter"] 갱신은 아래)
        for a, b in pairs:
            st["merge_tried"].append(f"{a}+{b}")
            prefer = "a" if self.rank_key(a) >= self.rank_key(b) else "b"
            child = C.merge(self.cands, a, b, prefer, mode, self.new_cid(), n, cfg.system_max, cfg.wiki_max)
            if child is None:
                continue
            self.reserve_cid()
            self.cands[child.cid] = child
            st["iter"], st["merges_done"] = n, st["merges_done"] + 1
            order = self.rng.sample(self.D, len(self.D))
            ref = a if prefer == "a" else b
            st["pending"] = {"iter": n, "kind": "merge", "parent": ref, "parents": [a, b], "ref": ref, "child": child.cid,
                             "order": order, "target": None, "reflect": {"summary": child.summary}}
            self.save()
            self.say(f"반복 {n}: 병합 {child.cid} = {a}+{b} ({mode}, 기준 {ref})")
            return True
        self.save()
        return False

    # ───────────── 시험·결정 ─────────────

    def _delta_info(self, child: str, ref: str, over: list[str]) -> dict:
        out = {}
        for o in self.objectives:
            d = S.paired_deltas(self.values(child, o), self.values(ref, o), over)
            out[o] = round(statistics.fmean(d), 4) if d else None
        return out

    def test_hillclimb(self, p: dict) -> tuple[str, dict]:
        child, ref = p["child"], p["ref"]
        self.eval_cells(child, self.D)
        d = S.paired_deltas(self.values(child, self.primary), self.values(ref, self.primary), self.D)
        m, _, n = S.mean_se(d)
        ok = n > 0 and m > -self.cfg.hill_tol
        return ("accepted" if ok else "rejected"), {"delta": self._delta_info(child, ref, self.D), "n": n, "tested_uuids": list(self.D)}

    def test_gepa(self, p: dict) -> tuple[str, dict]:
        """미니배치(minibatch명)에서 자식의 주 점수 합이 부모 합보다 클 때만 채택한다(병합은 두 부모 중 큰 쪽 이상)."""
        child, mb = p["child"], p["order"][:self.cfg.minibatch]
        self.eval_cells(child, mb)
        vc = self.values(child, self.primary)
        best_parent, sc, n = -float("inf"), 0.0, 0
        for r in p["parents"]:
            vp = self.values(r, self.primary)
            common = [u for u in mb if vc[u] is not None and vp[u] is not None]
            if common and sum(vp[u] for u in common) > best_parent:
                best_parent, sc, n = sum(vp[u] for u in common), sum(vc[u] for u in common), len(common)
        ok = n > 0 and (sc >= best_parent if p["kind"] == "merge" else sc > best_parent)
        return ("accepted" if ok else "rejected"), {"mb_sum": [round(sc, 3), round(best_parent, 3)], "n": n,
                                                    "delta": self._delta_info(child, p["ref"], mb), "tested_uuids": mb}

    def test_ours(self, p: dict) -> tuple[str, dict]:
        """같은 수신자 짝비교를 2명씩 늘려 가며 순차 검정한다. 뚜렷이 나아지면 채택, 가망이 없으면 조기 기각, 상한까지 못 가리면 기각."""
        cfg, child, ref, order = self.cfg, p["child"], p["ref"], p["order"]
        upto = min(cfg.seq_start, len(order))
        while True:
            batch = order[:upto]
            self.eval_cells(child, batch)
            v = S.verdict({o: S.paired_deltas(self.values(child, o), self.values(ref, o), batch) for o in self.objectives})
            if v != "more" or upto >= min(cfg.seq_max, len(order)):
                break
            upto = min(upto + cfg.seq_step, cfg.seq_max, len(order))
        return ("accepted" if v == "accept" else "rejected"), {
            "verdict": v if v != "more" else "undecided", "tested_uuids": order[:upto],
            "delta": self._delta_info(child, ref, order[:upto])}

    def settle(self, p: dict) -> None:
        """대기 중인 후보를 시험해 채택·기각하고 기록한다. 중간에 멈춰도 같은 --run으로 이어진다(칸 단위 캐시)."""
        st = self.state
        if "decision" not in p:
            p["decision"], p["extra"] = {"hillclimb": self.test_hillclimb, "gepa": self.test_gepa, "ours": self.test_ours}[self.cfg.strategy](p)
            self.save()
        if p["decision"] == "accepted":
            self.eval_cells(p["child"], self.D)  # 나머지 수신자까지 평가해 풀에 넣는다
        entry = {"iter": p["iter"], "kind": p["kind"], "parent": p["parent"], "parents": p["parents"], "child": p["child"],
                 "target": p.get("target"), "decision": p["decision"], **p["extra"], "calls_used": st["calls_used"]}
        st["history"].append(entry)
        st["pending"] = None
        if p["decision"] == "accepted":
            if p["child"] not in st["pool"]:
                st["pool"].append(p["child"])
            if self.cfg.strategy == "hillclimb":
                st["current"] = p["child"]
            elif p["kind"] == "reflect":
                st["want_merge"] = True
        self._write_iter(p["iter"], {**entry, "reflect": p["reflect"], "means": self.means(p["child"])})
        m = self.means(p["child"])
        self.say(f"반복 {p['iter']}: {p['child']} {p['decision']} {p['extra']} (D 평균 {LABEL['natural']} {_fmt(m['natural'])}, {LABEL['task']} {_fmt(m['task'])})")
        self.save()

    # ───────────── 실행 ─────────────

    def run_loop(self, max_iters: int | None = None) -> str:
        """끝날 때까지(예산·정체·서버 중단) 반복한다. 멈춘 이유를 돌려준다."""
        st, done = self.state, 0
        try:
            if st["prefetch"] and self._pf is None:  # 이어 돌리는 중: 저장된 선택으로 반성을 바로 다시 시작해 시험과 겹친다
                self._start_prefetch(st["prefetch"])
            if "c000" not in st["pool"]:
                self.say(f"예산 {self.cfg.budget}통화, 수신자 {len(self.D)}명, 전략 {self.cfg.strategy}, 우선순위 {self.objectives}")
                self.eval_cells("c000", self.D)
                st["pool"].append("c000")
                self.save()
            while True:
                if st["pending"]:
                    self.settle(st["pending"])
                    done += 1
                    continue
                if self.budget_left() < self.min_cost():
                    st["stopped"] = "budget"
                    break
                if st.get("idle", 0) >= MAX_IDLE:
                    st["stopped"] = "stalled"
                    break
                if max_iters is not None and done >= max_iters:
                    st["stopped"] = "max_iters"
                    break
                self.propose()
                if not st["pending"]:
                    done += 1
        except BudgetExhausted:
            st["stopped"] = "budget"
        except ServerDown as e:
            st["stopped"] = "server_down"
            self.say(str(e))
        finally:
            self.save()
        return st["stopped"]

    # ───────────── 보고 ─────────────

    def report(self) -> dict:
        st = self.state
        rows = []
        for cid in st["pool"]:
            cells = [st["scores"].get(cid, {}).get(u) for u in self.D]
            c1 = [c["c1"] for c in cells if c and c.get("c1") is not None]
            m = self.means(cid)
            rows.append({"cid": cid, "kind": self.cands[cid].kind, "parents": self.cands[cid].parents,
                         "natural": m["natural"], "task": m["task"],
                         "n": sum(1 for c in cells if c and c.get(self.primary) is not None),
                         "c1_rate": round(sum(c1) / len(c1), 3) if c1 else None})
        best = max(st["pool"], key=self.rank_key) if st["pool"] else None
        # 미니배치(시험)를 통과해 채택됐지만 D 전체로는 부모보다 못한 반성 후보의 비율
        accepted = [h for h in st["history"] if h["decision"] == "accepted" and h["kind"] == "reflect"]
        worse = [h for h in accepted if (cm := self.mean(h["child"], self.primary)) is not None
                 and (pm := self.mean(h["parent"], self.primary)) is not None and cm < pm]
        decided = [h for h in st["history"] if h["decision"] in ("accepted", "rejected")]
        calls_per_accept = round(st["calls_used"] / len([h for h in decided if h["decision"] == "accepted"]), 1) \
            if any(h["decision"] == "accepted" for h in decided) else None
        return {"strategy": self.cfg.strategy, "metric": self.cfg.metric, "stopped": st["stopped"], "calls_used": st["calls_used"],
                "budget": self.cfg.budget, "elapsed_s": st["elapsed_s"], "iterations": st["iter"], "candidates": len(self.cands),
                "accepted": len(accepted), "accepted_but_worse_on_D": len(worse), "merges_tried": st["merges_done"],
                "merges_accepted": sum(1 for h in decided if h["kind"] == "merge" and h["decision"] == "accepted"),
                "calls_per_accept": calls_per_accept, "calls_per_hour": round(st["calls_used"] / (st["elapsed_s"] / 3600), 2) if st["elapsed_s"] else None,
                "prefetch": st["prefetch_stats"], "slot_utilization": st.get("slot_utilization"), "best": best, "table": rows,
                "front": self.front() if self.cfg.strategy != "hillclimb" and st["pool"] else {}}

    def print_report(self) -> dict:
        r = self.report()
        print(f"\n== {self.cfg.strategy} ({r['calls_used']}/{r['budget']}통화, {r['elapsed_s'] / 3600:.1f}시간, 종료 이유: {r['stopped']}) ==")
        print(f"{'후보':6} {'종류':8} {'부모':14} {LABEL['natural']:>8} {LABEL['task']:>6} {'n':>3} {'C1(지어냄)':>9}")
        for t in r["table"]:
            mark = " ★" if t["cid"] == r["best"] else ""
            print(f"{t['cid']:6} {t['kind']:8} {','.join(t['parents']) or '-':14} {_fmt(t['natural']):>8} {_fmt(t['task']):>6} "
                  f"{t['n']:>3} {_fmt(t['c1_rate']):>9}{mark}")
        print(f"반복 {r['iterations']}, 후보 {r['candidates']}, 채택 {r['accepted']}(D 전체로는 부모보다 못함 {r['accepted_but_worse_on_D']}), "
              f"병합 시도 {r['merges_tried']}/채택 {r['merges_accepted']}, 채택 1건당 통화 {r['calls_per_accept']}")
        pf = r["prefetch"]
        print(f"속도: 시간당 통화 {r['calls_per_hour']}" + (f", 슬롯 사용률 {r['slot_utilization']}" if r["slot_utilization"] else "")
              + (f", 선행 반성 {pf['used']}회(반성 {pf['reflect_s'] / 60:.1f}분 중 {max(0.0, pf['reflect_s'] - pf['wait_s']) / 60:.1f}분이 시험과 겹침)"
                 if pf["used"] else ""))
        if r["front"]:
            print("프런티어(이긴 키 수):", r["front"])
        return r

    def write_best(self) -> Path | None:
        best = self.report()["best"]
        if not best:
            return None
        out = self.dir / "best"
        c = self.cands[best]
        (out / "wiki").mkdir(parents=True, exist_ok=True)
        (out / "system.md").write_text(c.text(C.SYSTEM) + "\n", encoding="utf-8")
        for key in c.wiki_keys():
            if c.docs[key]:
                p = out / key
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(c.text(key) + "\n", encoding="utf-8")
        (out / "cid.txt").write_text(best + "\n", encoding="utf-8")
        (self.dir / "result.json").write_text(json.dumps(self.report(), ensure_ascii=False, indent=1), encoding="utf-8")
        return out


# ───────────── 저장소에 적용 ─────────────

def apply_candidate(run: Run, cid: str, prompt_path: str, wiki_root: Path, yes: bool) -> None:
    """후보를 저장소의 system.md와 편집 가능한 위키 문서에 쓴다. 먼저 diff를 보여 주고, --yes일 때만 쓴다(원본은 백업)."""
    if cid not in run.cands:
        raise SystemExit(f"후보 {cid}가 없습니다. 있는 후보: {', '.join(run.state['pool']) or '(없음)'} (풀) / 전체 {len(run.cands)}개")
    c = run.cands[cid]
    plan: list[tuple[Path, str | None, str]] = [(Path(prompt_path), c.text(C.SYSTEM) + "\n", C.SYSTEM)]
    for key in c.wiki_keys():
        plan.append((wiki_root / key[len(C.WIKI):], c.text(key) + "\n" if c.docs[key] else None, key))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    changed = 0
    for path, new, label in plan:
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old == new:
            continue
        changed += 1
        diff = difflib.unified_diff((old or "").splitlines(), (new or "").splitlines(), f"현재 {path}", f"{cid} {path.name}", lineterm="")
        print("\n".join(diff) or f"(삭제) {path}")
        if yes:
            if old is not None:
                bak = run.dir / f"backup_{stamp}" / label  # 폴더 구조를 유지한다(다른 폴더의 같은 이름 문서가 덮어쓰이지 않게)
                bak.parent.mkdir(parents=True, exist_ok=True)
                bak.write_text(old, encoding="utf-8")
            if new is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new, encoding="utf-8")
    print(f"\n{changed}개 파일이 달라집니다." + (f" 적용했고 원본은 {run.dir / f'backup_{stamp}'}에 있습니다." if yes and changed else
                                                 " (미리보기) 적용하려면 --yes를 붙이세요." if changed else ""))


# ───────────── CLI ─────────────

def resolve_recipients(seeds: list[int], per_seed: int) -> list[dict]:
    """main.py --seed S --n N이 뽑는 수신자와 같다(시드마다 random.Random(S)에서 차례로 뽑는다)."""
    import personas
    out, seen = [], set()
    for seed in seeds:
        rng = random.Random(seed)
        k = 0
        while k < per_seed:
            p = personas.sample_persona(rng)
            if p["uuid"] in seen:
                continue
            seen.add(p["uuid"])
            out.append({"uuid": p["uuid"], "name": personas.persona_name(p), "seed": seed})
            k += 1
    return out


def add_llm_args(p: argparse.ArgumentParser) -> None:
    """통화·채점·반성 서버 설정 인자(runner와 scripts/gepa_compare.py가 같이 쓴다)."""
    from agent import DEFAULT_API_BASE, DEFAULT_MODEL, LLMConfig
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--temperature", type=float, default=LLMConfig.temperature)
    p.add_argument("--max-tokens", type=int, default=LLMConfig.max_tokens)
    p.add_argument("--plan-max-tokens", type=int, default=LLMConfig.plan_max_tokens)
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_LLM_MODEL"))
    p.add_argument("--judge-api-base", default=os.environ.get("JUDGE_LLM_API_BASE"))
    p.add_argument("--reflect-model", help="반성(학습자) 모델(기본: --model)")
    p.add_argument("--reflect-api-base", help="반성 서버(기본: --api-base). 평가 서버에 두면 통화 서버가 통화만 한다")
    p.add_argument("--call-slots", type=int, default=2, help="통화 서버 동시 요청 한도")
    p.add_argument("--judge-slots", type=int, default=2, help="채점 서버 동시 요청 한도")
    p.add_argument("--reflect-slots", type=int, default=2, help="반성 서버 동시 요청 한도(반성은 2슬롯을 쓴다)")


def make_llms(args: argparse.Namespace):
    """(통화 설정, 채점 설정, 반성 설정, 서버별 슬롯)."""
    from agent import LLMConfig
    from prompt_opt.evaluate import SlotPool
    from rl import judge_config

    call_llm = LLMConfig(model=args.model, api_base=args.api_base or None, temperature=args.temperature,
                         max_tokens=args.max_tokens, plan_max_tokens=args.plan_max_tokens)
    jk = {}
    if args.judge_model:
        jk["model"] = args.judge_model
    if args.judge_api_base is not None:
        jk["api_base"] = args.judge_api_base or None
    judge_llm = judge_config(**jk)
    reflect_llm = LLMConfig(model=getattr(args, "reflect_model", None) or args.model,
                            api_base=(args.api_base if getattr(args, "reflect_api_base", None) is None else args.reflect_api_base) or None,
                            temperature=args.temperature, max_tokens=args.max_tokens, plan_max_tokens=args.plan_max_tokens)
    cap: dict = {}
    for llm, n in ((call_llm, args.call_slots), (judge_llm, args.judge_slots), (reflect_llm, getattr(args, "reflect_slots", 2))):
        cap[llm.api_base] = max(cap.get(llm.api_base, 0), n)  # 같은 서버를 나눠 쓰면 큰 한도를 공유한다
    return call_llm, judge_llm, reflect_llm, SlotPool(cap)


def parse_args(argv=None) -> argparse.Namespace:
    d = Config("hillclimb", "")
    p = argparse.ArgumentParser(description="GEPA 계열 프롬프트 최적화(system.md + 위키). 세 전략을 같은 조건으로 비교한다")
    p.add_argument("--run", required=True, help="실행 이름(agent_workspace/gepa/<run>/). 같은 이름이면 이어서 돈다")
    p.add_argument("--strategy", choices=STRATEGIES, help="새 실행에서 필수")
    p.add_argument("--purpose", help="통화 목적(새 실행에서 필수)")
    p.add_argument("--metric", choices=("natural", "task"), default=d.metric, help="natural: 자연스러움 → 과제 / task: 과제 → 자연스러움")
    p.add_argument("--seeds", default="41,42", help="평가 수신자 집합 D의 시드(쉼표)")
    p.add_argument("--per-seed", type=int, default=d.per_seed)
    p.add_argument("--budget", type=int, default=d.budget, help="통화 수 예산(서버 오류 통화는 세지 않는다)")
    p.add_argument("--minibatch", type=int, default=d.minibatch)
    p.add_argument("--max-turns", type=int, default=d.max_turns)
    p.add_argument("--rng-seed", type=int, default=d.rng_seed)
    p.add_argument("--system-max", type=int, default=d.system_max)
    p.add_argument("--wiki-max", type=int, default=d.wiki_max, help="편집 가능한 위키 문서 전체 글자 수 상한")
    p.add_argument("--workers", type=int, default=d.workers, help="동시에 걸 통화 수(서버 슬롯 한도 안에서)")
    p.add_argument("--max-merges", type=int, default=d.max_merges)
    p.add_argument("--hill-tol", type=float, default=d.hill_tol)
    p.add_argument("--no-new-docs", action="store_true", help="반성이 새 위키 문서를 만들지 못하게 한다")
    p.add_argument("--no-prefetch", action="store_true", help="ours에서 선행 반성(다음 반성을 시험과 겹쳐 돌림)을 끈다")
    p.add_argument("--caller-prompt", default=learner.SYSTEM_PROMPT_PATH, help="첫 후보로 쓸 system.md")
    add_llm_args(p)
    p.add_argument("--max-iters", type=int, help="이번 실행에서 처리할 반복 수 상한(스모크용)")
    p.add_argument("--report", action="store_true", help="실행하지 않고 결과표만 출력")
    p.add_argument("--apply", metavar="CID", help="후보를 저장소 system.md·위키에 적용(diff 미리보기, --yes로 적용)")
    p.add_argument("--yes", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import functools

    import wiki_store
    from prompt_opt import reflect as R
    from prompt_opt.evaluate import Evaluator
    from prompts import GEPA_OBJECTIVE_NATURAL, GEPA_OBJECTIVE_TASK

    args = parse_args(argv)
    run_dir = RUN_ROOT / args.run
    fresh = not (run_dir / "state.json").exists()
    if fresh and not (args.strategy and args.purpose) and not (args.report or args.apply):
        raise SystemExit("새 실행에는 --strategy와 --purpose가 필요합니다")
    if not fresh:
        state = Run.load_state(run_dir)
        saved = Config.from_dict(state["config"])
        want = Config(args.strategy or saved.strategy, args.purpose or saved.purpose, args.metric, [int(s) for s in args.seeds.split(",")],
                      args.per_seed, args.budget, args.minibatch, args.max_turns, args.rng_seed, args.system_max, args.wiki_max,
                      args.workers, args.max_merges, args.hill_tol, saved.seq_start, saved.seq_step, saved.seq_max, not args.no_new_docs,
                      (args.strategy or saved.strategy) == "ours" and not args.no_prefetch)
        # 이어 돌릴 때는 저장된 설정이 기준이다. 명령줄에서 명시적으로 다르게 준 고정 값이 있으면 막는다.
        diff = [k for k in Config.FIXED if getattr(want, k) != getattr(saved, k)
                and not (k in ("strategy", "purpose") and not getattr(args, k))]
        if diff and not (args.report or args.apply):
            raise SystemExit("이어 돌리려면 다음 설정이 처음과 같아야 합니다: " + ", ".join(
                f"{k}(처음 {getattr(saved, k)!r}, 지금 {getattr(want, k)!r})" for k in diff))
        cfg = Config.from_dict({**asdict(saved), "budget": args.budget, "workers": args.workers, "max_merges": args.max_merges})
    else:
        cfg = Config(args.strategy or "hillclimb", args.purpose or "", args.metric, [int(s) for s in args.seeds.split(",")],
                     args.per_seed, args.budget, args.minibatch, args.max_turns, args.rng_seed, args.system_max, args.wiki_max,
                     args.workers, args.max_merges, args.hill_tol, allow_new_docs=not args.no_new_docs,
                     prefetch=(args.strategy == "ours" and not args.no_prefetch))

    call_llm, judge_llm, reflect_llm, slots = make_llms(args)

    if fresh:
        if args.report or args.apply:
            raise SystemExit(f"{run_dir}에 실행이 없습니다")
        pages = {rel: wiki_store.read(rel) for rel in wiki_store.list_pages()}
        readonly = {r: t for r, t in pages.items() if wiki_store.is_readonly(r)}
        editable = {r: t for r, t in pages.items() if not wiki_store.is_readonly(r)}
        recipients = resolve_recipients(cfg.seeds, cfg.per_seed)
        patterns, system_text = wiki_store.readonly_patterns(), learner.load_caller_profile(args.caller_prompt)
        state = None
    else:
        readonly, patterns = state["readonly_docs"], state["patterns"]

    evaluator = Evaluator(run_dir, cfg.purpose, call_llm, judge_llm, slots, readonly, patterns, cfg.max_turns)
    objective = GEPA_OBJECTIVE_NATURAL if cfg.metric == "natural" else GEPA_OBJECTIVE_TASK
    reflect_fn = functools.partial(R.reflect, llm=reflect_llm, slots=slots, objective=objective,
                                   system_max=cfg.system_max, wiki_max=cfg.wiki_max)
    if fresh:
        run = Run.create(run_dir, cfg, system_text, editable, readonly, patterns, recipients, evaluator, reflect_fn)
        print(f"새 실행 {run_dir}: 수신자 {len(recipients)}명, 편집 가능한 문서 {1 + len(editable)}개", flush=True)
    else:
        run = Run(run_dir, cfg, state, evaluator, reflect_fn)

    if args.apply:
        apply_candidate(run, args.apply, args.caller_prompt, wiki_store.WIKI_ROOT, args.yes)
        return
    if not args.report:
        import personas
        personas.open_dataset()  # 스레드를 만들기 전에 데이터셋을 연다
        why = run.run_loop(args.max_iters)
        run.state["slot_utilization"] = slots.utilization()  # 이번 실행 동안 서버별 슬롯 사용률(0~1)
        run.save()
        print(f"\n멈춤: {why}", flush=True)
    run.print_report()
    if out := run.write_best():
        print(f"best 후보 저장: {out}")


if __name__ == "__main__":
    main()
