"""세 전략(hillclimb | gepa | ours)의 실행 결과를 보류 집합 H에서 짝비교한다. 시나리오를 모른다.

H = 학습·선택에 쓰지 않은 수신자(기본: 시드 201의 10명). 처음 문서(c000)와 각 실행의 best 후보를 같은 수신자에게 걸어 보고(--reps번씩),
수신자별 평균의 짝 차이 Δ ± 표준오차를 낸다(자연스러움·단계 각각). 표준오차보다 작은 차이는 잡음 범위로 표시한다:
수신자 10명 규모에서는 전략 사이 차이가 잡음 안에 들 수 있으므로, 점수 말고도 비용 지표를 함께 낸다.
  - 미니배치 오채택률: 시험을 통과해 채택됐지만 D 전체로는 부모보다 못한 반성 후보의 비율(추가 통화 없음)
  - 채택 1건당 통화 수, 경과 시간, 시간당 통화 수, 슬롯 사용률, 선행 반성으로 겹친 시간
  - 프런티어 크기, 병합 시도·채택, 지어냄 비율(C1: 점수에는 쓰지 않고 기록만)
결과는 agent_workspace/gepa/compare_<날짜>.md와 .json에 남고, 통화 결과는 holdout.json에 캐시돼 중단 후 이어 돌 수 있다.

사용: uv run scripts/gepa_compare.py --runs hc,gepa,ours [--holdout-seeds 201 --holdout-per-seed 10 --reps 1 --workers 2]
      (--runs의 실행은 같은 --purpose·--metric으로 돌린 것이어야 한다. 실행 이름과 라벨은 --labels로 바꾼다)
"""
import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_opt import candidates as C  # noqa: E402
from prompt_opt import choose as S  # noqa: E402
from prompt_opt import runner as RN  # noqa: E402

OBJ_LABEL = {"natural": "자연스러움", "task": "단계"}
BASE = "base"


# ───────────── 불러오기 ─────────────

def load_runs(names: list[str], root: Path = RN.RUN_ROOT) -> dict[str, "RN.Run"]:
    runs = {}
    for n in names:
        d = root / n
        if not (d / "state.json").exists():
            raise SystemExit(f"{d}에 실행이 없습니다")
        st = RN.Run.load_state(d)
        runs[n] = RN.Run(d, RN.Config.from_dict(st["config"]), st, None, None)
    return runs


def check_comparable(runs: dict[str, "RN.Run"]) -> list[str]:
    """같은 조건에서 돌린 실행끼리만 비교할 수 있다. 어긋나는 점의 목록(비어 있으면 비교 가능)."""
    first = next(iter(runs.values()))
    problems = []
    for name, r in runs.items():
        for k in ("purpose", "metric", "seeds", "per_seed", "max_turns", "system_max", "wiki_max", "minibatch"):
            if getattr(r.cfg, k) != getattr(first.cfg, k):
                problems.append(f"{name}: {k}={getattr(r.cfg, k)!r} (첫 실행은 {getattr(first.cfg, k)!r})")
        if r.cands["c000"].docs != first.cands["c000"].docs and \
                {k: r.cands["c000"].text(k) for k in r.cands["c000"].docs} != {k: first.cands["c000"].text(k) for k in first.cands["c000"].docs}:
            problems.append(f"{name}: 처음 문서(c000)가 다릅니다")
        if r.cfg.budget != first.cfg.budget:
            problems.append(f"{name}: 예산 {r.cfg.budget} (첫 실행은 {first.cfg.budget}) — 같은 예산이어야 공정합니다")
    return problems


def text_hash(c: C.Candidate) -> str:
    return hashlib.sha256(json.dumps({k: c.text(k) for k in sorted(c.docs) if c.text(k)}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def pick_candidates(runs: dict[str, "RN.Run"]) -> dict[str, C.Candidate]:
    """라벨 → 후보. base는 처음 문서, 나머지는 각 실행의 best. 라벨이 cid가 되어 후보별 위키 폴더가 겹치지 않는다."""
    first = next(iter(runs.values()))
    out = {BASE: C.Candidate(BASE, {k: list(v) for k, v in first.cands["c000"].docs.items()})}
    for name, r in runs.items():
        best = r.report()["best"]
        if best is None:
            raise SystemExit(f"{name}: 풀에 후보가 없습니다(c000 평가가 끝나지 않았습니다)")
        out[name] = C.Candidate(name, {k: list(v) for k, v in r.cands[best].docs.items()}, [best], summary=f"{name}의 best = {best}")
    return out


# ───────────── 보류 집합 평가 ─────────────

def holdout_eval(cands: dict[str, C.Candidate], recipients: list[dict], reps: int, evaluator, workers: int,
                 cache_path: Path, say=print) -> dict[str, dict[str, list[dict]]]:
    """{라벨: {수신자 uuid: [칸(reps개)]}}. 글이 같은 후보(예: 두 실행의 best가 같음)는 한 번만 평가한다.
    칸은 cache_path에 쌓여, 같은 인자로 다시 실행하면 이미 한 통화는 건너뛴다. 서버 오류 칸은 저장하지 않는다."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    lock = threading.Lock()
    key = lambda h, u, k: f"{h}|{u}|{k}"
    hashes = {label: text_hash(c) for label, c in cands.items()}
    rep_of = {}                                  # 글 해시 → 그 글을 평가할 때 쓰는 후보(첫 라벨)
    for label, h in hashes.items():
        rep_of.setdefault(h, label)
    todo = [(rep_of[h], r, k) for h in rep_of for r in recipients for k in range(reps) if key(h, r["uuid"], k) not in cache]
    say(f"보류 평가: 서로 다른 후보 {len(rep_of)}개 × 수신자 {len(recipients)}명 × {reps}회 = {len(rep_of) * len(recipients) * reps}통화 중 {len(todo)}통화를 새로 겁니다")
    t0 = time.monotonic()

    def one(job):
        label, r, k = job
        cell = evaluator.run_cell(cands[label], r)
        with lock:
            if cell["end_reason"] not in RN.SERVER_ERRORS:
                cache[key(hashes[label], r["uuid"], k)] = cell
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            say(f"  {label} × {r['name']} #{k}: 자연스러움 {RN._fmt(cell.get('natural'))}, 단계 {cell.get('stage')}, {cell['end_reason']}")
    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(workers) as ex:
            for f in [ex.submit(one, j) for j in todo]:
                f.result()
    else:
        for j in todo:
            one(j)
    if todo:
        say(f"보류 평가 {time.monotonic() - t0:.0f}초")
    return {label: {r["uuid"]: [c for k in range(reps) if (c := cache.get(key(hashes[label], r["uuid"], k)))] for r in recipients}
            for label in cands}


# ───────────── 통계 ─────────────

def per_recipient(cells: dict[str, dict[str, list[dict]]], label: str, obj: str) -> dict[str, float | None]:
    """수신자별 평균(그 수신자의 반복 통화 중 점수가 있는 것들의 평균)."""
    out = {}
    for u, cs in cells[label].items():
        xs = [c[obj] for c in cs if c.get(obj) is not None]
        out[u] = statistics.fmean(xs) if xs else None
    return out


def paired(cells, a: str, b: str, obj: str) -> dict:
    """a − b의 수신자 짝 차이. wins/ties/losses는 수신자 단위."""
    va, vb = per_recipient(cells, a, obj), per_recipient(cells, b, obj)
    d = S.paired_deltas(va, vb, list(va))
    m, se, n = S.mean_se(d)
    return {"mean": round(m, 4) if n else None, "se": round(se, 4) if n >= 2 else None, "n": n,
            "wins": sum(x > 1e-9 for x in d), "ties": sum(abs(x) <= 1e-9 for x in d), "losses": sum(x < -1e-9 for x in d),
            "distinct": bool(n >= 2 and abs(m) > se)}


def c1_rate(cells, label: str) -> float | None:
    xs = [c["c1"] for cs in cells[label].values() for c in cs if c.get("c1") is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(cells, runs: dict[str, "RN.Run"], objectives: list[str]) -> dict:
    labels = [BASE, *runs]
    means = {l: {o: (lambda v: round(statistics.fmean(v), 4) if v else None)([x for x in per_recipient(cells, l, o).values() if x is not None])
                 for o in objectives} for l in labels}
    vs_base = {l: {o: paired(cells, l, BASE, o) for o in objectives} for l in runs}
    arms = list(runs)
    vs_arm = {f"{a} − {b}": {o: paired(cells, a, b, o) for o in objectives} for i, a in enumerate(arms) for b in arms[:i]}
    cost = {}
    for n, r in runs.items():
        rep = r.report()
        cost[n] = {"strategy": r.cfg.strategy, "best": rep["best"], "calls_used": rep["calls_used"], "elapsed_h": round(rep["elapsed_s"] / 3600, 2),
                   "calls_per_hour": rep["calls_per_hour"], "iterations": rep["iterations"], "accepted": rep["accepted"],
                   "accepted_but_worse_on_D": rep["accepted_but_worse_on_D"],
                   "misadopt_rate": round(rep["accepted_but_worse_on_D"] / rep["accepted"], 3) if rep["accepted"] else None,
                   "confirm_failed": rep["confirm_failed"],
                   "calls_per_accept": rep["calls_per_accept"], "front_size": len(rep["front"]) or None, "merges_tried": rep["merges_tried"],
                   "merges_accepted": rep["merges_accepted"], "slot_utilization": rep["slot_utilization"], "prefetch": rep["prefetch"],
                   "c1_rate_D": next((t["c1_rate"] for t in rep["table"] if t["cid"] == rep["best"]), None), "c1_rate_H": c1_rate(cells, n)}
    return {"labels": labels, "objectives": objectives, "means": means, "vs_base": vs_base, "vs_arm": vs_arm, "cost": cost,
            "c1_rate_H": {l: c1_rate(cells, l) for l in labels}}


# ───────────── 보고서 ─────────────

def _f(v, nd=2):
    return "?" if v is None else f"{v:.{nd}f}"


def _delta(d: dict) -> str:
    if d["mean"] is None:
        return "?"
    se = "—" if d["se"] is None else f"{d['se']:.2f}"
    flag = "구분됨" if d["distinct"] else "잡음 범위"
    return f"{d['mean']:+.2f} ± {se} ({d['wins']}승 {d['ties']}무 {d['losses']}패, {flag})"


def render_md(res: dict, meta: dict) -> str:
    objs = res["objectives"]
    L = []
    L.append(f"# GEPA 3팔 비교 ({meta['date']})\n")
    L.append(f"- 통화 목적: {meta['purpose']}\n- 보류 집합 H: 시드 {meta['seeds']} × {meta['per_seed']}명 = {meta['n_recipients']}명, 후보당 {meta['reps']}회씩\n"
             f"- 우선순위: {' → '.join(OBJ_LABEL[o] for o in meta['order'])}\n- 지어내기는 점수에서 깎지 않았다(성공만 본다). C1 비율만 기록했다.\n")
    L.append("## 보류 집합 점수 (수신자별 평균의 평균)\n")
    L.append("| 후보 | " + " | ".join(f"{OBJ_LABEL[o]}" for o in objs) + " | " + " | ".join(f"{OBJ_LABEL[o]} Δ vs base" for o in objs) + " | C1(H) |")
    L.append("|---|" + "---|" * (2 * len(objs) + 1))
    for l in res["labels"]:
        d = [("—" if l == BASE else _delta(res["vs_base"][l][o])) for o in objs]
        L.append(f"| {l} | " + " | ".join(_f(res["means"][l][o]) for o in objs) + " | " + " | ".join(d) + f" | {_f(res['c1_rate_H'][l])} |")
    if res["vs_arm"]:
        L.append("\n## 전략끼리 (짝 차이 a − b)\n")
        L.append("| 비교 | " + " | ".join(OBJ_LABEL[o] for o in objs) + " |")
        L.append("|---|" + "---|" * len(objs))
        for k, v in res["vs_arm"].items():
            L.append(f"| {k} | " + " | ".join(_delta(v[o]) for o in objs) + " |")
    L.append("\n## 비용과 채택의 질 (탐색 과정, 같은 예산)\n")
    L.append("| 실행 | 전략 | best | 통화 | 시간(h) | 통화/h | 채택 | D 전체로 부모보다 못함 | 확인 단계 탈락 | 채택 1건당 통화 | 프런티어 | 병합 시도/채택 | 슬롯 사용률 | 선행 반성 | C1(D) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, c in res["cost"].items():
        pf = c["prefetch"]
        pfs = f"{pf['used']}회, 반성 {pf['reflect_s'] / 60:.1f}분 중 {max(0.0, pf['reflect_s'] - pf['wait_s']) / 60:.1f}분 겹침" if pf["used"] else "—"
        mis = "—" if c["misadopt_rate"] is None else f"{c['accepted_but_worse_on_D']}/{c['accepted']} ({c['misadopt_rate']:.0%})"
        L.append(f"| {n} | {c['strategy']} | {c['best']} | {c['calls_used']} | {_f(c['elapsed_h'])} | {_f(c['calls_per_hour'], 1)} | {c['accepted']} | {mis} | {c['confirm_failed']} | "
                 f"{_f(c['calls_per_accept'], 1)} | {c['front_size'] or '—'} | {c['merges_tried']}/{c['merges_accepted']} | {c['slot_utilization'] or '—'} | {pfs} | {_f(c['c1_rate_D'])} |")
    L.append("\n## 읽는 법\n")
    L.append(f"- Δ가 표준오차(±)보다 크면 '구분됨', 아니면 '잡음 범위'다. 수신자 {meta['n_recipients']}명에서는 잡음 범위가 흔하다. 그래서 점수 차이만으로 결론내지 않고 "
             "'D 전체로 부모보다 못함'(미니배치를 통과했지만 실제로는 나빠진 채택), 채택 1건당 통화, 시간을 함께 본다.\n"
             "- '전략끼리' 표는 같은 보류 집합에서 a가 b보다 높은지를 본다. base 대비 개선이 구분되지 않으면 그 전략은 이번 예산에서 처음 문서보다 낫다고 말할 수 없다.\n"
             "- 자연스러움의 최종 기준은 심판 점수가 아니라 사람이다. best 후보의 표본 통화를 직접 읽고 `--apply`를 정한다.\n")
    return "\n".join(L) + "\n"


# ───────────── CLI ─────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="hillclimb·gepa·ours 실행 결과를 보류 집합에서 짝비교")
    p.add_argument("--runs", required=True, help="비교할 실행 이름들(쉼표). 예: hc,gepa,ours")
    p.add_argument("--labels", help="표에 쓸 이름들(쉼표, --runs와 같은 순서). 기본은 실행 이름")
    p.add_argument("--holdout-seeds", default="201", help="보류 집합 H의 시드(쉼표)")
    p.add_argument("--holdout-per-seed", type=int, default=10)
    p.add_argument("--reps", type=int, default=1, help="후보×수신자당 통화 수(늘리면 통화 잡음이 줄어든다)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--allow-overlap", action="store_true", help="H가 어느 실행의 D와 겹쳐도 진행(권장하지 않음)")
    p.add_argument("--out-dir", default=str(RN.RUN_ROOT))
    RN.add_llm_args(p)
    return p.parse_args(argv)


def main(argv=None, evaluator=None) -> None:
    args = parse_args(argv)
    names = args.runs.split(",")
    runs = load_runs(names, Path(args.out_dir))
    if args.labels:
        labels = args.labels.split(",")
        if len(labels) != len(names):
            raise SystemExit("--labels는 --runs와 같은 개수여야 합니다")
        runs = {l: r for l, r in zip(labels, runs.values())}
    if problems := check_comparable(runs):
        raise SystemExit("비교할 수 없는 실행입니다:\n  " + "\n  ".join(problems))
    first = next(iter(runs.values()))
    seeds = [int(s) for s in args.holdout_seeds.split(",")]
    recipients = RN.resolve_recipients(seeds, args.holdout_per_seed) if evaluator is None else evaluator.recipients(seeds, args.holdout_per_seed)
    overlap = {r["uuid"] for r in recipients} & {u for run in runs.values() for u in run.D}
    if overlap and not args.allow_overlap:
        raise SystemExit(f"보류 집합 H가 학습 수신자 D와 {len(overlap)}명 겹칩니다. --holdout-seeds를 바꾸세요(겹침을 허용하려면 --allow-overlap)")
    cands = pick_candidates(runs)
    out_dir = Path(args.out_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    work = out_dir / f"holdout_{'_'.join(runs)}_{first.cfg.metric}"
    work.mkdir(parents=True, exist_ok=True)
    if evaluator is None:
        import personas
        from prompt_opt.evaluate import Evaluator
        personas.open_dataset()
        call_llm, judge_llm, _, slots = RN.make_llms(args)
        st = first.state
        evaluator = Evaluator(work, first.cfg.purpose, call_llm, judge_llm, slots, st["readonly_docs"], st["patterns"], first.cfg.max_turns)
    cells = holdout_eval(cands, recipients, args.reps, evaluator, args.workers, work / "holdout.json")
    res = summarize(cells, runs, first.objectives)
    meta = {"date": stamp, "purpose": first.cfg.purpose, "seeds": seeds, "per_seed": args.holdout_per_seed, "n_recipients": len(recipients),
            "reps": args.reps, "order": first.objectives}
    md = render_md(res, meta)
    (out_dir / f"compare_{stamp}.md").write_text(md, encoding="utf-8")
    (out_dir / f"compare_{stamp}.json").write_text(json.dumps({"meta": meta, "result": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(md)
    print(f"저장: {out_dir / f'compare_{stamp}.md'}")


if __name__ == "__main__":
    main()
