"""자연스러움 심판(rl/reward.py의 judge_call)을 정답이 붙은 통화(scripts/natural_bench_cases.json)로 채점한다.
강화학습을 시작하기 전의 관문: 심판의 0점 재현율이 GATE_ZERO_RECALL 이상이어야 한다(부자연스러운 발화를 놓치면 그 말투가 보상을 받는다).
사례: 실제 통화 10건(종료 사유가 섞이게 뽑음)과 시나리오가 다른 합성 통화 6건. 정답은 사람이 아니라 Claude가 심판 출력을 보지 않고
매겼고, 사용자가 일부를 검수한다(검수 결과 어긋난 발화는 사례의 "reviewed"에 표시하고 기준 문구를 고친다).
심판 문구(prompts.py의 NATURAL_JUDGE_*, SUBMIT_NATURALNESS_*)나 모델을 바꿀 때마다 돌려, 지표가 같거나 올라야 채택한다.
모델 후보 비교: --model openai/... --api-base http://.../v1 (학습 심판은 기본 14B, 감사 심판은 다른 모델).
사용: uv run scripts/natural_bench.py [--out 결과.json] [--repeat 1] [--workers 1] [--model M --api-base URL]"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl import judge_config, reward  # noqa: E402

CASES = Path(__file__).with_name("natural_bench_cases.json")
GATE_ZERO_RECALL = 0.8
LOOPY = reward.LOOPY_ENDS  # 이 종료 사유는 코드가 끝맺음 부자연으로 정하므로 심판의 ending_ok는 채점하지 않는다


def build(case: dict) -> dict:
    turns = []
    for i, (speaker, text) in enumerate(case["turns"]):
        t = {"turn": i, "speaker": speaker, "text": text}
        if i > 0 and not text:
            t["hangup"] = True
        turns.append(t)
    return {"end_reason": case["end_reason"], "turns": turns}


def run_case(case: dict, llm, repeat: int) -> dict:
    rec = build(case)
    runs, t0, err = [], time.time(), None
    for _ in range(repeat):
        try:
            jr = reward.judge_call(rec, llm)
        except Exception as e:  # 서버 오류는 실패로 세되 원인을 남긴다(심판이 형식을 못 맞춘 실패와 구분한다)
            jr, err = None, f"{type(e).__name__}: {str(e)[:200]}"
        runs.append(None if jr is None else {"scores": [jr["scores"][n] for n in sorted(jr["scores"])],
                                             "ending_ok": jr["ending_ok"], "notes": jr["notes"]})
    return {"id": case["id"], "source": case["source"], "end_reason": case["end_reason"],
            "gold": case["scores"], "gold_ending": case["ending_ok"], "runs": runs, "error": err, "sec": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1, help="같은 통화를 심판에게 여러 번 물어 변동을 본다")
    p.add_argument("--model")
    p.add_argument("--api-base")
    p.add_argument("--source", choices=["real", "synthetic"])
    args = p.parse_args()

    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8")) if not args.source or c["source"] == args.source]
    over = {}
    if args.model:
        over["model"] = args.model
    if args.api_base is not None:
        over["api_base"] = args.api_base or None
    llm = judge_config(**over)
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda c: run_case(c, llm, args.repeat), cases))

    pairs, fails, end_pairs = [], 0, []  # (정답, 심판)
    for r in results:
        for run in r["runs"]:
            if run is None:
                fails += 1
                continue
            pairs += list(zip(r["gold"], run["scores"]))
            if r["end_reason"] not in LOOPY:
                end_pairs.append((r["gold_ending"], run["ending_ok"]))
        shown = " ".join("".join(str(x) for x in run["scores"]) if run else "실패" for run in r["runs"])
        gold = "".join(str(x) for x in r["gold"])
        print(f"{r['id']:12} {r['end_reason']:17} 정답 {gold:14} 심판 {shown}  ({r['sec']}s)" + (f"\n    서버 오류: {r['error']}" if r["error"] else ""))

    errs = sum(bool(r["error"]) for r in results)
    print(f"\n통화 {len(results)}건 × {args.repeat}회, 심판 실패 {fails}회" + (f"(서버 오류가 난 통화 {errs}건: 점수가 아니라 연결·인증부터 확인)" if errs else "")
          + f", 발화 판정 {len(pairs)}개")
    if not pairs:
        return
    exact = sum(g == j for g, j in pairs)
    near = sum(abs(g - j) <= 1 for g, j in pairs)
    zeros = [(g, j) for g, j in pairs if g == 0]
    flagged = [(g, j) for g, j in pairs if j == 0]
    recall = sum(j == 0 for _, j in zeros) / len(zeros) if zeros else 0.0
    prec = sum(g == 0 for g, _ in flagged) / len(flagged) if flagged else 0.0
    twos = [(g, j) for g, j in pairs if g == 2]
    print(f"정확 일치 {exact}/{len(pairs)} ({exact / len(pairs):.0%}), ±1 이내 {near}/{len(pairs)} ({near / len(pairs):.0%}), "
          f"평균 절대오차 {statistics.fmean(abs(g - j) for g, j in pairs):.2f}")
    print(f"0점 재현율 {sum(j == 0 for _, j in zeros)}/{len(zeros)} ({recall:.0%}), 0점 정밀도 {sum(g == 0 for g, _ in flagged)}/{len(flagged)} ({prec:.0%}), "
          f"자연스러운 발화(정답 2점)를 0점으로 오판 {sum(j == 0 for _, j in twos)}/{len(twos)}")
    print("혼동행렬(행=정답, 열=심판 0/1/2): " + " | ".join(
        f"{g}: " + "/".join(str(sum(1 for gg, jj in pairs if gg == g and jj == j)) for j in (0, 1, 2)) for g in (0, 1, 2)))
    if end_pairs:
        ok = sum(g == j for g, j in end_pairs)
        bad = [(g, j) for g, j in end_pairs if not g]
        print(f"끝맺음(종료 사유가 루프·max_turns가 아닌 통화) 일치 {ok}/{len(end_pairs)}, 부자연스러운 끝맺음 재현율 "
              f"{sum(not j for _, j in bad)}/{len(bad)}")
    print(f"관문(0점 재현율 ≥ {GATE_ZERO_RECALL}): {'통과' if recall >= GATE_ZERO_RECALL else '미달'} (재현율 {recall:.2f}, 정밀도 {prec:.2f})")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
