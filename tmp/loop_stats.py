"""최근 N개 통화 기록의 루프 지표를 출력한다. 사용: uv run tmp/loop_stats.py --last 10"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

LOOPY = {"repetition_loop", "max_turns", "empty_response"}
# 서버·연결 오류는 프롬프트·코드 품질 문제가 아니므로 루프성 종료와 따로 센다.


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--last", type=int, default=10)
    p.add_argument("--dir", default="agent_workspace/conversation_history")
    args = p.parse_args()

    files = sorted(Path(args.dir).glob("*.json"))[-args.last:]
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if not records:
        raise SystemExit("기록 없음")
    reviewed = [r for r in records if r.get("review")]
    caller = [t["text"] for r in records for t in r["turns"] if t["speaker"] == "caller" and t["text"]]
    sentences = [len([s for s in re.split(r"[.!?]+\s*", t) if s.strip()]) for t in caller]

    def rate(key: str) -> str:
        return f"{sum(bool(r['review'][key]) for r in reviewed)}/{len(reviewed)}" if reviewed else "리뷰 없음"

    ends = Counter(r["end_reason"] for r in records)
    print(f"통화 {len(records)}건 ({files[0].name} ~ {files[-1].name}), 리뷰 있음 {len(reviewed)}건")
    print(f"종료 사유: {dict(ends)}")
    print(f"루프성 종료: {sum(ends[k] for k in LOOPY)}건")
    if ends["error"]:
        print(f"서버 오류 종료: {ends['error']}건 (진단에서 제외하고 서버를 점검한 뒤 해당 통화를 다시 돌린다)")
    print(f"스팸 신고: {rate('spam')}")
    print(f"목적 달성: {rate('achieved')}")
    print(f"평균 턴: {sum(r['num_turns'] for r in records) / len(records):.1f}")
    if caller:
        print(f"발신자 발화: 평균 {sum(map(len, caller)) / len(caller):.0f}자, "
              f"{sum(sentences) / len(sentences):.1f}문장, 3문장 이상 {sum(s >= 3 for s in sentences)}/{len(caller)}")


if __name__ == "__main__":
    main()
