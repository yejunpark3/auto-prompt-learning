"""기록 몇 건을 리뷰어로 2번씩 다시 돌려 판정 일관성을 본다. 사용: uv run tmp/review_check.py [기록.json ...]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reviewer  # noqa: E402
from agent import LLMConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", type=Path)
    p.add_argument("--last", type=int, default=5, help="files가 없을 때 볼 최근 기록 수")
    p.add_argument("--dir", default="agent_workspace/conversation_history")
    args = p.parse_args()

    files = args.files or sorted(Path(args.dir).glob("*.json"))[-args.last:]
    llm = LLMConfig()
    for f in files:
        record = json.loads(f.read_text(encoding="utf-8"))
        runs = [reviewer.review_call(record, llm) for _ in range(2)]
        achieved = [r and r["achieved"] for r in runs]
        flag = "" if achieved[0] == achieved[1] else "  <-- 불일치"
        print(f"{f.name} [{record['end_reason']}] 달성: {achieved}{flag}")
        for r in runs:
            print("   ", (r["notes"].split("\n", 1)[0] if r else "(리뷰 실패: submit_review 호출 없음/형식 오류/잘림)"))


if __name__ == "__main__":
    main()
