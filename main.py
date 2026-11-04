import argparse
import random

import learner
import reviewer
import personas
from agent import DEFAULT_API_BASE, DEFAULT_MODEL, LLMConfig
from prompts import SYSTEM_PROMPT_PATH, load_caller_profile
from phone_call import DEFAULT_OUT_DIR, build_agents, save_record, simulate_call


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nemotron-Personas-Korea 페르소나 두 명의 전화통화 시뮬레이션"
    )
    p.add_argument("--purpose", required=True, help="발신자의 통화 목적")
    p.add_argument("--caller-prompt", default=SYSTEM_PROMPT_PATH, help="발신자 시스템 프롬프트 파일")
    p.add_argument("--callee-uuid", help="수신자 페르소나 uuid (없으면 무작위)")
    p.add_argument("--seed", type=int, help="페르소나 샘플링 시드")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--n", type=int, default=1, help="페르소나를 새로 뽑아 반복할 통화 수")
    p.add_argument("--model", default=DEFAULT_MODEL, help="litellm 모델 문자열")
    p.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help='빈 문자열이면 api_base와 MLX 전용 옵션을 끈다 (예: --model gemini/gemini-2.5-flash --api-base "")',
    )
    p.add_argument("--temperature", type=float, default=LLMConfig.temperature)
    p.add_argument("--max-tokens", type=int, default=LLMConfig.max_tokens)
    p.add_argument(
        "--plan-max-tokens",
        type=int,
        default=LLMConfig.plan_max_tokens,
        help="통화 전 계획 생성의 max_tokens (0이면 계획 생략)",
    )
    p.add_argument("--no-wiki", action="store_true", help="발신자에게 위키 도구(읽기·검색)를 주지 않는다")
    p.add_argument(
        "--learn-every", type=int, default=20,
        help="통화 N건마다 리뷰를 모아 system.md와 위키를 고친다 (0이면 끔)",
    )
    p.add_argument("--out", default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        caller_profile = load_caller_profile(args.caller_prompt)
    except OSError as e:
        raise SystemExit(f"발신자 프롬프트를 읽을 수 없습니다: {e}")
    rng = random.Random(args.seed)
    llm = LLMConfig(
        model=args.model,
        api_base=args.api_base or None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        plan_max_tokens=args.plan_max_tokens,
    )

    errors_in_row = 0
    for i in range(args.n):
        # uuid를 지정한 경우 첫 통화에만 쓰고, 이후 통화는 새로 뽑는다.
        callee_p = (
            personas.get_persona(args.callee_uuid)
            if i == 0 and args.callee_uuid
            else personas.sample_persona(rng)
        )

        print(f"\n=== 통화 {i + 1}/{args.n} ===")
        print(f"발신자: ({args.caller_prompt})")
        print(f"수신자: {personas.summarize(callee_p)}")
        print(f"목적: {args.purpose}\n")

        caller, callee = build_agents(
            caller_profile, callee_p, args.purpose, llm, wiki=not args.no_wiki
        )
        record = simulate_call(caller, callee, args.purpose, max_turns=args.max_turns)
        # 서버·연결 오류로 끊긴 통화는 학습에 넣지 않고 같은 수신자로 다시 돌린다.
        if args.learn_every and record["end_reason"] == "error":
            print(f"[경고] 서버 오류로 끊긴 통화라 리뷰하지 않습니다 (다시 돌리기: --callee-uuid {callee_p['uuid']} --n 1)", flush=True)
        elif args.learn_every:
            try:
                record["review"] = reviewer.review_call(record, llm)
                if record["review"] is None:
                    print("[경고] 리뷰가 비었거나 잘려 학습 대기열에 넣지 않습니다", flush=True)
            except Exception as e:
                print(f"[경고] 통화 리뷰 실패: {type(e).__name__}: {e}", flush=True)
        path = save_record(record, args.out)
        print(f"\n종료: {record['end_reason']} ({record['num_turns']}턴) → {path}")

        # 서버가 죽었는데 남은 통화를 계속 돌리면 오류 기록만 쌓인다. 연속 3통화가 오류면 멈춘다.
        errors_in_row = errors_in_row + 1 if record["end_reason"] == "error" else 0
        if errors_in_row >= 3:
            print(f"[중단] 서버 오류로 끊긴 통화가 {errors_in_row}통화 연속입니다. 남은 통화를 건너뜁니다.", flush=True)
            break

        if args.learn_every and (record.get("review") or {}).get("notes"):
            try:
                learner.enqueue(record, path)
                if round_dir := learner.maybe_learn(llm, args.caller_prompt, args.learn_every):
                    print(f"[학습] {round_dir}", flush=True)
                    caller_profile = load_caller_profile(args.caller_prompt)
            except Exception as e:
                print(f"[경고] 학습 실패: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
