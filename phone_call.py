import json
import re
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import personas
import wiki_store
from agent import LLMConfig, PhoneAgent
from prompts import (
    CALLEE_GREETING,
    END_CALL_TOOL,
    PLAN_REQUEST,
    PLAN_WIKI_HINT,
    REPORT_SPAM_NAME,
    REPORT_SPAM_TOOL,
    RING_CUE,
    WRAP_UP_HINT,
    callee_prompt,
    caller_prompt,
)

DEFAULT_OUT_DIR = "agent_workspace/conversation_history"
WRAP_UP_REMAINING = 4

REPEAT_RATIO = 0.9
# "네.", "알겠습니다." 같은 짧은 확인 응답은 다른 질문에 같은 말로 답하는 정상 대화일 수
# 있으므로 반복으로 보지 않는다. 짧은 말의 반복은 max_turns가 막는다.
REPEAT_MIN_CHARS = 10

_LABELS = {"caller": "발신", "callee": "수신"}


def _is_repeat(prev: str, text: str) -> bool:
    norm = lambda s: re.sub(r"[\s.,!?~]+", "", s)
    prev, text = norm(prev), norm(text)
    if min(len(prev), len(text)) < REPEAT_MIN_CHARS:
        return False
    return SequenceMatcher(None, prev, text).ratio() >= REPEAT_RATIO


def _wiki_ready() -> bool:
    """위키에 문서가 있고 인덱스·모델을 미리 올릴 수 있을 때만 발신자에게 위키 도구를 준다."""
    if not wiki_store.list_pages():
        return False
    try:
        wiki_store.warm()
    except Exception as e:
        print(f"[경고] 위키를 쓸 수 없어 도구 없이 진행합니다: {type(e).__name__}: {e}", flush=True)
        return False
    return True


def build_agents(
    caller_profile: str, callee_persona: dict, purpose: str, llm: LLMConfig,
    wiki: bool = True,
) -> tuple[PhoneAgent, PhoneAgent]:
    callee_name = personas.persona_name(callee_persona)
    use_wiki = wiki and _wiki_ready()
    caller = PhoneAgent(
        role="caller",
        persona=None,
        name="발신자",
        system_prompt=caller_prompt(
            caller_profile, purpose, personas.render_known(callee_persona)
        ),
        llm=llm,
        tools=[END_CALL_TOOL, *(wiki_store.WIKI_READ_TOOLS if use_wiki else [])],
        tool_runner=wiki_store.run_read_tool if use_wiki else None,
    )
    callee = PhoneAgent(
        role="callee",
        persona=callee_persona,
        name=callee_name,
        system_prompt=callee_prompt(personas.render_profile(callee_persona)),
        llm=llm,
        tools=[END_CALL_TOOL, REPORT_SPAM_TOOL],
    )
    return caller, callee


def print_turn(turn: dict) -> None:
    label = _LABELS[turn["speaker"]]
    print(f"[{label}] {turn['name']}: {turn['text']}", flush=True)


def print_plan(plan: dict) -> None:
    print(f"[계획]\n{plan['text']}\n", flush=True)


def simulate_call(
    caller: PhoneAgent,
    callee: PhoneAgent,
    purpose: str,
    max_turns: int = 30,
    on_turn=print_turn,
    on_plan=print_plan,
) -> dict:
    started_at = datetime.now()
    turns: list[dict] = []
    end_reason = "max_turns"

    # 통화 전에 발신자가 계획을 세운다. 실패해도 계획 없이 통화는 진행한다.
    plan = None
    if caller.llm.plan_max_tokens > 0:
        try:
            plan = caller.plan(
                PLAN_REQUEST.format(wiki=PLAN_WIKI_HINT if caller.tool_runner else "")
            )
            if plan:
                on_plan(plan)
            else:
                print(
                    "[경고] 계획이 비었거나 plan_max_tokens에서 잘려 계획 없이 진행합니다",
                    flush=True,
                )
        except Exception as e:
            print(f"[경고] 계획 생성 실패: {type(e).__name__}: {e}", flush=True)

    # 수신자가 먼저 전화를 받고, 이후 발신자와 수신자가 번갈아 말한다.
    # 첫 인사는 모델에 맡기면 상황을 해설하듯 말하므로("전화가 왔나 봐요") 고정 대사로 넣는다.
    callee.messages += [
        {"role": "user", "content": RING_CUE},
        {"role": "assistant", "content": CALLEE_GREETING},
    ]
    greeting = {"turn": 0, "speaker": callee.role, "name": callee.name,
                "text": CALLEE_GREETING}
    turns.append(greeting)
    on_turn(greeting)
    speaker, incoming = caller, CALLEE_GREETING
    while len(turns) < max_turns:
        if max_turns - len(turns) <= WRAP_UP_REMAINING and len(turns) > 0:
            incoming = f"{incoming} {WRAP_UP_HINT}"
        try:
            text, hung_up, meta = speaker.respond(incoming)
        except Exception as e:  # litellm 연결/응답 오류
            end_reason = "error"
            turns.append(
                {"turn": len(turns), "speaker": speaker.role, "name": speaker.name,
                 "text": "", "error": f"{type(e).__name__}: {e}"}
            )
            print(f"[오류] {speaker.name}: {type(e).__name__}: {e}", flush=True)
            break

        turn = {"turn": len(turns), "speaker": speaker.role, "name": speaker.name,
                "text": text, **meta}
        if hung_up:
            turn["hangup"] = True
        if hung_up == REPORT_SPAM_NAME:
            turn["spam_report"] = True
        turns.append(turn)
        on_turn(turn)

        if hung_up:
            kind = "spam_reported" if hung_up == REPORT_SPAM_NAME else "hangup"
            end_reason = f"{kind}_by_{speaker.role}"
            break
        if not text:
            end_reason = "empty_response"
            break
        # 같은 화자가 직전 발화를 되풀이하면 루프로 보고 끊는다 (기록은 end_reason으로 구분).
        prev = next(
            (t["text"] for t in reversed(turns[:-1]) if t["speaker"] == speaker.role),
            None,
        )
        if prev and _is_repeat(prev, text):
            end_reason = "repetition_loop"
            break
        speaker = caller if speaker is callee else callee
        incoming = text

    return {
        "call_id": uuid.uuid4().hex[:8],
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": purpose,
        "end_reason": end_reason,
        "num_turns": len(turns),
        "llm": caller.llm.public(),
        "caller": {**_agent_record(caller), "plan": plan},
        "callee": _agent_record(callee),
        "turns": turns,
    }


def _agent_record(agent: PhoneAgent) -> dict:
    return {
        "uuid": agent.persona["uuid"] if agent.persona else None,
        "name": agent.name,
        "system_prompt": agent.system_prompt,
        "persona": agent.persona,
    }


def save_record(record: dict, out_dir: str = DEFAULT_OUT_DIR) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(record["started_at"]).strftime("%Y%m%d-%H%M%S")
    name = (
        f"{stamp}_{(record['caller']['uuid'] or 'agent')[:6]}-{record['callee']['uuid'][:6]}"
        f"_{record['call_id']}.json"
    )
    path = out / name
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
