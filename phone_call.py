import json
import re
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import personas
from agent import LLMConfig, PhoneAgent
from prompts import (
    HANGUP_TOKEN,
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


def build_agents(
    caller_persona: dict, callee_persona: dict, purpose: str, llm: LLMConfig
) -> tuple[PhoneAgent, PhoneAgent]:
    caller_name = personas.persona_name(caller_persona)
    callee_name = personas.persona_name(callee_persona)
    caller = PhoneAgent(
        role="caller",
        persona=caller_persona,
        name=caller_name,
        system_prompt=caller_prompt(
            personas.render_profile(caller_persona), caller_name, callee_name, purpose
        ),
        llm=llm,
        hangup_token=HANGUP_TOKEN,
    )
    callee = PhoneAgent(
        role="callee",
        persona=callee_persona,
        name=callee_name,
        system_prompt=callee_prompt(
            personas.render_profile(callee_persona), callee_name
        ),
        llm=llm,
        hangup_token=HANGUP_TOKEN,
    )
    return caller, callee


def print_turn(turn: dict) -> None:
    label = _LABELS[turn["speaker"]]
    print(f"[{label}] {turn['name']}: {turn['text']}", flush=True)


def simulate_call(
    caller: PhoneAgent,
    callee: PhoneAgent,
    purpose: str,
    max_turns: int = 30,
    on_turn=print_turn,
) -> dict:
    started_at = datetime.now()
    turns: list[dict] = []
    end_reason = "max_turns"

    # 수신자가 먼저 전화를 받고, 이후 발신자와 수신자가 번갈아 말한다.
    speaker, incoming = callee, RING_CUE
    while len(turns) < max_turns:
        if max_turns - len(turns) <= WRAP_UP_REMAINING and len(turns) > 0:
            incoming = f"{incoming} {WRAP_UP_HINT}"
        try:
            text, meta = speaker.respond(incoming)
        except Exception as e:  # litellm 연결/응답 오류
            end_reason = "error"
            turns.append(
                {"turn": len(turns), "speaker": speaker.role, "name": speaker.name,
                 "text": "", "error": f"{type(e).__name__}: {e}"}
            )
            print(f"[오류] {speaker.name}: {type(e).__name__}: {e}", flush=True)
            break

        hung_up = HANGUP_TOKEN in text
        text = text.replace(HANGUP_TOKEN, "").strip()
        turn = {"turn": len(turns), "speaker": speaker.role, "name": speaker.name,
                "text": text, **meta}
        if hung_up:
            turn["hangup"] = True
        turns.append(turn)
        on_turn(turn)

        if hung_up:
            end_reason = f"hangup_by_{speaker.role}"
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
        "caller": _agent_record(caller),
        "callee": _agent_record(callee),
        "turns": turns,
    }


def _agent_record(agent: PhoneAgent) -> dict:
    return {
        "uuid": agent.persona["uuid"],
        "name": agent.name,
        "system_prompt": agent.system_prompt,
        "persona": agent.persona,
    }


def save_record(record: dict, out_dir: str = DEFAULT_OUT_DIR) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(record["started_at"]).strftime("%Y%m%d-%H%M%S")
    name = (
        f"{stamp}_{record['caller']['uuid'][:6]}-{record['callee']['uuid'][:6]}"
        f"_{record['call_id']}.json"
    )
    path = out / name
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
