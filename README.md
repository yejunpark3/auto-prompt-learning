# auto-prompt-learning

[Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)에서 페르소나 두 명을 뽑아 발신자/수신자 에이전트로 만들고, litellm으로 LLM을 호출해 전화통화를 텍스트로 시뮬레이션한다. 통화 기록은 JSON으로 저장된다.

- 발신자는 `--purpose`로 준 통화 목적을 알고 있고, 수신자는 모르는 번호로 전화를 받는다.
- 수신자가 먼저 "여보세요"로 받고, 이후 번갈아 말한다. 한쪽이 `end_call` 도구(tool calling)를 호출하면 통화가 끝난다.

## 준비

```
uv sync
```

데이터셋(`nvidia/Nemotron-Personas-Korea`)은 Hugging Face 캐시에 없으면 처음 실행할 때 내려받는다(약 4GB).

## 로컬 서버

기본 설정은 `optiq serve`로 띄운 로컬 서버(`http://127.0.0.1:8080/v1`)의 `mlx-community/Qwen3.5-9B-OptiQ-4bit:no-think`(thinking 끈 변형)를 쓴다. `optiq serve`는 `sk-optiq-` 접두사의 키를 요구해서 자리표시 키 `sk-optiq-local`을 보낸다.

로컬 서버(`api_base`를 줄 때)에는 요청마다 `enable_thinking: false`, `top_k: 20`, `presence_penalty: 1.5`도 함께 보낸다. `optiq serve`는 이 필드를 받아들이지만 실제로 반영하는지는 확인하지 않았다.

통화 종료에 tool calling을 쓴다. `mlx_lm.server`는 `tools`를 `tool_calls`로 파싱해 준다. 서버가 파싱하지 않고 content에 `<tool_call>` 블록을 남기면 그 블록에서 `end_call`을 찾고 발화에서는 지운다.

`mlx_lm.server`로 더 작은 모델을 돌릴 수도 있다. 이 서버는 모델의 `generation_config.json`을 읽지 않아서 `presence_penalty`를 빼면 같은 말을 되풀이하는 루프에 빠지기 쉽다(4B에서 확인).

```
uv run mlx_lm.server --model mlx-community/Qwen3.5-4B-OptiQ-4bit --port 8081 \
  --chat-template-args '{"enable_thinking": false}'

uv run main.py --purpose "..." --model openai/mlx-community/Qwen3.5-4B-OptiQ-4bit \
  --api-base http://127.0.0.1:8081/v1
```

## 실행

```
uv run main.py --purpose "다음 달 동창회 참석 여부를 물어보려고 전화함" --seed 1
```

| 옵션 | 설명 |
|---|---|
| `--purpose` | (필수) 발신자가 전화를 건 목적 |
| `--caller-uuid`, `--callee-uuid` | 페르소나를 지정. 없으면 무작위 |
| `--seed` | 페르소나 샘플링 시드 |
| `--n` | 페르소나를 새로 뽑아 반복할 통화 수 |
| `--max-turns` | 최대 턴 수 (기본 30) |
| `--model`, `--api-base` | litellm 모델 문자열과 엔드포인트 |
| `--temperature`, `--max-tokens` | 샘플링 설정 |
| `--out` | 기록 저장 폴더 (기본 `agent_workspace/conversation_history`) |

호스팅 모델을 쓰려면 `--api-base ""`로 `api_base`와 MLX 전용 옵션을 끈다.

```
uv run main.py --purpose "..." --model gemini/gemini-2.5-flash --api-base ""
```

## 출력

통화마다 `{시작시각}_{발신자uuid6}-{수신자uuid6}_{call_id}.json`이 저장된다.

```
call_id, started_at, ended_at, purpose, end_reason, num_turns,
llm: {model, api_base, temperature, top_p, max_tokens, extra_body},
caller / callee: {uuid, name, system_prompt, persona(원본 26개 필드)},
turns: [{turn, speaker, name, text, latency_s, usage}]
```

`end_reason`: `hangup_by_caller`, `hangup_by_callee`, `max_turns`, `repetition_loop`(같은 화자가 직전 발화를 되풀이), `empty_response`, `error`.

## 구조

| 파일 | 역할 |
|---|---|
| `personas.py` | 데이터셋 로드, 샘플링, 프로필 렌더링 |
| `prompts.py` | 시스템 프롬프트와 전화 규칙 (프롬프트를 고칠 때는 여기만) |
| `agent.py` | `LLMConfig`, `PhoneAgent` (litellm 호출, 출력 정리) |
| `phone_call.py` | 통화 루프, 종료 판정, 기록 저장 |
| `main.py` | CLI |

## 알려진 한계

- 4B 4비트 모델은 맥락을 놓치거나 상대의 말투와 이름을 따라 하는 일이 잦다. 9B가 훨씬 낫다.
- 9B 서버는 턴당 지연이 길고 들쭉날쭉하다(관측: 7~74초, 프롬프트 약 800~1100토큰). 한 통화에 몇 분이 걸린다. `--max-concurrent 1` 서버라 다른 요청과 겹치면 더 느려질 수 있다.
- 페르소나 두 명은 서로 무관하게 무작위로 뽑힌다. 통화 목적은 어떤 사람에게나 걸 수 있는 내용(안부, 설문, 영업 등)이어야 자연스럽다. 예를 들어 "치과에 예약 문의"처럼 수신자의 직업을 전제하는 목적은 수신자가 건물 청소원일 수 있어 어긋난다.
