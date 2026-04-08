import gradio as gr
import json
import os

# ── Default / placeholder data ────────────────────────────────────────────────
PLACEHOLDER = {
    "meta": {
        "task_name": "execution-desk-assistant",
        "benchmark": "openenv_execution_desk",
        "model": "Qwen/Qwen2.5-72B-Instruct",
    },
    "steps": [
        {
            "step": 1,
            "observation": {
                "task_stage": "data_validation",
                "position_state": {
                    "current_position": -123,
                    "target_position": 163,
                    "tolerance": 7,
                    "tracking_error": 286,
                    "recent_slippage_bps": 0.0,
                },
                "system_status": {
                    "oms_connected": True,
                    "strategy_status": "paused",
                    "escalated": False,
                    "current_broker": "broker_alpha",
                },
                "order_state": {
                    "outstanding_orders": [],
                    "open_order_count": 0,
                    "recent_fills": [],
                },
                "data_validation": {
                    "ready": False,
                    "issues": [
                        "missing_tool:oms_position_check",
                        "missing_tool:risk_system_check",
                        "missing_field:bloomberg_pull:mid_price",
                    ],
                    "consistent": True,
                },
                "execution_status": {"ready": False, "tracking_error": 286},
            },
            "available_actions": [
                "CALL_TOOL","DECLARE","RESTART_STRATEGY","ESCALATE",
                "SUBMIT_ORDER","SPLIT_ORDER","CANCEL_ORDER","CHANGE_BROKER",
            ],
            "action": {"action_type": "CALL_TOOL", "tool_name": "bloomberg_pull"},
            "action_str": "call_tool('bloomberg_pull')",
            "hidden_state": {
                "completed_flags": {
                    "data_ready": False,
                    "systems_ready": False,
                    "execution_complete": False,
                },
                "issue_log": ["Strategy status is paused."],
                "event": {
                    "last_tool_result": {"ok": True, "timestamp": 0, "volume": 187977},
                    "tool_failure": False,
                    "useful_tool": True,
                    "redundant_tool": False,
                },
            },
            "reward": -0.1725,
            "done": False,
            "error": None,
            "prompts": {
                "system": "You are controlling a three-stage execution desk environment.\nChoose exactly one next action.",
                "user": '{"task_stage":"data_validation","step":1}',
                "model_output": '{\n  "action_type": "CALL_TOOL",\n  "tool_name": "bloomberg_pull"\n}',
            },
            "grader": {},
        }
    ],
    "final": {"success": False, "score": 0.0, "steps": 1, "rewards": [-0.1725]},
}

# ── State ──────────────────────────────────────────────────────────────────────
episode_data = {"data": PLACEHOLDER}
current_step_idx = {"idx": 0}


def load_episode(file_obj):
    try:
        if file_obj is None:
            episode_data["data"] = PLACEHOLDER
        else:
            path = file_obj.name if hasattr(file_obj, "name") else file_obj

            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]

            # Detect JSONL vs JSON
            if lines and lines[0].startswith("{") and len(lines) > 1:
                # JSONL case
                steps = [json.loads(line) for line in lines]
                episode_data["data"] = {
                    "meta": PLACEHOLDER["meta"],  # fallback meta
                    "steps": steps,
                    "final": {
                        "success": False,
                        "score": sum(s.get("reward", 0) for s in steps),
                        "steps": len(steps),
                        "rewards": [s.get("reward", 0) for s in steps],
                    },
                }
            else:
                # normal JSON
                episode_data["data"] = json.loads(lines[0])

        current_step_idx["idx"] = 0
        return render_step(0)

    except Exception as e:
        return [f"❌ Error loading file: {e}"] + [""] * 11


def total_steps():
    return len(episode_data["data"].get("steps", []))


def render_step(idx):
    data = episode_data["data"]
    steps = data.get("steps", [])
    if not steps:
        return ["No steps found"] + [""] * 11

    idx = max(0, min(idx, len(steps) - 1))
    s = steps[idx]
    meta = data.get("meta", {})

    # ── Meta bar ──────────────────────────────────────────────────────────────
    meta_md = (
        f"**Task:** `{meta.get('task_name','—')}`  ·  "
        f"**Benchmark:** `{meta.get('benchmark','—')}`  ·  "
        f"**Model:** `{meta.get('model','—')}`  ·  "
        f"**Step:** {s['step']} / {len(steps)}"
    )

    # ── Observation panel ─────────────────────────────────────────────────────
    obs = s.get("observation", {})
    ps = obs.get("position_state", {})
    ss = obs.get("system_status", {})
    dv = obs.get("data_validation", {})
    os_ = obs.get("order_state", {})
    es = obs.get("execution_status", {})

    obs_md = f"""### 🗺 Agent Observation

**Stage:** `{obs.get('task_stage','—')}`

#### Position State
| Field | Value |
|---|---|
| Current Position | `{ps.get('current_position','—')}` |
| Target Position | `{ps.get('target_position','—')}` |
| Tolerance | `{ps.get('tolerance','—')}` |
| Tracking Error | `{ps.get('tracking_error','—')}` |
| Recent Slippage (bps) | `{ps.get('recent_slippage_bps','—')}` |

#### System Status
| Field | Value |
|---|---|
| OMS Connected | `{ss.get('oms_connected','—')}` |
| Strategy Status | `{ss.get('strategy_status','—')}` |
| Escalated | `{ss.get('escalated','—')}` |
| Current Broker | `{ss.get('current_broker','—')}` |

#### Order State
- Open orders: `{os_.get('open_order_count', 0)}`
- Recent fills: `{os_.get('recent_fills', [])}`

#### Data Validation
- Ready: `{dv.get('ready','—')}` · Consistent: `{dv.get('consistent','—')}`
"""
    issues = dv.get("issues", [])
    if issues:
        obs_md += "\n**Issues:**\n" + "\n".join(f"- `{i}`" for i in issues)

    # ── Available actions + chosen action ─────────────────────────────────────
    avail = s.get("available_actions", [])
    chosen = s.get("action_str", "—")
    action_detail = s.get("action", {})

    actions_md = f"""### ⚡ Actions

**Available:**  
{" · ".join(f"`{a}`" for a in avail)}

**Chosen Action:**  
```
{chosen}
```

**Action Detail:**
```json
{json.dumps(action_detail, indent=2)}
```
"""

    # ── Hidden state panel ────────────────────────────────────────────────────
    hs = s.get("hidden_state", {})
    flags = hs.get("completed_flags", {})
    ev = hs.get("event", {})
    sr = hs.get("system_readiness", {})

    def flag_icon(v):
        return "✅" if v else "❌"

    hidden_md = f"""### 🔒 Hidden State

#### Completion Flags
| Flag | Status |
|---|---|
| Data Ready | {flag_icon(flags.get('data_ready'))} |
| Systems Ready | {flag_icon(flags.get('systems_ready'))} |
| Execution Complete | {flag_icon(flags.get('execution_complete'))} |

#### Last Tool Event
| Field | Value |
|---|---|
| Tool Failure | `{ev.get('tool_failure','—')}` |
| Useful Tool | `{ev.get('useful_tool','—')}` |
| Redundant Tool | `{ev.get('redundant_tool','—')}` |
| Found Inconsistency | `{ev.get('found_inconsistency','—')}` |

**Last Tool Result:**
```json
{json.dumps(ev.get('last_tool_result', {}), indent=2)}
```
"""
    issue_log = hs.get("issue_log", [])
    if issue_log:
        hidden_md += "\n**Issue Log:**\n" + "\n".join(f"- {i}" for i in issue_log)

    sr_issues = sr.get("issues", [])
    if sr_issues:
        hidden_md += "\n\n**System Readiness Issues:**\n" + "\n".join(f"- `{i}`" for i in sr_issues)

    # ── Reward & grader panel ─────────────────────────────────────────────────
    reward = s.get("reward", 0)
    done = s.get("done", False)
    error = s.get("error")
    grader = s.get("grader", {})
    final = data.get("final", {})

    reward_color = "🟢" if reward >= 0 else "🔴"
    reward_md = f"""### 🏆 Reward & Grader

| Field | Value |
|---|---|
| Reward | {reward_color} `{reward}` |
| Done | `{done}` |
| Error | `{error if error else 'None'}` |

#### Task Info
| Field | Value |
|---|---|
| Task Name | `{meta.get('task_name','—')}` |
| Step | `{s['step']}` of `{len(steps)}` |

#### Episode Final (so far)
| Field | Value |
|---|---|
| Success | `{final.get('success','—')}` |
| Score | `{final.get('score','—')}` |
| Total Rewards | `{final.get('rewards',[])}` |
"""
    if grader:
        reward_md += f"\n**Grader Output:**\n```json\n{json.dumps(grader, indent=2)}\n```"
    else:
        reward_md += "\n*No grader output for this step.*"

    # ── Prompt viewer ─────────────────────────────────────────────────────────
    prompts = s.get("prompts", {})

    sys_prompt = f"""### 💬 System Prompt
```
{prompts.get('system', '—')}
```"""

    user_prompt = f"""### 👤 User Prompt
```json
{prompts.get('user', '—')}
```

### 🤖 Model Output
```json
{prompts.get('model_output', '—')}
```"""

    grader_prompt = prompts.get("grader_prompt", "")
    grader_prompt_md = f"\n### 📋 Grader Prompt\n```\n{grader_prompt}\n```" if grader_prompt else ""

    prompt_sys_md = sys_prompt
    prompt_user_md = user_prompt + grader_prompt_md

    # ── Step history table ────────────────────────────────────────────────────
    history_rows = []
    for st in steps[: idx + 1]:
        gr_out = st.get("grader", {})
        gr_str = json.dumps(gr_out) if gr_out else "—"
        history_rows.append(
            [
                st["step"],
                st.get("action_str", "—"),
                st.get("reward", "—"),
                str(st.get("done", "—")),
                gr_str[:60] + ("…" if len(gr_str) > 60 else ""),
            ]
        )

    step_label = f"Step {s['step']} / {len(steps)}"
    return [
        meta_md,
        obs_md,
        actions_md,
        hidden_md,
        reward_md,
        prompt_sys_md,
        prompt_user_md,
        history_rows,
        step_label,
        gr.update(interactive=idx > 0),
        gr.update(interactive=idx < len(steps) - 1),
        idx,
    ]


def step_forward(idx):
    new_idx = min(idx + 1, total_steps() - 1)
    current_step_idx["idx"] = new_idx
    return render_step(new_idx)


def step_backward(idx):
    new_idx = max(idx - 1, 0)
    current_step_idx["idx"] = new_idx
    return render_step(new_idx)


def reset_episode(idx):
    current_step_idx["idx"] = 0
    return render_step(0)


def jump_to_step(step_num, idx):
    new_idx = int(step_num) - 1
    current_step_idx["idx"] = new_idx
    return render_step(new_idx)


# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #141720;
    --surface2: #1c2030;
    --border: #2a2f45;
    --accent: #4fd1c5;
    --accent2: #f6ad55;
    --accent3: #fc8181;
    --text: #e2e8f0;
    --text-dim: #718096;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}

/* header */
.rl-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    margin-bottom: 0;
}

.rl-header h1 {
    font-family: var(--mono);
    font-size: 1.1rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    margin: 0;
}

/* panels */
.panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0 !important;
}

/* tabs */
.tab-nav button {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
}

.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* markdown inside panels */
.panel .prose, .panel p, .panel table, .panel pre, .panel code {
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    color: var(--text) !important;
}

.panel table {
    border-collapse: collapse;
    width: 100%;
}

.panel table th {
    color: var(--text-dim);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 4px 8px;
    border-bottom: 1px solid var(--border);
    text-align: left;
}

.panel table td {
    padding: 5px 8px;
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
}

.panel pre, .panel code {
    background: #0a0c10 !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
    color: var(--accent) !important;
    font-size: 0.75rem !important;
    white-space: pre-wrap !important;
}

/* control bar */
.ctrl-bar {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* buttons */
button.primary-btn, .gr-button-primary {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 8px 18px !important;
}

button.secondary-btn, .gr-button-secondary {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 8px 18px !important;
}

button.secondary-btn:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* step counter badge */
.step-badge {
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--accent2);
    background: #1a1500;
    border: 1px solid #3d3000;
    border-radius: 4px;
    padding: 4px 12px;
}

/* dataframe */
.gr-dataframe table {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
}

.gr-dataframe th {
    background: var(--surface2) !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* file upload */
.gr-file-upload {
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
    background: var(--surface) !important;
}

/* meta bar */
.meta-bar {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    padding: 8px 16px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    border-radius: 6px 6px 0 0;
}

/* label override */
label span, .gr-form label {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-dim) !important;
}
/* FIX: prevent prompt panels from breaking layout */
.panel {
    display: flex;
    flex-direction: column;
    height: 100%;
}

/* scrollable content inside panels */
.panel .prose {
    overflow-y: auto !important;
    max-height: 420px;
    padding: 10px 12px !important;
}

/* FIX: long code blocks (system prompt etc.) */
.panel pre {
    overflow-x: auto !important;
    white-space: pre !important;
    max-height: 300px;
}

/* FIX: markdown spacing */
.panel h3 {
    margin-top: 6px !important;
    margin-bottom: 6px !important;
    font-size: 0.85rem;
}

/* FIX: tabs height consistency */
.gr-tabitem {
    height: 480px;
    display: flex;
    flex-direction: column;
}

/* FIX: prevent content jumping */
.gradio-container .row {
    align-items: stretch !important;
}
"""

# ── Layout ─────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="RL Episode Inspector") as demo:

    # internal state
    step_idx_state = gr.State(0)

    gr.HTML("""
    <div class='rl-header'>
        <h1>⬡ RL EPISODE INSPECTOR</h1>
        <span style='font-family:var(--mono);font-size:0.65rem;color:var(--text-dim);letter-spacing:0.08em;'>
            REINFORCEMENT LEARNING · ENVIRONMENT DEBUG VIEWER
        </span>
    </div>
    """)

    # ── Top: file load + meta ──────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="Load episode_log.jsonl",
                file_types=[".json", "jsonl"],
                type="filepath",
            )
        with gr.Column(scale=5):
            meta_display = gr.Markdown(
                "Upload an `episode_log.jsonl` or use placeholder data.",
                elem_classes=["meta-bar"],
            )

    # ── Controls ───────────────────────────────────────────────────────────────
    with gr.Row(elem_classes=["ctrl-bar"]):
        btn_reset = gr.Button("⟳ Reset", elem_classes=["secondary-btn"])
        btn_prev  = gr.Button("◀ Prev",  elem_classes=["secondary-btn"])
        btn_next  = gr.Button("Next ▶",  elem_classes=["primary-btn"])
        step_label_out = gr.Textbox(
            value="Step — / —",
            interactive=False,
            show_label=False,
            elem_classes=["step-badge"],
            scale=0,
            min_width=120,
        )
        step_jump = gr.Number(
            label="Jump to step",
            value=1,
            minimum=1,
            precision=0,
            scale=0,
            min_width=130,
        )
        btn_jump = gr.Button("Go", elem_classes=["secondary-btn"], scale=0)

    # ── Main panels ────────────────────────────────────────────────────────────
    with gr.Row():
        # Left column
        with gr.Column(scale=3):
            with gr.Tab("Observation"):
                obs_panel    = gr.Markdown(elem_classes=["panel"])
            with gr.Tab("Actions"):
                action_panel = gr.Markdown(elem_classes=["panel"])

        # Middle column
        with gr.Column(scale=3):
            with gr.Tab("Hidden State"):
                hidden_panel = gr.Markdown(elem_classes=["panel"])
            with gr.Tab("Reward & Grader"):
                reward_panel = gr.Markdown(elem_classes=["panel"])

        # Right column – prompts
        with gr.Column(scale=4):
            with gr.Tab("System Prompt"):
                prompt_sys_panel  = gr.Markdown(elem_classes=["panel"])
            with gr.Tab("User / Model"):
                prompt_user_panel = gr.Markdown(elem_classes=["panel"])

    # ── Step history ───────────────────────────────────────────────────────────
    gr.HTML("<div style='margin-top:12px;font-family:var(--mono);font-size:0.7rem;color:var(--text-dim);letter-spacing:0.06em;text-transform:uppercase;'>Step History</div>")
    history_table = gr.Dataframe(
        headers=["Step", "Action", "Reward", "Done", "Grader Result"],
        datatype=["number", "str", "number", "str", "str"],
        row_count=(5, "dynamic"),
        col_count=(5, "fixed"),
        interactive=False,
    )

    # ── All outputs (ordered to match render_step return) ─────────────────────
    all_outputs = [
        meta_display,
        obs_panel,
        action_panel,
        hidden_panel,
        reward_panel,
        prompt_sys_panel,
        prompt_user_panel,
        history_table,
        step_label_out,
        btn_prev,
        btn_next,
        step_idx_state,
    ]

    # ── Wire events ────────────────────────────────────────────────────────────
    file_input.change(fn=load_episode, inputs=file_input, outputs=all_outputs)

    btn_reset.click(fn=reset_episode, inputs=step_idx_state, outputs=all_outputs)

    btn_prev.click(fn=step_backward, inputs=step_idx_state, outputs=all_outputs)

    btn_next.click(fn=step_forward, inputs=step_idx_state, outputs=all_outputs)

    btn_jump.click(fn=jump_to_step, inputs=[step_jump, step_idx_state], outputs=all_outputs)

    # Load placeholder on startup
    demo.load(fn=lambda: render_step(0), outputs=all_outputs)


if __name__ == "__main__":
    demo.launch()