"""
RL Episode Inspector — Gradio 6 compatible
Supports: JSON episode log, JSONL (one step per line), live FastAPI backend
"""
import gradio as gr
import json
import requests

# ── Placeholder data ───────────────────────────────────────────────────────────
PLACEHOLDER_STEPS = [
    {
        "step": 1,
        "observation": {
            "task_stage": "data_validation",
            "position_state": {"current_position": -123, "target_position": 163,
                               "tolerance": 7, "tracking_error": 286, "recent_slippage_bps": 0.0},
            "system_status": {"oms_connected": True, "strategy_status": "paused",
                              "escalated": False, "current_broker": "broker_alpha"},
            "order_state": {"outstanding_orders": [], "open_order_count": 0, "recent_fills": []},
            "data_validation": {"ready": False, "consistent": True,
                                "issues": ["missing_tool:oms_position_check",
                                           "missing_field:bloomberg_pull:mid_price"]},
            "execution_status": {"ready": False, "tracking_error": 286},
        },
        "available_actions": ["CALL_TOOL", "DECLARE", "RESTART_STRATEGY", "ESCALATE",
                              "SUBMIT_ORDER", "SPLIT_ORDER", "CANCEL_ORDER", "CHANGE_BROKER"],
        "action": {"action_type": "CALL_TOOL", "tool_name": "bloomberg_pull"},
        "action_str": "call_tool('bloomberg_pull')",
        "hidden_state": {
            "completed_flags": {"data_ready": False, "systems_ready": False, "execution_complete": False},
            "issue_log": ["Strategy status is paused."],
            "event": {"last_tool_result": {"ok": True, "timestamp": 0, "volume": 187977},
                      "tool_failure": False, "useful_tool": True, "redundant_tool": False},
        },
        "reward": -0.1725, "done": False, "error": None,
        "prompts": {
            "system": "You are controlling a three-stage execution desk environment.\nChoose exactly one next action.",
            "user": '{"task_stage":"data_validation","step":1}',
            "model_output": '{\n  "action_type": "CALL_TOOL",\n  "tool_name": "bloomberg_pull"\n}',
        },
        "grader": {},
    }
]
PH_META  = {"task_name": "execution-desk-assistant", "benchmark": "openenv_execution_desk", "model": "placeholder"}
PH_FINAL = {"success": False, "score": 0.0, "steps": 1, "rewards": [-0.1725]}

# ── Mutable state ──────────────────────────────────────────────────────────────
_S = {"steps": PLACEHOLDER_STEPS.copy(), "meta": PH_META.copy(),
      "final": PH_FINAL.copy(), "source": "placeholder", "api_url": ""}


# ════════════════════════════════════════════════════════════════════════════════
#  FILE LOADING
# ════════════════════════════════════════════════════════════════════════════════
def load_episode(file_obj):
    if file_obj is None:
        _S.update(steps=PLACEHOLDER_STEPS.copy(), meta=PH_META.copy(),
                  final=PH_FINAL.copy(), source="placeholder")
        return render_step(0)
    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(path) as f:
            raw = f.read().strip()

        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        steps, meta, final = [], {}, {}

        if len(lines) == 1:
            obj = json.loads(lines[0])
            if "steps" in obj:           # full episode JSON
                steps = obj["steps"];  meta = obj.get("meta", {}); final = obj.get("final", {})
            elif "step" in obj:           # single-step JSON
                steps = [obj]
            else:
                raise ValueError("Unrecognised JSON structure.")
        else:
            # JSONL: each line is one step (or optional meta/final line)
            for line in lines:
                obj = json.loads(line)
                if "step" in obj:
                    steps.append(obj)
                elif "meta" in obj and not meta:
                    meta = obj["meta"]
                elif ("success" in obj or "score" in obj) and not final:
                    final = obj

        if not steps:
            raise ValueError("No step data found in file.")

        if not meta:
            meta = {"task_name": "unknown", "benchmark": "—", "model": "—"}
        if not final:
            rewards = [s.get("reward", 0) for s in steps]
            final = {"success": steps[-1].get("done", False),
                     "score": round(sum(rewards), 4),
                     "steps": len(steps), "rewards": rewards}

        _S["steps"] = steps; _S["meta"] = meta; _S["final"] = final; _S["source"] = "file"
        return render_step(0)
    except Exception as e:
        return [f"❌ Load error: {e}"] + [""] * 11


# ════════════════════════════════════════════════════════════════════════════════
#  LIVE BACKEND
# ════════════════════════════════════════════════════════════════════════════════
def _norm(resp: dict, n: int) -> dict:
    obs = resp.get("observation", resp.get("obs", {}))
    return {"step": n, "observation": obs if isinstance(obs, dict) else {},
            "available_actions": resp.get("available_actions", []),
            "action": resp.get("action", {}), "action_str": resp.get("action_str", "—"),
            "hidden_state": resp.get("hidden_state", {}), "reward": resp.get("reward", 0.0),
            "done": resp.get("done", False), "error": resp.get("error"),
            "prompts": resp.get("prompts", {}), "grader": resp.get("grader", {})}

def connect_backend(url):
    try:
        url = url.strip().rstrip("/")
        r = requests.post(f"{url}/reset", timeout=10); r.raise_for_status()
        data = r.json()
        _S["steps"]  = [_norm(data, 1)]
        _S["meta"]   = data.get("meta", {"task_name": "live", "benchmark": url, "model": "—"})
        _S["final"]  = {}; _S["source"] = "live"; _S["api_url"] = url
        return render_step(0) + [f"✅ Connected to `{url}` — reset OK."]
    except Exception as e:
        return render_step(0) + [f"❌ Failed: {e}"]

def send_action(action_str, idx):
    try:
        r = requests.post(f"{_S['api_url']}/step", json=json.loads(action_str), timeout=10)
        r.raise_for_status()
        _S["steps"].append(_norm(r.json(), len(_S["steps"]) + 1))
        return render_step(len(_S["steps"]) - 1)
    except Exception as e:
        return [f"❌ Step error: {e}"] + [""] * 11


# ════════════════════════════════════════════════════════════════════════════════
#  RENDER
# ════════════════════════════════════════════════════════════════════════════════
def render_step(idx):
    steps = _S["steps"]
    if not steps:
        return ["No steps"] + [""] * 11
    idx = max(0, min(idx, len(steps) - 1))
    s = steps[idx]; meta = _S["meta"]

    # meta bar
    meta_md = (f"**Source:** `{_S['source']}`  ·  **Task:** `{meta.get('task_name','—')}`  ·  "
               f"**Benchmark:** `{meta.get('benchmark','—')}`  ·  **Model:** `{meta.get('model','—')}`  ·  "
               f"**Step:** {s.get('step', idx+1)} / {len(steps)}")

    # ── Observation ───────────────────────────────────────────────────────────
    obs = s.get("observation", {})
    ps  = obs.get("position_state", {})
    ss  = obs.get("system_status", {})
    dv  = obs.get("data_validation", {})
    os_ = obs.get("order_state", {})

    obs_md = f"### 🗺 Observation\n\n**Stage:** `{obs.get('task_stage','—')}`\n\n"
    if ps:
        obs_md += "#### Position\n| Field | Value |\n|---|---|\n"
        for k, v in ps.items(): obs_md += f"| {k} | `{v}` |\n"
        obs_md += "\n"
    if ss:
        obs_md += "#### System Status\n| Field | Value |\n|---|---|\n"
        for k, v in ss.items(): obs_md += f"| {k} | `{v}` |\n"
        obs_md += "\n"
    if os_:
        obs_md += f"#### Orders\n- Open: `{os_.get('open_order_count',0)}`  · Fills: `{os_.get('recent_fills',[])}`\n\n"
    if dv:
        obs_md += f"#### Data Validation\n- Ready: `{dv.get('ready','—')}` · Consistent: `{dv.get('consistent','—')}`\n"
        iss = dv.get("issues", [])
        if iss: obs_md += "\n**Issues:**\n" + "\n".join(f"- `{i}`" for i in iss)

    # ── Actions ───────────────────────────────────────────────────────────────
    avail = s.get("available_actions", [])
    actions_md = (f"### ⚡ Actions\n\n**Available:**  \n{' · '.join(f'`{a}`' for a in avail)}\n\n"
                  f"**Chosen:**\n```\n{s.get('action_str','—')}\n```\n\n"
                  f"**Detail:**\n```json\n{json.dumps(s.get('action',{}), indent=2)}\n```\n")

    # ── Hidden State ──────────────────────────────────────────────────────────
    hs = s.get("hidden_state", {})
    flags = hs.get("completed_flags", {})
    ev    = hs.get("event", {})
    sr    = hs.get("system_readiness", {})
    fi = lambda v: "✅" if v else "❌"

    hidden_md = "### 🔒 Hidden State\n\n"
    if flags:
        hidden_md += "#### Completion Flags\n| Flag | Status |\n|---|---|\n"
        for k, v in flags.items(): hidden_md += f"| {k} | {fi(v)} |\n"
        hidden_md += "\n"
    if ev:
        hidden_md += "#### Last Tool Event\n| Field | Value |\n|---|---|\n"
        for k, v in ev.items():
            if k != "last_tool_result": hidden_md += f"| {k} | `{v}` |\n"
        ltr = ev.get("last_tool_result", {})
        if ltr: hidden_md += f"\n**Tool Result:**\n```json\n{json.dumps(ltr, indent=2)}\n```\n"
    il = hs.get("issue_log", [])
    if il: hidden_md += "\n**Issue Log:**\n" + "\n".join(f"- {i}" for i in il)
    sri = sr.get("issues", [])
    if sri: hidden_md += "\n\n**System Readiness:**\n" + "\n".join(f"- `{i}`" for i in sri)

    # ── Reward ────────────────────────────────────────────────────────────────
    reward = s.get("reward", 0); done = s.get("done", False)
    error  = s.get("error"); grader = s.get("grader", {}); final = _S["final"]
    ri = "🟢" if reward >= 0 else "🔴"

    reward_md = (f"### 🏆 Reward & Grader\n\n| Field | Value |\n|---|---|\n"
                 f"| Reward | {ri} `{reward}` |\n| Done | `{done}` |\n"
                 f"| Error | `{error if error else 'None'}` |\n\n"
                 f"#### Task\n| Field | Value |\n|---|---|\n"
                 f"| Name | `{meta.get('task_name','—')}` |\n"
                 f"| Step | `{s.get('step',idx+1)}` of `{len(steps)}` |\n")
    if final:
        reward_md += (f"\n#### Episode Final\n| Field | Value |\n|---|---|\n"
                      f"| Success | `{final.get('success','—')}` |\n"
                      f"| Score | `{final.get('score','—')}` |\n"
                      f"| Rewards | `{final.get('rewards',[])}` |\n")
    reward_md += (f"\n**Grader:**\n```json\n{json.dumps(grader, indent=2)}\n```"
                  if grader else "\n*No grader output.*")

    # ── Prompts ───────────────────────────────────────────────────────────────
    pr = s.get("prompts", {})
    sys_md  = f"### 💬 System Prompt\n```\n{pr.get('system','—')}\n```"
    user_md = (f"### 👤 User Prompt\n```json\n{pr.get('user','—')}\n```\n\n"
               f"### 🤖 Model Output\n```json\n{pr.get('model_output','—')}\n```")
    gp = pr.get("grader_prompt", "")
    if gp: user_md += f"\n\n### 📋 Grader Prompt\n```\n{gp}\n```"

    # ── History ───────────────────────────────────────────────────────────────
    rows = []
    for st in steps[:idx+1]:
        gr_s = json.dumps(st.get("grader", {})) if st.get("grader") else "—"
        rows.append([st.get("step","—"), st.get("action_str","—"), st.get("reward","—"),
                     str(st.get("done","—")), (gr_s[:60]+"…") if len(gr_s)>60 else gr_s])

    return [
        meta_md, obs_md, actions_md, hidden_md, reward_md, sys_md, user_md,
        rows,
        f"Step {s.get('step',idx+1)} / {len(steps)}",
        gr.update(interactive=idx > 0),
        gr.update(interactive=idx < len(steps) - 1),
        idx,
    ]

def step_fwd(idx):  return render_step(min(idx+1, len(_S["steps"])-1))
def step_bwd(idx):  return render_step(max(idx-1, 0))
def reset_v(_):     return render_step(0)
def jump(n, _):     return render_step(int(n)-1)


# ════════════════════════════════════════════════════════════════════════════════
#  CSS  (Gradio 6: passed to launch())
# ════════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
:root{--bg:#0d0f14;--surf:#141720;--surf2:#1c2030;--bdr:#2a2f45;
      --acc:#4fd1c5;--acc2:#f6ad55;--txt:#e2e8f0;--dim:#718096;
      --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;}

body,.gradio-container{background:var(--bg)!important;font-family:var(--sans)!important;color:var(--txt)!important;}

.rl-hdr{background:var(--surf);border-bottom:1px solid var(--bdr);padding:14px 24px;margin-bottom:4px;}
.rl-hdr h1{font-family:var(--mono);font-size:1rem;letter-spacing:.1em;color:var(--acc);margin:0;}
.rl-sub{font-family:var(--mono);font-size:.6rem;color:var(--dim);letter-spacing:.08em;}

/* panels */
.panel{background:var(--surf)!important;border:1px solid var(--bdr)!important;border-radius:6px!important;}
.panel .prose{overflow-y:auto!important;max-height:460px;padding:10px 14px!important;}
.panel pre{overflow-x:auto!important;max-height:260px;white-space:pre!important;}
.panel pre,.panel code{background:#0a0c10!important;border:1px solid var(--bdr)!important;
    border-radius:4px!important;padding:6px 10px!important;color:var(--acc)!important;
    font-size:.71rem!important;font-family:var(--mono)!important;}
.panel h3{font-family:var(--mono);font-size:.8rem;color:var(--acc2);margin:8px 0 4px!important;letter-spacing:.05em;}
.panel h4{font-family:var(--mono);font-size:.68rem;color:var(--dim);margin:6px 0 3px!important;
    text-transform:uppercase;letter-spacing:.06em;}
.panel table{border-collapse:collapse;width:100%;margin-bottom:8px;}
.panel table th{color:var(--dim);font-size:.62rem;text-transform:uppercase;letter-spacing:.05em;
    padding:3px 8px;border-bottom:1px solid var(--bdr);}
.panel table td{padding:4px 8px;border-bottom:1px solid #1e2330;font-size:.72rem;font-family:var(--mono);}
.panel p,.panel li{font-size:.75rem;font-family:var(--mono);}

/* tabs */
.tab-nav button{font-family:var(--mono)!important;font-size:.67rem!important;letter-spacing:.06em!important;
    text-transform:uppercase!important;color:var(--dim)!important;background:transparent!important;
    border-bottom:2px solid transparent!important;padding:7px 14px!important;}
.tab-nav button.selected{color:var(--acc)!important;border-bottom-color:var(--acc)!important;}

/* controls */
.ctrl-row{background:var(--surf2);border:1px solid var(--bdr);border-radius:6px;padding:10px 14px;}
.gr-button{font-family:var(--mono)!important;font-size:.72rem!important;border-radius:4px!important;
    padding:7px 16px!important;letter-spacing:.04em!important;}
.gr-button-primary{background:var(--acc)!important;color:#0d0f14!important;font-weight:600!important;border:none!important;}
.gr-button-secondary{background:transparent!important;color:var(--dim)!important;border:1px solid var(--bdr)!important;}
.gr-button-secondary:hover{border-color:var(--acc)!important;color:var(--acc)!important;}

/* step badge */
.step-badge textarea,.step-badge input{font-family:var(--mono)!important;font-size:.82rem!important;
    color:var(--acc2)!important;background:#1a1500!important;border:1px solid #3d3000!important;
    border-radius:4px!important;text-align:center;}

/* live status */
.live-st{background:#0a0c10;border:1px solid var(--bdr);border-radius:4px;padding:6px 12px;
    font-family:var(--mono);font-size:.72rem;}

/* labels */
label span{font-family:var(--mono)!important;font-size:.62rem!important;
    text-transform:uppercase!important;letter-spacing:.06em!important;color:var(--dim)!important;}

/* table */
.gr-dataframe,table{font-family:var(--mono)!important;font-size:.7rem!important;}
"""

# ════════════════════════════════════════════════════════════════════════════════
#  LAYOUT  (Gradio 6)
# ════════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="RL Episode Inspector") as demo:

    idx_state = gr.State(0)

    gr.HTML("""
    <div class='rl-hdr'>
        <h1>⬡ RL EPISODE INSPECTOR</h1>
        <div class='rl-sub'>REINFORCEMENT LEARNING · ENVIRONMENT DEBUG VIEWER</div>
    </div>""")

    # ── Source selector ───────────────────────────────────────────────────────
    with gr.Tabs():

        with gr.Tab("📂 File  (JSON / JSONL)"):
            with gr.Row():
                file_input = gr.File(
                    label="episode_log.json  or  episode_log.jsonl",
                    file_types=[".json", ".jsonl"],
                    type="filepath",
                )
            gr.Markdown(
                "Supports **full episode JSON** `{meta, steps:[…], final}` "
                "and **JSONL** (one step-object per line). "
                "Placeholder data is shown until a file is loaded."
            )

        with gr.Tab("🔌 Live FastAPI Backend"):
            gr.Markdown(
                "Connect to a running `uvicorn server.app:app` server.  \n"
                "**Connect** calls `/reset` · **Send Action** calls `/step`"
            )
            with gr.Row():
                api_url_in  = gr.Textbox(label="Backend URL", placeholder="http://localhost:8000")
                btn_connect = gr.Button("⚡ Connect & Reset", variant="primary")
            live_status = gr.Markdown("*Not connected.*", elem_classes=["live-st"])

            gr.Markdown("#### Send Action")
            with gr.Row():
                action_in     = gr.Textbox(
                    label='Action JSON',
                    placeholder='{"action_type": "CALL_TOOL", "tool_name": "bloomberg_pull"}',
                    lines=2,
                )
                btn_live_step = gr.Button("▶ Send", variant="primary")

    # ── Meta ──────────────────────────────────────────────────────────────────
    meta_out = gr.Markdown("*Load a file or connect to a backend.*")

    # ── Controls ──────────────────────────────────────────────────────────────
    with gr.Row(elem_classes=["ctrl-row"]):
        btn_reset = gr.Button("⟳ Reset View")
        btn_prev  = gr.Button("◀ Prev")
        btn_next  = gr.Button("Next ▶", variant="primary")
        step_lbl  = gr.Textbox(value="Step — / —", interactive=False,
                               show_label=False, min_width=130, elem_classes=["step-badge"])
        step_num  = gr.Number(label="Jump to step", value=1, minimum=1, precision=0, min_width=130)
        btn_jump  = gr.Button("Go")

    # ── Panels ────────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Observation"):
                    obs_out = gr.Markdown(elem_classes=["panel"])
                with gr.Tab("Actions"):
                    act_out = gr.Markdown(elem_classes=["panel"])

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Hidden State"):
                    hid_out = gr.Markdown(elem_classes=["panel"])
                with gr.Tab("Reward & Grader"):
                    rew_out = gr.Markdown(elem_classes=["panel"])

        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.Tab("System Prompt"):
                    sys_out  = gr.Markdown(elem_classes=["panel"])
                with gr.Tab("User / Model"):
                    user_out = gr.Markdown(elem_classes=["panel"])

    # ── History ───────────────────────────────────────────────────────────────
    gr.HTML("<div style='margin-top:14px;font-family:var(--mono,monospace);font-size:.63rem;"
            "color:#718096;letter-spacing:.07em;text-transform:uppercase;padding:4px 0;'>"
            "Step History</div>")
    hist_tbl = gr.Dataframe(
        headers=["Step", "Action", "Reward", "Done", "Grader Result"],
        datatype=["number", "str", "number", "str", "str"],
        row_count=(5, "dynamic"),
        interactive=False,
    )

    # ── Output list (12 items, matches render_step) ───────────────────────────
    OUTS = [meta_out, obs_out, act_out, hid_out, rew_out, sys_out, user_out,
            hist_tbl, step_lbl, btn_prev, btn_next, idx_state]

    # ── Events ────────────────────────────────────────────────────────────────
    file_input.change(fn=load_episode, inputs=file_input, outputs=OUTS)
    btn_reset.click(fn=reset_v,  inputs=idx_state,            outputs=OUTS)
    btn_prev.click(fn=step_bwd,  inputs=idx_state,            outputs=OUTS)
    btn_next.click(fn=step_fwd,  inputs=idx_state,            outputs=OUTS)
    btn_jump.click(fn=jump,      inputs=[step_num, idx_state], outputs=OUTS)

    def _conn(url):
        res = connect_backend(url)       # 12 panel vals + status string
        return res[:12] + [res[12]]

    btn_connect.click(fn=_conn, inputs=api_url_in, outputs=OUTS + [live_status])
    btn_live_step.click(fn=send_action, inputs=[action_in, idx_state], outputs=OUTS)

    demo.load(fn=lambda: render_step(0), outputs=OUTS)


if __name__ == "__main__":
    demo.launch(css=CSS)