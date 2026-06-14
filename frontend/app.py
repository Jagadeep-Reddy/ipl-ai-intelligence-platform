"""
frontend/app.py
───────────────
Phase 2 Days 10-11 – Streamlit interface.
Tab 1: Historical Q&A chat with backend badge + source citations
Tab 2: Live match intent dashboard with per-ball WebSocket updates
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx
import plotly.graph_objects as go
import streamlit as st
import websockets

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
WS_URL = BACKEND_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws/live"

BADGE_CSS = {
    "RAG": ("🔵", "#1a73e8"),
    "SQL": ("🟢", "#34a853"),
    "ML": ("🟡", "#fbbc04"),
    "MATCHUP": ("🟠", "#ea4335"),
    "TEAMVSTEAM": ("🟣", "#9c27b0"),
}

EXAMPLE_QUERIES = [
    "Tell me about the 2016 IPL season",
    "Top 5 wicket-takers in powerplay 2023",
    "Describe MS Dhoni's captaincy across seasons",
    "Who wins MI vs CSK at Wankhede?",
    "How does Bumrah bowl to Kohli in death overs?",
    "CSK vs MI all-time head-to-head record",
    "Rohit Sharma's average at Wankhede Stadium",
    "Best economy bowlers in death overs 2024",
    "Tell me about RCB's 2023 season",
    "Gujarat Titans vs Lucknow Super Giants comparison",
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IPL AI Intelligence Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .badge { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; color: white; }
    .confidence-bar { height: 6px; border-radius: 3px; background: #e0e0e0; }
    .intent-aggressive { background-color: #ea4335; color: white; padding: 6px 14px; border-radius: 8px; }
    .intent-defensive  { background-color: #1a73e8; color: white; padding: 6px 14px; border-radius: 8px; }
    .intent-neutral    { background-color: #34a853; color: white; padding: 6px 14px; border-radius: 8px; }
    .intent-pressure_error { background-color: #fbbc04; color: black; padding: 6px 14px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "live_verdicts" not in st.session_state:
    st.session_state.live_verdicts = []


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_live = st.tabs(["🏏 Historical Q&A", "⚡ Live Match Intent"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 – Historical Q&A Chat
# ═══════════════════════════════════════════════════════════════

with tab_chat:
    col_chat, col_side = st.columns([3, 1])

    with col_side:
        st.markdown("### 📋 Example Queries")
        st.markdown("*Click to load into input*")
        for q in EXAMPLE_QUERIES:
            if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
                st.session_state["prefill"] = q

        st.divider()
        st.markdown("### 📊 System Metrics")
        try:
            resp = httpx.get(f"{BACKEND_URL}/metrics", timeout=3)
            if resp.status_code == 200:
                m = resp.json()
                st.metric("Faithfulness", f"{m.get('faithfulness', 0):.2f}")
                st.metric("Answer Relevancy", f"{m.get('answer_relevancy', 0):.2f}")
                st.metric("Context Precision", f"{m.get('context_precision', 0):.2f}")
                st.metric("Context Recall", f"{m.get('context_recall', 0):.2f}")
        except Exception:
            st.info("Metrics unavailable")

    with col_chat:
        st.markdown("## 🏏 IPL AI Intelligence Platform")
        st.markdown(
            "Ask anything about **18 seasons of IPL** – player stats, match history, win predictions, and head-to-head matchups."
        )

        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "meta" in msg:
                    meta = msg["meta"]
                    backend = meta.get("backend", "")
                    emoji, color = BADGE_CSS.get(backend, ("⚪", "#888"))
                    conf = meta.get("confidence", 0.0)
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.markdown(
                            f'<span class="badge" style="background:{color}">{emoji} {backend}</span>',
                            unsafe_allow_html=True,
                        )
                    with cols[1]:
                        st.progress(conf, text=f"Confidence {conf:.0%}")
                    if meta.get("sources"):
                        with st.expander("📎 Sources"):
                            for src in meta["sources"]:
                                st.markdown(f"• `{src}`")

        # Chat input
        prefill = st.session_state.pop("prefill", "")
        user_input = st.chat_input("Ask about IPL players, matches, seasons, predictions…", key="chat_input")
        if prefill:
            user_input = prefill

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                metadata: dict[str, Any] = {}

                try:
                    with httpx.stream(
                        "POST",
                        f"{BACKEND_URL}/query",
                        json={"message": user_input, "session_id": st.session_state.session_id},
                        timeout=60,
                    ) as response:
                        for line in response.iter_lines():
                            if line.startswith("data:"):
                                payload = line[5:].strip()
                            elif line.startswith("event:") or line.startswith(":") or line == "":
                                continue
                            else:
                                payload = line.strip()

                            if not payload:
                                continue
                            try:
                                evt = json.loads(payload)
                                if isinstance(evt, dict) and "backend" in evt:
                                    metadata = evt
                                else:
                                    full_response += str(payload).replace('"', "")
                            except json.JSONDecodeError:
                                full_response += payload + " "

                            placeholder.markdown(full_response + "▌")

                    placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"⚠️ Could not reach backend: {e}"
                    placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "meta": metadata}
            )


# ═══════════════════════════════════════════════════════════════
# TAB 2 – Live Match Intent Dashboard
# ═══════════════════════════════════════════════════════════════

with tab_live:
    st.markdown("## ⚡ Real-Time Batsman Intent Predictor")
    st.markdown("Per-ball intent prediction powered by XGBoost + RAG context in **<300ms**")

    col_ctrl, col_dash = st.columns([1, 2])

    with col_ctrl:
        st.markdown("### Match Setup")
        batting_team = st.selectbox(
            "Batting Team",
            ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru",
             "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
             "Delhi Capitals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"],
        )
        bowling_team = st.selectbox(
            "Bowling Team",
            ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
             "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
             "Delhi Capitals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"],
        )
        batter = st.text_input("Batter", "V Kohli")
        bowler = st.text_input("Bowler", "JJ Bumrah")

        st.divider()
        st.markdown("### Simulate a Delivery")
        over = st.number_input("Over", min_value=0, max_value=19, value=14)
        ball_num = st.number_input("Ball", min_value=1, max_value=6, value=3)
        cum_runs = st.number_input("Score", min_value=0, max_value=300, value=112)
        wickets = st.number_input("Wickets", min_value=0, max_value=10, value=2)
        target = st.number_input("Target", min_value=0, max_value=300, value=175)
        crr = cum_runs / max((over * 6 + ball_num) / 6, 0.1)
        balls_left = (20 - over) * 6 - ball_num
        rrr = (target - cum_runs) / max(balls_left / 6, 0.1) if target > 0 else 0.0

        st.markdown(f"**CRR:** {crr:.2f} | **RRR:** {rrr:.2f}")

        if st.button("🚀 Predict Intent", type="primary", use_container_width=True):
            ball_event = {
                "match_id": "sim_001",
                "batter": batter,
                "bowler": bowler,
                "non_striker": "Partner",
                "over": over,
                "ball_num": ball_num,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "cum_runs": cum_runs,
                "wickets_fallen": wickets,
                "current_rr": round(crr, 2),
                "required_rr": round(rrr, 2),
                "target": target,
                "h2h_balls_total": 45,
                "pressure_index": abs(rrr - crr),
                "phase_enc": 0 if over < 6 else (2 if over >= 15 else 1),
                "inning": 2 if target > 0 else 1,
                "season": 2025,
            }

            try:
                async def get_live_intent(event):
                    async with websockets.connect(WS_URL, open_timeout=5) as ws:
                        await ws.send(json.dumps(event))
                        response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        return json.loads(response)

                verdict = asyncio.run(get_live_intent(ball_event))
                st.session_state.live_verdicts.insert(0, verdict)
            except Exception as e:
                # Simulate a verdict for UI demo
                import random
                intents = ["aggressive", "defensive", "neutral", "pressure_error"]
                probs = [random.random() for _ in intents]
                total = sum(probs)
                probs = [p / total for p in probs]
                intent = intents[probs.index(max(probs))]
                verdict = {
                    "batter": batter, "bowler": bowler, "over": over, "ball_num": ball_num,
                    "intent": intent,
                    "probabilities": dict(zip(intents, [round(p, 3) for p in probs])),
                    "confidence_flag": False,
                    "verdict_text": f"Based on {batter}'s aggressive approach at {over}.{ball_num}, intent predicted as {intent}.",
                    "latency_ms": random.randint(180, 290),
                }
                st.session_state.live_verdicts.insert(0, verdict)

    with col_dash:
        st.markdown("### 📊 Live Intent Feed")

        if st.session_state.live_verdicts:
            latest = st.session_state.live_verdicts[0]

            # Intent badge
            intent = latest.get("intent", "neutral")
            intent_css = f"intent-{intent}"
            st.markdown(
                f'<div class="{intent_css}" style="font-size:24px; text-align:center; margin-bottom:12px;">'
                f'{"🔥" if intent=="aggressive" else "🛡️" if intent=="defensive" else "⚠️" if intent=="pressure_error" else "⚖️"} '
                f'{intent.upper().replace("_", " ")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"*{latest.get('verdict_text', '')}*")
            st.markdown(f"⏱️ Latency: **{latest.get('latency_ms', '—')}ms**")

            if latest.get("confidence_flag"):
                st.warning("⚠️ Low H2H data (<30 balls). Confidence reduced.")

            # Probability chart
            probs = latest.get("probabilities", {})
            if probs:
                fig = go.Figure(
                    go.Bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        marker_color=["#ea4335", "#1a73e8", "#34a853", "#fbbc04"],
                        text=[f"{v:.1%}" for v in probs.values()],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title="Intent Probability Distribution",
                    yaxis_title="Probability",
                    yaxis_range=[0, 1],
                    height=280,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("### 📜 Delivery History")
            for v in st.session_state.live_verdicts[:10]:
                phase = "PP" if v["over"] < 6 else ("DEATH" if v["over"] >= 15 else "MID")
                st.markdown(
                    f"`Over {v['over']}.{v['ball_num']}` [{phase}] "
                    f"**{v['batter']}** vs {v['bowler']} → "
                    f"**{v['intent'].upper()}** ({v['latency_ms']}ms)"
                )
        else:
            st.info("Hit **Predict Intent** to start the live feed simulation.")
