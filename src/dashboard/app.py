import streamlit as st
import json
import plotly.graph_objects as go
import time
import os
import pandas as pd

st.set_page_config(page_title="Axon RL Terminal", layout="wide", page_icon="🤖")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-label { color: #8b949e; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

def load_telemetry():
    if os.path.exists("telemetry.json"):
        try:
            with open("telemetry.json", "r") as f:
                return json.load(f)
        except:
            return None
    return None

if 'pnl_history' not in st.session_state:
    st.session_state.pnl_history = []

st.title("🛰️ Axon Institutional RL Terminal")
st.caption("AI Strategy: H4 Multi-Model Ensemble | Symbol: BTCUSDm")

data = load_telemetry()

if data:
    # Top Row Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Portfolio Balance", f"${data.get('balance', 0.0):,.2f}", delta=f"{data.get('pnl', 0.0):.2f}")
    m2.metric("Unrealized PnL", f"${data.get('pnl', 0.0):.2f}")
    m3.metric("Neural Confidence", f"{data.get('confidence', 0.0)*100:.1f}%")
    pos_val = data.get('action', 0)
    pos_label = "SHORT" if pos_val == -1 else ("LONG" if pos_val == 1 else "FLAT")
    m4.metric("Market Position", pos_label)

    st.markdown("---")
    
    # Middle Row: Visuals
    left, mid, right = st.columns([1.5, 1, 1])
    
    with left:
        st.markdown("### 📊 Market Context")
        # Simple Price + SL/TP Chart
        price = data.get('price', 0.0)
        entry = data.get('entry', 0.0)
        sl = data.get('sl', 0.0)
        tp = data.get('tp', 0.0)
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Indicator(
            mode = "number+delta",
            value = price,
            title = {"text": "BTCUSDm Live Price"},
            delta = {'reference': entry, 'relative': False} if entry > 0 else None
        ))
        fig_price.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_price, width='stretch', key="price_indicator")

    with mid:
        st.markdown("### 🧠 Ensemble Policy")
        probs = data.get('probs', [0.33, 0.33, 0.34])
        st.progress(probs[1], text=f"BUY Probability: {probs[1]*100:.1f}%")
        st.progress(probs[2], text=f"SELL Probability: {probs[2]*100:.1f}%")
        st.progress(probs[0], text=f"HOLD Probability: {probs[0]*100:.1f}%")

    with right:
        st.markdown("### 🔍 SMC Structure")
        inds = data.get('indicators', {})
        st.info(f"ATR (Vol): **{inds.get('ATR', 0.0):.2f}**")
        st.success(f"BOS (Bullish): **{'YES' if inds.get('BOS') else 'NO'}**")
        st.warning(f"CHOCH (Bearish): **{'YES' if inds.get('CHOCH') else 'NO'}**")

    # Bottom Row: Active Trade Details
    if pos_val != 0:
        st.markdown("---")
        st.markdown("### 🛡️ Institutional Risk Management")
        r1, r2, r3, r4 = st.columns(4)
        r1.write(f"**Entry**: {entry:.2f}")
        r2.write(f"**Current**: {price:.2f}")
        r3.error(f"**Stop Loss**: {sl:.2f}")
        r4.success(f"**Take Profit**: {tp:.2f}")
        
        # PnL Gauge moved here
        pnl = data.get('pnl', 0.0)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pnl,
            gauge = {'axis': {'range': [-50, 50]}, 'bar': {'color': "#00d4ff"}},
            title = {'text': "Open PnL ($)"}
        ))
        fig_gauge.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, width='stretch', key="pnl_gauge_bottom")

# Auto-refresh
time.sleep(1)
st.rerun()
