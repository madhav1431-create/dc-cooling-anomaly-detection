from pathlib import Path
import pandas as pd
import json
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent

def load():
    df = pd.read_parquet(ROOT / "data" / "dashboard_data.parquet")

    with open(ROOT / "outputs" / "thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    with open(ROOT / "outputs" / "event_level_results.csv", "r", encoding="utf-8") as f:
        events = pd.read_csv(f)

    return df, thresholds, events

df, thr, ev = load()
META = ["node", "time", "score_lstm", "score_if", "is_anomaly"]
sensors = [c for c in df.columns if c not in META]
n_episodes = int(ev["episodes"].iloc[0])

# ---------------- header ----------------
st.title("Data Center Cooling Anomaly Monitor")
st.markdown("**Project:** DC Cooling Anomaly Detection")
st.caption("Deep Learning project | Team: 3 members")

# ---------------- sidebar controls ----------------
st.sidebar.header("Controls")
st.sidebar.caption("These controls change the 'Monitor' tab.")
node = st.sidebar.selectbox("Server (node)", sorted(df["node"].unique()),
                            help="Which server to look at. There are 9 servers in the test data.")
model = st.sidebar.radio("AI model", ["LSTM-AE", "IsoForest"],
                         help="LSTM-AE is our deep learning model. IsoForest (Isolation Forest) is the simpler comparison model.")
rule = st.sidebar.selectbox("Alarm sensitivity (threshold rule)", ["95th pct", "99th pct", "mean+3std"],
                            help="How high the score must go before an alarm is raised. 95th pct is the most sensitive "
                                 "(more alarms), 99th pct and mean+3std are stricter (fewer alarms).")
log_scale = st.sidebar.checkbox("Log scale for anomaly score",
                                help="Makes small scores easier to see next to very large spikes.")

key = {"LSTM-AE": "lstm", "IsoForest": "if"}[model]
suffix = {"95th pct": "95", "99th pct": "99", "mean+3std": "mean3std"}[rule]
threshold = thr[f"{key}_{suffix}"]
score_col = f"score_{key}"

d = df[df["node"] == node].sort_values("time").reset_index(drop=True)
d["alarm"] = d[score_col] > threshold

upto = st.sidebar.slider("Replay position", 1, len(d), len(d),
                         help="Drag left to go back in time, then drag right to replay what the monitor would have shown.")
v = d.iloc[:upto]

tab_about, tab_monitor, tab_results = st.tabs(["About this project", "Monitor (replay)", "Model results"])

# ---------------- tab 1: about (how to use) ----------------
with tab_about:
    st.header("What you can do with this dashboard")
    st.markdown(
        "- **Pick a server, a model and an alarm sensitivity** using the Controls on the left.\n"
        "- **Replay time:** drag the Replay position slider to see what the monitor would have shown at any point.\n"
        "- **Watch the anomaly score** in the Monitor tab. When it goes above the dashed line, an alarm is raised. "
        "Red dots are real alerts recorded by the data center.\n"
        "- **Check the sensors** (fans, temperatures, power) behind the score.\n"
        "- **Compare the models** in the Model results tab."
    )

# ---------------- tab 2: monitor ----------------
with tab_monitor:
    st.subheader(f"Server {node}: {model}, {rule}")

    recent = v.iloc[-48:]   # latest 48 windows, about 24 hours of replay time
    now_time = v["time"].iloc[-1]
    n_recent = int(recent["alarm"].sum())
    st.caption(f"Replay time: {now_time:%d %b %Y, %H:%M} UTC. Drag the Replay position slider to see the status change.")
    if n_recent > 0:
        last_alarm = recent.loc[recent["alarm"], "time"].iloc[-1]
        st.error(f"ALERT: {n_recent} alarm windows in the last 24 hours or so of replay. "
                 f"Latest alarm: {last_alarm:%d %b %Y, %H:%M} UTC.")
    else:
        st.success("NORMAL: no alarms in the last 24 hours or so of replay.")

    n_win = len(v)
    n_alarm = int(v["alarm"].sum())
    n_label = int(v["is_anomaly"].sum())
    n_hit = int((v["alarm"] & (v["is_anomaly"] == 1)).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Windows checked so far", n_win, help="How many 6-hour windows the replay has covered so far.")
    c2.metric("Alarm windows", n_alarm, help="Windows whose anomaly score is above the threshold.")
    c3.metric("Labelled anomaly windows", n_label, help="Windows that overlap a real Nagios CRITICAL alert.")

    # ---- summary for the current view ----
    alarm_rate = 100 * n_alarm / n_win if n_win else 0
    lines = [f"- The model raised an alarm on **{n_alarm} of {n_win}** windows ({alarm_rate:.1f}%).",
             f"- **{n_label}** windows overlapped a real alert."]
    if n_label > 0:
        lines.append(f"- Of those real-alert windows, **{n_hit}** ({100 * n_hit / n_label:.0f}%) were also flagged by an alarm.")
    if n_alarm > 0:
        lines.append(f"- Of all alarm windows, only **{n_hit}** ({100 * n_hit / n_alarm:.1f}%) overlapped a real alert, "
                     "so the rest are possible false alarms.")
    st.info("**Summary for this view**\n" + "\n".join(lines))

    st.markdown("#### Anomaly score over time")
    st.caption("The blue line is how unusual each window is. The dashed orange line is the alarm threshold. "
               "Red dots mark windows that overlap a real alert.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v["time"], y=v[score_col], mode="lines", name="anomaly score"))
    a = v[v["is_anomaly"] == 1]
    fig.add_trace(go.Scatter(x=a["time"], y=a[score_col], mode="markers",
                             name="real alert (Nagios)", marker=dict(color="red", size=5)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="orange", annotation_text="alarm threshold")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="anomaly score")
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Sensor readings")
    st.caption("The raw readings behind the score. Pick the sensors you want to see.")
    default = [s for s in ["fan0_0", "p0_vdd_temp", "total_power"] if s in sensors]
    chosen = st.multiselect("Choose sensors", sensors, default=default)
    if chosen:
        plot_df = v.set_index("time")[chosen]
        if st.checkbox("Rescale sensors to 0-1 so they fit on one chart", value=True):
            plot_df = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-9)
        fig2 = go.Figure()
        for c in chosen:
            fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df[c], mode="lines", name=c))
        fig2.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- tab 3: results ----------------
with tab_results:
    st.subheader(f"How well did the models do? ({n_episodes} labelled episodes in the test period)")

    def pick(m, t):
        r = ev[(ev["model"] == m) & (ev["threshold"] == t)]
        return r.iloc[0] if len(r) else None

    # ---- summary ----
    lines = []
    always_as_good = True
    for t in ["95th pct", "99th pct", "mean+3std"]:
        l, i = pick("LSTM-AE", t), pick("IsoForest", t)
        if l is None or i is None:
            continue
        lines.append(f"- **{t}:** LSTM-AE caught **{int(l['detected'])} of {int(l['episodes'])}** episodes with "
                     f"{int(l['false_bursts'])} false alarm bursts. Isolation Forest caught **{int(i['detected'])}** "
                     f"with {int(i['false_bursts'])}.")
        if not (l["detected"] >= i["detected"] and l["false_bursts"] <= i["false_bursts"]):
            always_as_good = False
    if always_as_good:
        verdict = "LSTM-AE did at least as well as Isolation Forest on both measures at every sensitivity."
    else:
        verdict = "The two models are mixed: neither is better on every measure."
    best_prec = float(ev["burst_precision"].max())
    lines.append(f"- {verdict}")
    lines.append(f"- The alarms are still noisy: even in the best case only about {best_prec:.0%} of alarm bursts "
                 "matched a labelled episode.")
    st.info("**Summary of results**\n" + "\n".join(lines))

    pretty = ev.rename(columns={
        "threshold": "alarm sensitivity",
        "episodes": "real episodes",
        "detected": "episodes caught",
        "event_recall": "share of episodes caught",
        "alarm_bursts": "alarm bursts",
        "false_bursts": "false alarm bursts",
        "burst_precision": "share of alarms that were real",
    })
    st.dataframe(pretty)
    st.caption("An episode counts as caught if at least one alarm fell inside it. An alarm burst is a run of alarm windows "
               "next to each other. A false alarm burst matches no labelled episode.")
    st.caption(f"Only {n_episodes} labelled episodes, so small differences are not statistically solid. "
               "The labels are general server-health alerts (Nagios), not only cooling faults, and detection here "
               "does not show early warning.")

st.divider()
st.caption("DC Cooling Anomaly Detection | Data: M100 (Marconi100, CINECA)")