import json
from datetime import timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = next(p for p in [HERE / "dashboard_final7.parquet",
                        ROOT / "data" / "dashboard_final7.parquet"] if p.exists())
META = json.load(open(ROOT / "models" / "final7_meta.json"))
SENSORS = META["sensors"]
alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="Cooling anomaly replay", layout="wide")


@st.cache_data
def load():
    d = pd.read_parquet(DATA)
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.tz_localize(None)
    return d.sort_values(["node", "timestamp"]).reset_index(drop=True)


@st.cache_data
def load_csv(name):
    p = ROOT / "outputs" / name
    return pd.read_csv(p) if p.exists() else None


def run_ids(flag):
    """Number each run of consecutive True values 1, 2, 3...; False stays 0."""
    start = flag & ~np.r_[False, flag[:-1]]
    return np.cumsum(start) * flag


def metrics(sub, thr):
    sub = sub[sub["label_known"]]
    tp = fp = fn = tn = episodes = caught = bursts = false_bursts = 0
    for _, g in sub.groupby("node"):
        y = g["anomaly"].to_numpy(dtype=bool)
        p = g["score"].to_numpy() > thr
        tp += int((y & p).sum()); fp += int((~y & p).sum())
        fn += int((y & ~p).sum()); tn += int((~y & ~p).sum())
        ep, br = run_ids(y), run_ids(p)
        episodes += int(ep.max())
        caught += len(np.unique(ep[(ep > 0) & p]))
        bursts += int(br.max())
        false_bursts += int(br.max()) - len(np.unique(br[(br > 0) & y]))
    total = tp + fp + fn + tn
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, total=total,
                accuracy=div(tp + tn, total), baseline=div(tn + fp, total),
                precision=prec, recall=rec, f1=div(2 * prec * rec, prec + rec),
                episodes=episodes, caught=caught, bursts=bursts, false_bursts=false_bursts)


data = load()

st.title("Data-centre cooling anomaly detector")
st.caption("Replay of recorded M100 data (a test period the model never trained on) - "
           "not live monitoring. Detector: LSTM autoencoder on 7 fan/ambient sensors, 6-hour windows.")

# ---------- controls ----------
node = st.sidebar.selectbox("Node", sorted(data["node"].unique()))
days = st.sidebar.selectbox("Chart history (days)", [3, 7, 14], index=1)
log_scale = st.sidebar.checkbox("Log scale for score", value=True)
thr_name = st.sidebar.selectbox("Alarm threshold", ["99th percentile (default)", "95th percentile"])
THR = META["thr_p99"] if thr_name.startswith("99") else META["thr_p95"]
st.sidebar.caption("Threshold = a percentile of the scores on normal validation data. "
                   "Lower threshold = more alarms (higher recall, lower precision).")

data["alarm"] = data["score"] > THR
d = data[data["node"] == node].reset_index(drop=True)

tab1, tab2, tab3 = st.tabs(["Replay", "Detector performance", "Sensor study (1 vs several sensors)"])

# ---------- tab 1: replay ----------
with tab1:
    t0, t1 = d["timestamp"].iloc[0], d["timestamp"].iloc[-1]
    alerts = d[d["anomaly"]]
    start = alerts["timestamp"].iloc[0] if len(alerts) else t0
    when = st.slider("Replay time (end of the 6-hour window being checked)",
                     min_value=t0.to_pydatetime(), max_value=t1.to_pydatetime(),
                     value=start.to_pydatetime(), step=timedelta(minutes=15),
                     format="YYYY-MM-DD HH:mm")
    i = min(int(d["timestamp"].searchsorted(pd.Timestamp(when))), len(d) - 1)
    row = d.iloc[i]
    alarm = bool(row["alarm"])

    if alarm:
        st.error(f"POSSIBLE ANOMALY - node {node}, window ending {row['timestamp']}")
    else:
        st.success(f"NORMAL - node {node}, window ending {row['timestamp']}")

    c1, c2 = st.columns(2)
    c1.metric("Anomaly score", f"{row['score']:.4f}", f"{row['score'] / THR:.1f}x the alarm threshold",
              delta_color="off")
    c2.metric("Recorded health alert in this window?",
              "Yes" if row["anomaly"] else ("No" if row["label_known"] else "Unknown"))

    lo = row["timestamp"] - timedelta(days=days)
    h = d[(d["timestamp"] >= lo) & (d["timestamp"] <= row["timestamp"])]
    ys = alt.Scale(type="log" if log_scale else "linear")
    base = alt.Chart(h).encode(x=alt.X("timestamp:T", title=None))
    line = base.mark_line().encode(y=alt.Y("score:Q", scale=ys, title="Anomaly score"))
    pts = base.transform_filter("datum.alarm").mark_point(color="red", size=30).encode(
        y=alt.Y("score:Q", scale=ys))
    rule = alt.Chart(pd.DataFrame({"thr": [THR]})).mark_rule(color="red", strokeDash=[6, 4]).encode(
        y=alt.Y("thr:Q", scale=ys))
    rec = base.transform_filter("datum.anomaly").mark_rule(color="orange", opacity=0.25, strokeWidth=3)
    st.altair_chart((rec + line + pts + rule).properties(width="container", height=320))
    st.caption("Blue line = anomaly score. Red dashed line = alarm threshold. Red dots = alarms. "
               "Orange bands = recorded Nagios health alerts (for comparison only).")

    errs = row[[f"err_{c}" for c in SENSORS]].astype(float).to_numpy()
    share = errs / errs.sum() if errs.sum() > 0 else errs
    table = pd.DataFrame({"sensor": SENSORS,
                          "latest reading": [round(float(row[c]), 2) for c in SENSORS],
                          "share of anomaly score (%)": (share * 100).round(1)})
    top = table.sort_values("share of anomaly score (%)", ascending=False).iloc[0]
    st.subheader("Which sensors drive the score?")
    st.write(f"Largest contribution: **{top['sensor']}** ({top['share of anomaly score (%)']}% of the score).")
    st.dataframe(table, hide_index=True)

# ---------- tab 2: performance of the final 7-sensor model on the replay data ----------
with tab2:
    SCOPES = {f"This node (node {node})": [node],
              "Report nodes (7, 8, 9, 10)": [7, 8, 9, 10],
              "Selection nodes (11, 13, 14, 16, 17)": [11, 13, 14, 16, 17],
              "All nodes": sorted(data["node"].unique())}
    scope = st.selectbox("Calculate for", list(SCOPES))
    m = metrics(data[data["node"].isin(SCOPES[scope])], THR)
    st.write(f"Final 7-sensor model, threshold: **{thr_name}**. "
             f"Windows with a known label: **{m['total']:,}**.")

    a, b, c, e = st.columns(4)
    a.metric("Accuracy", f"{m['accuracy']:.2%}",
             f"{(m['accuracy'] - m['baseline']) * 100:+.2f} pts vs always-normal", delta_color="off")
    b.metric("Precision", f"{m['precision']:.3f}")
    c.metric("Recall", f"{m['recall']:.3f}")
    e.metric("F1", f"{m['f1']:.3f}")
    f, g, h2 = st.columns(3)
    f.metric("Alert episodes caught", f"{m['caught']} / {m['episodes']}")
    g.metric("False alarm bursts", m["false_bursts"])
    h2.metric("Always-normal accuracy", f"{m['baseline']:.2%}")
    st.dataframe(pd.DataFrame({"": ["Alarm", "No alarm"],
                               "Recorded alert": [m["tp"], m["fn"]],
                               "No recorded alert": [m["fp"], m["tn"]]}), hide_index=True)
    st.caption("Accuracy looks high because most windows are normal; predicting 'normal' every time "
               "already scores the 'always-normal' value. Precision and recall matter more. "
               "These replay numbers use every 15-minute window, so they differ slightly from the "
               "official evaluation numbers in the Sensor study tab.")

# ---------- tab 3: separate models trained on 1 sensor, or on several sensors ----------
with tab3:
    st.markdown("Every row here is a **separate model trained only on those sensors**, "
                "alarm threshold = 99th percentile of validation scores. "
                "Nodes 7-10 were not used to choose the sensors; nodes 11-17 were.")
    grp_label = st.selectbox("Evaluate on", ["Report nodes (7, 8, 9, 10)",
                                              "Selection nodes (11, 13, 14, 16, 17)"])
    grp = "report" if grp_label.startswith("Report") else "selection"
    mode = st.radio("Model", ["One sensor", "Several sensors (added best-first)"], horizontal=True)

    def show_row(r):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{r['accuracy']:.2%}",
                  f"{(r['accuracy'] - r['always_normal_acc']) * 100:+.2f} pts vs always-normal",
                  delta_color="off")
        k2.metric("Precision", f"{r['precision']:.3f}")
        k3.metric("Recall", f"{r['recall']:.3f}")
        k4.metric("Alert episodes caught", r["events"])
        k5, k6, k7 = st.columns(3)
        k5.metric("False alarm bursts", int(r["false_bursts"]))
        k6.metric("ROC-AUC", f"{r['ROC']:.3f}")
        k7.metric("PR-AUC", f"{r['PR']:.4f}")

    if mode == "One sensor":
        singles = load_csv("single_sensor_by_group.csv")
        if singles is None:
            st.info("Add outputs/single_sensor_by_group.csv (Step 10) to see single-sensor results.")
        else:
            s = singles[singles["group"] == grp].sort_values("PR", ascending=False)
            sensor = st.selectbox("Sensor (sorted by PR-AUC)", s["sensor"].tolist())
            show_row(s[s["sensor"] == sensor].iloc[0])
            st.caption("All single-sensor models on this node group:")
            st.dataframe(s.drop(columns=["group"]), hide_index=True)
    else:
        combos = load_csv("final_sensor_study.csv")
        if combos is None:
            st.info("outputs/final_sensor_study.csv not found.")
        else:
            cg = combos[combos["group"] == grp]
            n = st.select_slider("Number of sensors", options=cg["n_sensors"].tolist(), value=7)
            r = cg[cg["n_sensors"] == n].iloc[0]
            st.write(f"Sensors used: **{r['sensors']}**")
            show_row(r)
            chart_df = cg.assign(n=cg["n_sensors"].astype(str)).set_index("n")
            col_a, col_b = st.columns(2)
            col_a.caption("Precision and recall vs number of sensors")
            col_a.bar_chart(chart_df[["precision", "recall"]])
            col_b.caption("False alarm bursts vs number of sensors")
            col_b.bar_chart(chart_df[["false_bursts"]])
            st.dataframe(cg.drop(columns=["group"]), hide_index=True)

    st.caption("Small study: about 15 alert episodes per node group and one training run per model, "
               "so small differences between neighbouring rows are within noise. "
               "Labels are Nagios health alerts, not confirmed cooling faults.")