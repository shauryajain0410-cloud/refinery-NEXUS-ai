import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Refinery Nexus AI",
    page_icon="⚙️",
    layout="wide"
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    dashboard_file = Path("dashboard_data.csv")
    shap_file = Path("risk_explanations.csv")

    if not dashboard_file.exists() or not shap_file.exists():
        return None, None

    return (
        pd.read_csv(dashboard_file),
        pd.read_csv(shap_file)
    )

dashboard_data, shap_data = load_data()

if dashboard_data is None:
    st.error(
        "Dashboard data is missing. "
        "Please run the notebook pipeline first to generate "
        "dashboard_data.csv and risk_explanations.csv."
    )
    st.stop()



st.sidebar.title("🔧 Asset Control")

selected_asset = st.sidebar.selectbox(
    "Select Asset",
    dashboard_data["asset"].unique()
)


# -----------------------------
# Header
# -----------------------------
st.title("⚙️ Refinery Nexus AI")
st.caption("Predictive Maintenance • Risk Propagation • Explainable AI")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Asset Deep-Dive",
    "Network & What-If",
    "Priority Queue"
])




with tab1:
    c1, c2, c3, c4 = st.columns(4)

        # Overview
    ...

    st.markdown("""
    <div style="margin: 10px 0 20px 0;">
        <b>Risk Level:</b>
        <span style="color:#00a65a;"> 🟢 Low (0–30)</span>
        <span style="color:#e6c300;"> 🟡 Medium (30–60)</span>
        <span style="color:#ff8c00;"> 🟠 High (60–80)</span>
        <span style="color:#e00000;"> 🔴 Critical (80–100)</span>
    </div>
    """, unsafe_allow_html=True)

    with c1:
        st.metric(
            "Assets Monitored",
            dashboard_data["asset"].nunique()
        )

    with c2:
        st.metric(
            "High-Risk Assets",
            int((dashboard_data["final_risk"] >= 75).sum())
        )

    with c3:
        st.metric(
            "Average RUL",
            f"{dashboard_data['RUL_pred'].mean():.1f}"
        )

    with c4:
        st.metric(
            "Avg Uncertainty",
            f"{dashboard_data['uncertainty'].mean():.2f}"
        )

    st.divider()

    # -----------------------------
    # Priority Alert
    # -----------------------------

    top_asset = dashboard_data.sort_values(
        "priority_rank"
    ).iloc[0]

    if "priority_score" in dashboard_data.columns:
        priority_score = float(top_asset["priority_score"])
    else:
        priority_score = None

    st.warning(
        f"⚠️ Highest Priority Asset: {top_asset['asset']} "
        f"({top_asset['asset_type']}) — "
        + (
            f"Priority Score: {priority_score:.2f} | "
            if priority_score is not None
            else ""
        )
        + f"RUL: {float(top_asset['RUL_pred']):.1f}"
    )









with tab2:
    # -----------------------------
    # Selected asset
    # -----------------------------

    asset_data = dashboard_data[
        dashboard_data["asset"] == selected_asset
    ].iloc[0]

    # -----------------------------
    # Main asset information
    # -----------------------------
    st.subheader(f"Asset: {selected_asset}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Priority Rank",
            int(asset_data["priority_rank"])
        )

    with col2:
        st.metric("Priority Score", f"{float(asset_data['priority_score']):.2f}")

    with col3:
        st.metric("RUL", 
                f"{float(asset_data['RUL_pred']):.1f}")

    with col4:
        st.metric("Uncertainty", f"{float(asset_data['uncertainty']):.2f}")

    st.divider()

    # -----------------------------
    # Risk breakdown
    # -----------------------------
    st.subheader("Risk Breakdown")
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Direct Risk", "Corrosion Risk", "Propagated Risk", "Final Risk"],
        y=[
            asset_data["direct_risk"],
            asset_data["corrosion_score"],
            asset_data["propagated_risk"],
            asset_data["final_risk"]
        ]
    ))

    fig.update_layout(
        title=f"Risk Profile — {selected_asset}",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 100]),
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Direct Risk",
            round(asset_data["direct_risk"], 2)
        )

    with c2:
        st.metric(
            "Corrosion Risk",
            round(asset_data["corrosion_score"], 2)
        )

    with c3:
        st.metric(
            "Propagated Risk",
            round(asset_data["propagated_risk"], 2)
        )

    with c4:
       st.metric("Risk", f"{float(asset_data['final_risk']):.2f}")

    # -----------------------------
    # Explainable AI
    # -----------------------------

    st.divider()
    st.header("🔍 Why Is This Asset at Risk?")


    selected_unit = int(asset_data["unit"])

    asset_explanation = shap_data[
        shap_data["unit"] == selected_unit
    ]

    if not asset_explanation.empty:

        latest_explanation = asset_explanation.iloc[-1]

        st.write(
            f"Top contributing features for **{selected_asset}**:"
        )

        feature_data = pd.DataFrame({
            "Feature": [
                latest_explanation["top_feature_1"],
                latest_explanation["top_feature_2"],
                latest_explanation["top_feature_3"]
            ]
        })

        st.dataframe(
            feature_data,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No feature explanation is available for this asset."
        )

    # -----------------------------
    # Maintenance Recommendation
    # -----------------------------

    st.divider()
    st.header("🛠️ Maintenance Recommendation")

    selected_row = dashboard_data[
        dashboard_data["asset"] == selected_asset
    ].iloc[0]

    priority_score = float(selected_row["priority_score"])
    rul = float(selected_row["RUL_pred"])
    uncertainty = float(selected_row["uncertainty"])

    if priority_score >= 80:
        recommendation = "Immediate inspection recommended."
    elif priority_score >= 60:
        recommendation = "Schedule maintenance inspection soon."
    else:
        recommendation = "Continue monitoring."

    st.info(
        f"**{selected_asset}** — {recommendation} "
        f"Priority Score: **{priority_score:.2f}**, "
        f"Predicted RUL: **{rul:.1f}**, "
        f"Uncertainty: **{uncertainty:.2f}**."
    )



with tab3:
    # -----------------------------
    # Refinery Risk Network
    # -----------------------------



    st.divider()
    st.subheader("Refinery Risk Network")

    # Same branched topology used in the notebook
    edges = [
        ("T101", "P101"),
        ("P101", "PIPE101"),
        ("P101", "PIPE102"),
        ("PIPE101", "V101"),
        ("V101", "HX101"),
        ("HX101", "T102"),
        ("PIPE102", "V102"),
        ("V102", "P102"),
        ("P102", "PIPE103"),
        ("PIPE103", "T102")
    ]

    # Node positions
    positions = {
        "T101": (0, 0),
        "P101": (1, 0),
        "PIPE101": (2, 0.7),
        "PIPE102": (2, -0.7),
        "V101": (3, 0.7),
        "V102": (3, -0.7),
        "HX101": (4, 0.7),
        "P102": (4, -0.7),
        "PIPE103": (5, -0.7),
        "T102": (6, 0)
    }

    # Risk lookup
    risk_lookup = dashboard_data.set_index("asset")["final_risk"].to_dict()

    # Edge lines
    edge_x = []
    edge_y = []

    for source, target in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]

        edge_x += [x1, x2, None]
        edge_y += [y1, y2, None]

    graph_fig = go.Figure()

    graph_fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2),
        hoverinfo="none"
    ))

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_risk = []

    for node, (x, y) in positions.items():
        node_x.append(x)
        node_y.append(y)

        risk = risk_lookup.get(node, 0)

        node_risk.append(risk)
        node_text.append(
            f"{node}<br>Final Risk: {risk:.2f}"
        )

    graph_fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(positions.keys()),
        textposition="bottom center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=35,
            color=node_risk,
            colorscale="RdYlGn_r",
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(title="Risk")
        )
    ))

    graph_fig.update_layout(
        height=550,
        title="Asset Risk Propagation Network",
        xaxis=dict(
            visible=False,
            showgrid=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False
        ),
        showlegend=False
    )

    st.plotly_chart(
        graph_fig,
        use_container_width=True
    )


    # -----------------------------
    # What-If Risk Simulation
    # -----------------------------

    st.divider()
    st.subheader("🧪 What-If Risk Simulation")

    st.write(
        "Select an asset to simulate its repair. "
        "Its direct risk is set to zero and the downstream "
        "cascading impact is recalculated."
    )

    what_if_asset = st.selectbox(
        "Select asset to hypothetically repair",
        dashboard_data["asset"].unique(),
        index=list(dashboard_data["asset"].unique()).index("P101"),
        key="what_if_asset"
    )

    # Propagation configuration
    EDGE_WEIGHT = 0.35
    DEPTH_DECAY = 0.5
    MAX_DEPTH = 5

    # Build adjacency list
    graph = {}

    for source, target in edges:
        graph.setdefault(source, []).append(target)

    # Direct-risk lookup
    direct_risk = dashboard_data.set_index(
        "asset"
    )["direct_risk"].to_dict()

    # Calculate multi-level downstream propagation
    source_risk = direct_risk.get(what_if_asset, 0)

    propagated_before = {}

    for node in dashboard_data["asset"].unique():

        if node == what_if_asset:
            continue

        # Find shortest downstream distance
        queue = [(what_if_asset, 0)]
        visited = set()
        distance = None

        while queue:
            current, d = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)

            if current == node:
                distance = d
                break

            for child in graph.get(current, []):
                queue.append((child, d + 1))

        if distance is not None and 0 < distance <= MAX_DEPTH:

            propagated_before[node] = (
                source_risk
                * EDGE_WEIGHT
                * (DEPTH_DECAY ** (distance - 1))
            )

    # After repair, source direct risk becomes zero
    propagated_after = {
        node: 0
        for node in propagated_before
    }

    # Calculate risk reduction for each downstream asset

    impact_rows = []

    for node in propagated_before:

        before = propagated_before[node]
        after = propagated_after[node]

        impact_rows.append({
            "Downstream Asset": node,
            "Cascading Risk Before": before,
            "Cascading Risk After": after,
            "Risk Reduction": before - after
        })

    impact_df = pd.DataFrame(
        impact_rows,
        columns=[
            "Downstream Asset",
            "Cascading Risk Before",
            "Cascading Risk After",
            "Risk Reduction"
        ]
    )

    # Total downstream risk reduction

    if not impact_df.empty:

        total_downstream_reduction = impact_df["Risk Reduction"].sum()

        st.metric(
            "Downstream Risk Reduction",
            f"{total_downstream_reduction:.2f}"
        )
        st.success(
            f"Repairing {what_if_asset} removes "
            f"{total_downstream_reduction:.2f} points of "
            f"cascading risk from downstream assets."
        )

        st.subheader(f"Impact of repairing {what_if_asset}")

        st.dataframe(
            impact_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        total_downstream_reduction = 0

        st.metric(
            "Downstream Risk Reduction",
            "0.00"
        )

        st.info(
            "This asset has no downstream connections within "
            "the configured propagation depth."
        )




with tab4:
    # -----------------------------
    # Priority Queue
    # -----------------------------
    st.divider()
    st.subheader("🚨 Maintenance Priority Queue")

    st.write(
        "Assets are ranked using the final risk score, "
        "urgency, and model uncertainty."
    )

    priority_display = dashboard_data[
        [
            "priority_rank",
            "asset",
            "asset_type",
            "RUL_pred",
            "direct_risk",
            "corrosion_score",
            "propagated_risk",
            "final_risk",
            "uncertainty",
            "urgency",
            "priority_score",
            "risk_level"
        ]
    ].copy()

    priority_display = priority_display.sort_values(
        "priority_rank"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Highest Priority Asset",
            priority_display.iloc[0]["asset"]
        )

    with c2:
        st.metric(
            "Priority Score",
            f"{priority_display.iloc[0]['priority_score']:.2f}"
        )

    with c3:
        st.metric(
            "Predicted RUL",
            f"{priority_display.iloc[0]['RUL_pred']:.1f}"
        )
    def color_risk_level(value):
        if value == "CRITICAL":
            return "background-color: #ff4b4b; color: white;"
        elif value == "HIGH":
            return "background-color: #ff8c00; color: white;"
        elif value == "MEDIUM":
            return "background-color: #ffd43b; color: black;"
        elif value == "LOW":
            return "background-color: #51cf66; color: black;"
        return ""

    priority_styled = priority_display.style.map(
        color_risk_level,
        subset=["risk_level"]
    )

    st.dataframe(
        priority_display,
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Final Risk",
            f"{asset_data['final_risk']:.2f}"
        )

    with c2:
        st.metric(
            "Predicted RUL",
            f"{asset_data['RUL_pred']:.1f}"
        )

    with c3:
        st.metric(
            "Priority Rank",
            int(asset_data["priority_rank"])
        )

    with c4:
        st.metric(
            "Uncertainty",
            f"{asset_data['uncertainty']:.2f}"
        )
    with c5:
        st.metric(
            "Urgency",
            str(asset_data["urgency"])
        )





        # -----------------------------
        # Model Uncertainty
        # -----------------------------
    st.divider()
    st.subheader("🎯 Model Confidence / Uncertainty")

    st.write(
        "Uncertainty is estimated from disagreement between "
        "the Random Forest and XGBoost RUL predictions."
    )

    uncertainty_display = dashboard_data[
        [
            "asset",
            "RUL_pred",
            "uncertainty",
            "final_risk"
        ]
    ].copy()

    uncertainty_display = uncertainty_display.sort_values(
        "uncertainty",
        ascending=False
    )

    highest_uncertainty = uncertainty_display.iloc[0]

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Highest Uncertainty Asset",
            highest_uncertainty["asset"]
        )

    with c2:
        st.metric(
            "Uncertainty",
            f"{highest_uncertainty['uncertainty']:.2f}"
        )

    st.dataframe(
        uncertainty_display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Higher uncertainty means greater disagreement between the "
        "two models and therefore lower confidence in the point estimate."
    )
