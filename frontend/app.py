"""TALUS — industrial-style Streamlit control room.
The frontend is an API client: all predictions and mine-state data come from FastAPI.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="TALUS | Slope Risk Intelligence", page_icon="⛰", layout="wide", initial_sidebar_state="expanded")
API_URL = os.getenv("TALUS_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.markdown("""
<style>
:root { --panel:#111923; --panel2:#172230; --line:#2b3a4c; --muted:#91a2b5; --text:#edf3f8; }
.stApp { background: radial-gradient(circle at 20% 0%, #182536 0, #0a0f15 45%, #080c11 100%); color:var(--text); }
section[data-testid="stSidebar"] { background:#0b1118; border-right:1px solid var(--line); }
.block-container { max-width: 1500px; padding-top: 1.4rem; }
.hero { padding:24px 28px; border:1px solid var(--line); border-radius:18px; background:linear-gradient(135deg,#142333,#0e151e); margin-bottom:18px; }
.hero h1 { margin:0; font-size:2.1rem; letter-spacing:-.03em; }
.hero p { color:var(--muted); margin:.4rem 0 0; }
.kpi { background:rgba(17,25,35,.9); border:1px solid var(--line); border-radius:14px; padding:17px; min-height:105px; }
.kpi-label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; }
.kpi-value { font-size:1.75rem; font-weight:750; margin-top:8px; }
.sector-card { background:rgba(17,25,35,.92); border:1px solid var(--line); border-radius:15px; padding:16px; min-height:190px; }
.sector-id { font-weight:800; font-size:1.05rem; }
.small { color:var(--muted); font-size:.82rem; }
.risk-low { color:#5ee28a; } .risk-moderate { color:#ffd166; } .risk-high { color:#ff9b54; } .risk-critical { color:#ff6b6b; }
.status { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:3px 10px; font-size:.72rem; font-weight:700; }
.insight { background:#101a24; border-left:4px solid #6ea8fe; border-radius:8px; padding:12px 15px; margin:8px 0; }
.warning { background:#251b12; border-left:4px solid #ff9b54; border-radius:8px; padding:12px 15px; }
@keyframes pulse { 0%,100% { box-shadow:0 0 0 rgba(240,68,85,0); } 50% { box-shadow:0 0 18px rgba(240,68,85,.45); } }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=15, show_spinner=False)
def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=6)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def risk_class(label: str) -> str:
    return f"risk-{str(label).lower()}"

def metric(label, value, note=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="small">{note}</div></div>', unsafe_allow_html=True)

def plot_layout(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=55,b=20), legend_title_text="")
    return fig

# Connection gate
try:
    health = api_get("/health")
    connected = health.get("status") == "healthy"
except Exception as exc:
    connected = False
    health = {"status":"offline"}

with st.sidebar:
    st.markdown("# ⛰ TALUS")
    st.caption("PHYSICS-INFORMED SLOPE RISK INTELLIGENCE")
    st.divider()
    nav = st.radio("CONTROL ROOM", ["Executive Summary", "Mine Overview", "3D Mine Model", "Analyze Sector", "Risk Timeline", "Data Intelligence", "Materials Gallery", "Model Performance", "ML Rigor", "Scenario Simulator", "Virtual Scenarios", "TALUS Voice", "Alert Console", "Advanced Analytics", "System Health", "System & Viva"], label_visibility="visible")
    st.divider()
    if connected:
        st.success("API ONLINE")
        st.caption(f"FastAPI: {API_URL}")
    else:
        st.error("API OFFLINE")
        st.caption(f"Expected: {API_URL}")
        st.code("python run_backend.py", language="powershell")
    st.divider()
    st.caption("TALUS v3.0 • Phase II + III")

if not connected:
    st.markdown('<div class="hero"><h1>TALUS Control Room</h1><p>Backend connection required for live mine intelligence.</p></div>', unsafe_allow_html=True)
    st.warning("The frontend intentionally does not silently execute predictions locally. Start FastAPI first, or set TALUS_API_URL to your deployed API.")
    st.stop()

try:
    overview = api_get("/mine/overview").get("sectors", [])
except Exception as exc:
    st.error(f"Could not load mine overview: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# Mine Overview
# -----------------------------------------------------------------------------
if nav == "Mine Overview":
    st.markdown('<div class="hero"><h1>Mine Risk Command Center</h1><p>Live sector state, physics stability and ML risk intelligence from the TALUS API.</p></div>', unsafe_allow_html=True)
    labels = [s.get("ml_risk_label","Low") for s in overview]
    probs = [float(s.get("ml_risk_probability",0)) for s in overview]
    fos = [float(s.get("factor_of_safety",0)) for s in overview]
    high = sum(x in {"High","Critical"} for x in labels)
    critical = sum(x == "Critical" for x in labels)
    highest = overview[int(np.argmax(probs))] if overview else {}
    c = st.columns(5)
    with c[0]: metric("Monitored sectors", len(overview), "Data-driven sector registry")
    with c[1]: metric("High / critical", high, f"{critical} critical")
    with c[2]: metric("Highest-risk sector", highest.get("sector_id","—"), highest.get("ml_risk_label","—"))
    with c[3]: metric("Mean FoS", f"{np.mean(fos):.2f}" if fos else "—", "Physics stability indicator")
    with c[4]: metric("API state", "HEALTHY", "FastAPI + SQLite + model")

    st.subheader("Sector Watchboard")
    for start in range(0, len(overview), 4):
        cols = st.columns(4)
        for col, s in zip(cols, overview[start:start+4]):
            lbl=s.get("ml_risk_label","Low"); prob=float(s.get("ml_risk_probability",0))*100; f=float(s.get("factor_of_safety",0))
            with col:
                st.markdown(f'''<div class="sector-card"><div class="sector-id">{s.get("sector_id")} · {s.get("sector_name","Monitoring Sector")}</div><div class="small">{s.get("pit_zone","Telemetry Grid")}</div><hr><div class="{risk_class(lbl)}"><b>{lbl.upper()}</b> · {prob:.1f}% model probability</div><p>FoS <b>{f:.2f}</b> &nbsp;|&nbsp; Deformation <b>{float(s.get("deformation_rate_mm_day",0)):.2f} mm/day</b></p><p>Pore pressure <b>{float(s.get("pore_pressure_kpa",0)):.1f} kPa</b></p></div>''', unsafe_allow_html=True)

    left,right=st.columns(2)
    with left:
        counts=pd.Series(labels).value_counts().reindex(["Low","Moderate","High","Critical"],fill_value=0)
        st.plotly_chart(plot_layout(px.bar(x=counts.index,y=counts.values,title="Current risk distribution",labels={"x":"Risk class","y":"Sectors"})),use_container_width=True)
    with right:
        df=pd.DataFrame(overview)
        if not df.empty:
            fig=px.scatter(df,x="factor_of_safety",y="deformation_rate_mm_day",size="pore_pressure_kpa",hover_name="sector_id",title="Physics vs telemetry stress")
            st.plotly_chart(plot_layout(fig),use_container_width=True)

    st.subheader("Decision-support signals")
    for s in overview[:5]:
        lbl=s.get("ml_risk_label","Low"); f=float(s.get("factor_of_safety",0)); d=float(s.get("deformation_rate_mm_day",0)); pp=float(s.get("pore_pressure_kpa",0))
        if lbl in {"High","Critical"} or f < 1.3:
            st.markdown(f'<div class="warning"><b>{s.get("sector_id")}</b>: {lbl} state. Review FoS ({f:.2f}), deformation ({d:.2f} mm/day) and pore pressure ({pp:.1f} kPa) before operational decisions.</div>',unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Analyze Sector
# -----------------------------------------------------------------------------
elif nav == "Executive Summary":
    st.markdown('<div class="hero"><h1>Executive Mine Briefing</h1><p>One-screen synthetic-data overview for a quick operational discussion.</p></div>', unsafe_allow_html=True)
    labels = [s.get("ml_risk_label", "Low") for s in overview]
    ranked = sorted(overview, key=lambda s: (s.get("ml_risk_label") != "Critical", -float(s.get("ml_risk_probability", 0))))
    c = st.columns(4)
    c[0].metric("Sectors tracked", len(overview))
    c[1].metric("High / Critical", sum(x in ("High", "Critical") for x in labels))
    c[2].metric("Critical", sum(x == "Critical" for x in labels))
    c[3].metric("Mean FoS", f"{np.mean([float(s.get('factor_of_safety', 0)) for s in overview]):.2f}" if overview else "—")
    st.subheader("Priority watchlist")
    st.dataframe(pd.DataFrame([{"Sector":s.get("sector_id"),"Risk":s.get("ml_risk_label"),"FoS":s.get("factor_of_safety"),"Probability":s.get("ml_risk_probability"),"Deformation mm/day":s.get("deformation_rate_mm_day"),"Pore pressure kPa":s.get("pore_pressure_kpa")} for s in ranked[:3]]), use_container_width=True, hide_index=True)
    st.markdown("**Recommended next steps:** verify high-priority readings against approved instruments; follow the site's TARP; review water, blast and deformation trends; ask the responsible geotechnical engineer to authorize any action.")
    st.caption("Synthetic-data prototype briefing. Not a real mine-site alert or a certified safety determination.")

elif nav == "3D Mine Model":
    st.markdown('<div class="hero"><h1>3D Open-Pit Digital Terrain</h1><p>Procedurally generated terraced geometry, colored by the current prototype risk snapshot.</p></div>', unsafe_allow_html=True)
    sector_ids = [s.get("sector_id", f"S{i+1}") for i,s in enumerate(overview)]
    risk_num = {"Low":0,"Moderate":1,"High":2,"Critical":3}
    n=100
    gx=np.linspace(-1,1,n); gy=np.linspace(-1,1,n); xx,yy=np.meshgrid(gx,gy)
    rr=np.sqrt(xx**2+yy**2); angle=(np.arctan2(yy,xx)+np.pi)/(2*np.pi)
    z=1.0 + np.floor(np.clip(rr,0,1)*7)/7*0.85
    z=np.where(rr<0.15, 0.25, z)
    sec_idx=np.minimum(len(overview)-1,(angle*max(1,len(overview))).astype(int)) if overview else np.zeros_like(rr,dtype=int)
    classes=np.array([risk_num.get(s.get("ml_risk_label","Low"),0) for s in overview]) if overview else np.zeros(1)
    surfacecolor=classes[sec_idx] if overview else np.zeros_like(z)
    fig=go.Figure(data=[go.Surface(x=xx*500,y=yy*500,z=z*180,surfacecolor=surfacecolor,colorscale=[[0,"#35c979"],[0.33,"#e9c94b"],[0.66,"#ff914d"],[1,"#f04455"]],cmin=0,cmax=3,showscale=True,colorbar=dict(title="Risk",tickvals=[0,1,2,3],ticktext=["Low","Moderate","High","Critical"]),lighting=dict(ambient=0.55,diffuse=0.8,specular=0.25,roughness=0.8),hovertemplate="X %{x:.0f} m<br>Y %{y:.0f} m<br>Elevation %{z:.0f} m<extra>Procedural terrain</extra>")])
    for i,s in enumerate(overview):
        theta=2*np.pi*i/max(1,len(overview)); r=0.68
        fig.add_trace(go.Scatter3d(x=[500*r*np.cos(theta)],y=[500*r*np.sin(theta)],z=[150],mode="markers+text",text=[s.get("sector_id")],textposition="top center",marker=dict(size=7,color=[{"Low":"#35c979","Moderate":"#e9c94b","High":"#ff914d","Critical":"#f04455"}.get(s.get("ml_risk_label"),"#35c979")],line=dict(color="white",width=1)),name=f"{s.get('sector_id')} · {s.get('ml_risk_label')}",hovertemplate=f"{s.get('sector_id')}<br>Risk: {s.get('ml_risk_label')}<br>FoS: {float(s.get('factor_of_safety',0)):.2f}<extra></extra>"))
    fig.update_layout(height=650,scene=dict(xaxis_title="East (m)",yaxis_title="North (m)",zaxis_title="Elevation (m)",aspectratio=dict(x=1,y=1,z=.55),bgcolor="#0b1118"),margin=dict(l=0,r=0,t=35,b=0),title="TALUS Terraced Pit · Procedural geometry")
    st.plotly_chart(plot_layout(fig), use_container_width=True)
    st.caption("The terrain is procedurally generated for visualization. It is not a surveyed mine model. Sector markers use current API snapshot labels.")
    st.subheader("3D feature-space view")
    feature_df=pd.read_csv(ROOT/"data/processed/cleaned_features.csv")
    from sklearn.decomposition import PCA
    feature_cols=[c for c in ["factor_of_safety","deformation_rate_mm_day","pore_pressure_kpa","slope_angle_deg","rainfall_mm","displacement_mm"] if c in feature_df.columns]
    matrix=feature_df[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(feature_df[feature_cols].median())
    coords=PCA(n_components=3,random_state=42).fit_transform((matrix-matrix.mean())/(matrix.std().replace(0,1)))
    feature_df["PC1"],feature_df["PC2"],feature_df["PC3"]=coords[:,0],coords[:,1],coords[:,2]
    feat_fig=px.scatter_3d(feature_df.sample(min(700,len(feature_df)),random_state=42),x="PC1",y="PC2",z="PC3",color="risk_label",hover_data=["sector_id","factor_of_safety"],title="PCA projection of synthetic telemetry")
    st.plotly_chart(plot_layout(feat_fig),use_container_width=True)
    st.caption("PCA compresses selected numeric telemetry into three dimensions for visual inspection; axes are components, not physical units.")
    selected=st.selectbox("Inspect sector cross-section",sector_ids,key="cross_section_sector")
    s=next((x for x in overview if x.get("sector_id")==selected), overview[0] if overview else {})
    height=float(s.get("slope_height_m",60)); angle=float(s.get("slope_angle_deg",40)); run=max(1,height/np.tan(np.radians(max(5,min(85,angle)))))
    cross=go.Figure()
    cross.add_trace(go.Scatter(x=[0,run,run,0,0],y=[0,0,height,height,0],fill="toself",name="Slope section",line=dict(color="#d6a66b"),fillcolor="rgba(173,117,66,.35)"))
    fos_value=float(s.get("factor_of_safety",1.5)); slip_color="#f04455" if fos_value<1 else ("#ff914d" if fos_value<1.3 else "#e9c94b")
    cross.add_trace(go.Scatter(x=np.linspace(run*.15,run*.95,80),y=np.linspace(height*.82,height*.08,80)-np.sin(np.linspace(0,np.pi,80))*height*.1,mode="lines",name="Illustrative slip surface",line=dict(color=slip_color,width=4)))
    gw=float(s.get("groundwater_level",10)); cross.add_hline(y=min(height,max(0,gw)),line_dash="dash",line_color="#54b7ff",annotation_text="Groundwater reference")
    if int(s.get("recent_blasting",0))==1:
        cross.add_trace(go.Scatter(x=[run*.82],y=[height*.88],mode="markers+text",text=["Blast"],textposition="top center",marker=dict(symbol="star",size=13,color="#ff914d"),name="Recent blast marker"))
    cross.add_trace(go.Scatter(x=[run*.92],y=[height*.98],mode="markers+text",text=[f"Equipment load {float(s.get('equipment_load',0)):.0f} kPa"],textposition="top center",marker=dict(symbol="square",size=10,color="#e9c94b"),name="Crest surcharge marker"))
    cross.update_layout(title=f"{selected} · FoS {fos_value:.2f} · schematic slope cross-section",xaxis_title="Horizontal distance (m)",yaxis_title="Elevation (m)",height=380)
    st.plotly_chart(plot_layout(cross), use_container_width=True)
    st.caption("Cross-section uses sector inputs where available; slip surface is schematic, not a calibrated limit-equilibrium solution.")

elif nav == "Analyze Sector":

    st.markdown('<div class="hero"><h1>Sector Assessment</h1><p>Submit a telemetry/geotechnical snapshot to the FastAPI prediction service.</p></div>', unsafe_allow_html=True)
    sector_ids=[s["sector_id"] for s in overview]
    selected=st.selectbox("Sector", sector_ids)
    base=next((s for s in overview if s["sector_id"]==selected), overview[0] if overview else {})
    with st.form("assessment"):
        a,b,c,d=st.columns(4)
        with a:
            h=st.number_input("Slope height (m)",1.0,300.0,float(base.get("slope_height_m",60)))
            ang=st.number_input("Slope angle (°)",5.0,85.0,float(base.get("slope_angle_deg",40)))
            coh=st.number_input("Cohesion (kPa)",0.0,500.0,float(base.get("cohesion_kpa",35)))
            phi=st.number_input("Friction angle (°)",1.0,75.0,float(base.get("friction_angle_deg",30)))
        with b:
            gamma=st.number_input("Unit weight (kN/m³)",10.0,40.0,float(base.get("unit_weight_kn_m3",24)))
            rain=st.number_input("Rainfall (mm)",0.0,500.0,float(base.get("rainfall_mm",25)))
            gw=st.number_input("Groundwater level",0.0,100.0,float(base.get("groundwater_level",10)))
            pp=st.number_input("Pore pressure (kPa)",0.0,1000.0,float(base.get("pore_pressure_kpa",50)))
        with c:
            dr=st.number_input("Deformation (mm/day)",0.0,200.0,float(base.get("deformation_rate_mm_day",3.5)))
            disp=st.number_input("Displacement (mm)",0.0,2000.0,float(base.get("displacement_mm",25)))
            blasting=st.toggle("Recent blasting",bool(base.get("recent_blasting",0)))
            blast=st.number_input("Blast intensity",0.0,10.0,float(base.get("blast_intensity",0)))
        with d:
            load=st.number_input("Equipment load (kPa)",0.0,300.0,float(base.get("equipment_load",0)))
            scenario=st.text_input("Scenario name","Live sector assessment")
            st.caption("All ranges are validated by the backend schema.")
        submitted=st.form_submit_button("RUN PHYSICS + ML ASSESSMENT",use_container_width=True)
    if submitted:
        payload={"sector_id":selected,"slope_height_m":h,"slope_angle_deg":ang,"cohesion_kpa":coh,"friction_angle_deg":phi,"unit_weight_kn_m3":gamma,"rainfall_mm":rain,"groundwater_level":gw,"pore_pressure_kpa":pp,"deformation_rate_mm_day":dr,"displacement_mm":disp,"recent_blasting":int(blasting),"blast_intensity":blast,"equipment_load":load,"scenario_name":scenario}
        try:
            with st.spinner("Running physics engine, ML inference and XAI..."):
                result=api_post("/predict",payload)
            st.success("Assessment completed and persisted to the TALUS history store.")
            r1,r2,r3,r4=st.columns(4)
            with r1: metric("Risk class",result["ml_risk_label"],f'{result["ml_risk_probability"]*100:.1f}% model probability')
            with r2: metric("Factor of Safety",f'{result["factor_of_safety"]:.2f}',result["physics_category"])
            with r3: metric("Decision record",f'#{result["saved_db_id"]}',"SQLite persistence")
            with r4: metric("Assessment",result["scenario_name"],"Timestamped")
            st.subheader("All class probabilities")
            class_probs = result.get("class_probabilities", {})
            if class_probs:
                st.plotly_chart(plot_layout(px.bar(x=list(class_probs.keys()), y=[100*float(v) for v in class_probs.values()], labels={"x":"Risk class","y":"Probability (%)"}, title="Probability across all four risk classes")), use_container_width=True)
            st.subheader("Engineering decision support")
            st.markdown(f'<div class="insight">{result["decision_recommendation"]}</div>',unsafe_allow_html=True)
            st.subheader("Why did the model flag this?")
            contrib=pd.DataFrame(result.get("top_contributors",[]))
            if not contrib.empty:
                st.plotly_chart(plot_layout(px.bar(contrib.sort_values("impact_score"),x="impact_score",y="feature",orientation="h",title="Local feature contribution")),use_container_width=True)
                st.dataframe(contrib,use_container_width=True,hide_index=True)
            st.caption(result["disclaimer"])
            try:
                report_response=requests.get(f"{API_URL}/report/{selected}.pdf",timeout=20)
                report_response.raise_for_status()
                st.download_button("DOWNLOAD SECTOR PDF REPORT",data=report_response.content,file_name=f"TALUS_{selected}_report.pdf",mime="application/pdf")
            except Exception as report_exc:
                st.caption(f"PDF report not available: {report_exc}")
            contributors_for_scenario=result.get("top_contributors",[])
            contributor_names=" ".join(str(x.get("feature","")) for x in contributors_for_scenario[:3]).lower()
            suggested="blasting" if "blast" in contributor_names else ("groundwater" if "pore_pressure" in contributor_names else ("deformation" if "deformation" in contributor_names else "rainfall"))
            st.info(f"Suggested conceptual scenario to review: {suggested.title()} — open Virtual Scenarios for a checklist.")
        except requests.HTTPError as exc:
            st.error(f"Backend rejected the assessment: {exc.response.text if exc.response is not None else exc}")

# -----------------------------------------------------------------------------
# Timeline
# -----------------------------------------------------------------------------
elif nav == "Risk Timeline":
    st.markdown('<div class="hero"><h1>Risk Trajectory</h1><p>Historical observations stored by sector. Use the trend to support change-point discussions in the viva.</p></div>',unsafe_allow_html=True)
    sid=st.selectbox("Sector",[s["sector_id"] for s in overview],key="timeline_sector")
    hist=api_get(f"/sector/{sid}/history?limit=500").get("history",[])
    df=pd.DataFrame(hist)
    if df.empty: st.info("No history available.")
    else:
        df["timestamp"]=pd.to_datetime(df["timestamp"])
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(plot_layout(px.line(df,x="timestamp",y="factor_of_safety",title="Factor of Safety trajectory",markers=True)),use_container_width=True)
        with c2:
            st.plotly_chart(plot_layout(px.line(df,x="timestamp",y="deformation_rate_mm_day",title="Deformation trajectory",markers=True)),use_container_width=True)
        c3,c4=st.columns(2)
        with c3: st.plotly_chart(plot_layout(px.line(df,x="timestamp",y="pore_pressure_kpa",title="Pore pressure trajectory")),use_container_width=True)
        with c4: st.plotly_chart(plot_layout(px.line(df,x="timestamp",y="ml_risk_probability",title="Risk probability trajectory")),use_container_width=True)
        st.dataframe(df.tail(25),use_container_width=True,hide_index=True)
        st.subheader("Displacement time-lapse")
        ordered=df.sort_values("timestamp").reset_index(drop=True)
        if len(ordered)>2:
            frames=[go.Frame(data=[go.Scatter(x=ordered["timestamp"].iloc[:i+1],y=ordered["displacement_mm"].iloc[:i+1],mode="lines+markers",line=dict(color="#54b7ff",width=3))],name=str(i)) for i in range(1,len(ordered),max(1,len(ordered)//24))]
            anim=go.Figure(data=[go.Scatter(x=ordered["timestamp"].iloc[:1],y=ordered["displacement_mm"].iloc[:1],mode="lines+markers")],frames=frames)
            anim.update_layout(title=f"{sid} · displacement build-up (record playback)",xaxis_title="Recorded timestamp",yaxis_title="Displacement (mm)",updatemenus=[dict(type="buttons",buttons=[dict(label="Play",method="animate",args=[None,{"frame":{"duration":180,"redraw":True},"fromcurrent":True}]),dict(label="Pause",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0}}])])])
            st.plotly_chart(plot_layout(anim),use_container_width=True)

# -----------------------------------------------------------------------------
# Data Intelligence / EDA
# -----------------------------------------------------------------------------
elif nav == "Data Intelligence":
    st.markdown('<div class="hero"><h1>Data Intelligence Lab</h1><p>Reproducible preprocessing and EDA across telemetry, geotechnical samples and image-derived features.</p></div>',unsafe_allow_html=True)
    slope=pd.read_csv(ROOT/"data/processed/cleaned_features.csv")
    rock=pd.read_csv(ROOT/"data/processed/rock_soil_clean.csv")
    img=pd.read_csv(ROOT/"data/processed/image_features.csv")
    t1,t2,t3=st.tabs(["Slope telemetry","Rock / soil","Image features"])
    with t1:
        c=st.columns(4)
        c[0].metric("Records",len(slope)); c[1].metric("Sectors",slope.sector_id.nunique()); c[2].metric("Features",14); c[3].metric("Missing cells",int(slope.isna().sum().sum()))
        l,r=st.columns(2)
        with l: st.plotly_chart(plot_layout(px.histogram(slope,x="factor_of_safety",color="risk_label",title="Factor of Safety distribution")),use_container_width=True)
        with r: st.plotly_chart(plot_layout(px.box(slope,x="risk_label",y="deformation_rate_mm_day",title="Deformation by risk class")),use_container_width=True)
        corr=slope.select_dtypes("number").corr(numeric_only=True)[["factor_of_safety"]].sort_values("factor_of_safety")
        st.plotly_chart(plot_layout(px.bar(corr.reset_index(),x="factor_of_safety",y="index",orientation="h",title="Numeric correlation with FoS")),use_container_width=True)
        st.dataframe(slope.head(30),use_container_width=True,hide_index=True)
    with t2:
        c=st.columns(4); c[0].metric("Samples",len(rock)); c[1].metric("Materials",rock.material.nunique()); c[2].metric("Avg GSI",f'{rock.gsi.mean():.1f}'); c[3].metric("Avg RQD",f'{rock.rqd_pct.mean():.1f}%')
        l,r=st.columns(2)
        with l: st.plotly_chart(plot_layout(px.scatter(rock,x="cohesion_kpa",y="friction_angle_deg",color="material_type",title="Cohesion vs friction angle")),use_container_width=True)
        with r: st.plotly_chart(plot_layout(px.box(rock,x="weathering_grade",y="compressive_strength_mpa",title="Strength by weathering grade")),use_container_width=True)
        st.dataframe(rock.head(30),use_container_width=True,hide_index=True)
    with t3:
        c=st.columns(4); c[0].metric("Images",len(img)); c[1].metric("Materials",img.material.nunique()); c[2].metric("Avg entropy",f'{img.entropy.mean():.2f}'); c[3].metric("Avg edge density",f'{img.edge_density.mean():.3f}')
        l,r=st.columns(2)
        with l: st.plotly_chart(plot_layout(px.scatter(img,x="brightness",y="contrast",color="material",title="Image brightness vs contrast")),use_container_width=True)
        with r: st.plotly_chart(plot_layout(px.box(img,x="material",y="glcm_homogeneity",title="Texture homogeneity by material")),use_container_width=True)
        st.dataframe(img.head(30),use_container_width=True,hide_index=True)
    st.subheader("Pipeline evidence")
    st.caption("Processed artifacts: cleaned_features.csv · rock_soil_clean.csv · image_features.csv · slope_cleaning_audit_log.csv · preprocessing_report.json")

# -----------------------------------------------------------------------------
# Model Performance
# -----------------------------------------------------------------------------
elif nav == "Materials Gallery":
    st.markdown('<div class="hero"><h1>Rock & Soil Intelligence</h1><p>Local image gallery and offline texture-feature classifier.</p></div>', unsafe_allow_html=True)
    img_df=pd.read_csv(ROOT/"data/processed/image_features.csv")
    material=st.selectbox("Material group", sorted(img_df["material"].dropna().unique()))
    candidates=img_df[img_df["material"]==material].head(8)
    cols=st.columns(4)
    for i,(_,row) in enumerate(candidates.iterrows()):
        rel=str(row.get("filepath","")); path=ROOT/"data/raw"/rel
        if not path.exists():
            matches=list((ROOT/"data/raw/images").rglob(Path(rel).name))
            path=matches[0] if matches else None
        with cols[i%4]:
            if path and path.exists(): st.image(str(path), caption=f"{material} · {row.get('moisture_state','unknown')}", use_container_width=True)
            st.caption(f"Entropy {float(row.get('entropy',0)):.2f} · Contrast {float(row.get('contrast',0)):.3f}")
    st.divider(); st.subheader("Classify an uploaded photo")
    upload=st.file_uploader("Choose a rock/soil image",type=["jpg","jpeg","png","webp","bmp"])
    if upload and st.button("CLASSIFY MATERIAL",type="primary"):
        try:
            resp=requests.post(f"{API_URL}/image/classify",files={"file":(upload.name,upload.getvalue(),upload.type or "image/jpeg")},timeout=60); resp.raise_for_status(); result=resp.json()
            st.success(f"Predicted material: {result['predicted_material']} · confidence {result['confidence']:.1%}")
            st.json(result["illustrative_strength_ranges"])
            st.plotly_chart(plot_layout(px.bar(x=list(result["class_probabilities"].keys()),y=list(result["class_probabilities"].values()),labels={"x":"Material","y":"Model probability"},title="Material class probabilities")),use_container_width=True)
            st.caption(result["disclaimer"])
        except Exception as exc: st.error(f"Could not classify image: {exc}")
    st.info("The classifier learns from the supplied image-feature table. The output is a prototype visual classification, not a geological determination. Strength ranges are illustrative only.")

elif nav == "Model Performance":
    st.markdown('<div class="hero"><h1>Model Performance & Explainability</h1><p>Model selection is tied to the persisted evaluation artifact; no dashboard label is manually invented.</p></div>',unsafe_allow_html=True)
    info=api_get("/model-info")
    comp=pd.read_csv(ROOT/"results/model_comparison.csv")
    selected=info.get("selected_model","Unknown")
    p=info.get("performance",{})
    c=st.columns(5)
    c[0].metric("Selected model",selected); c[1].metric("F1",f'{p.get("f1_score",0):.3f}'); c[2].metric("ROC-AUC",f'{p.get("roc_auc",0):.3f}'); c[3].metric("Accuracy",f'{p.get("accuracy",0):.3f}'); c[4].metric("Test records",info.get("test_size","—"))
    st.plotly_chart(plot_layout(px.bar(comp.sort_values("f1_score"),x="f1_score",y="model_name",orientation="h",title="Candidate model comparison — 5-fold CV macro-F1")),use_container_width=True)
    st.dataframe(comp,use_container_width=True,hide_index=True)
    st.subheader("Interpretation")
    st.markdown(f'<div class="insight"><b>Why {selected}?</b> The persisted training metadata says the selection metric is <b>{info.get("selection_metric","macro-F1")}</b>. Selection uses mean stratified 5-fold macro-F1. The serving model receives 14 physics-aware features, including Factor of Safety.</div>',unsafe_allow_html=True)
    st.warning("The reported score is an offline evaluation on the supplied dataset. It is not evidence of operational mine-site performance.")

# -----------------------------------------------------------------------------
# Scenario Simulator
# -----------------------------------------------------------------------------
elif nav == "ML Rigor":
    st.markdown('<div class="hero"><h1>ML Rigor & Validation</h1><p>Class imbalance, factor-of-safety ablation, cross-validation and calibration evidence.</p></div>', unsafe_allow_html=True)
    summary_path=ROOT/"results/ml_rigor_summary.json"
    if summary_path.exists():
        rigor=__import__("json").loads(summary_path.read_text(encoding="utf-8"))
        st.warning("Factor of Safety is a physics-informed input that is also closely tied to the synthetic label-generation process. The ablation quantifies this dependence; it does not by itself prove or disprove leakage.")
        st.json({k:v for k,v in rigor.items() if k not in {"per_class_report"}})
        if "per_class_report" in rigor: st.dataframe(pd.DataFrame(rigor["per_class_report"]).T,use_container_width=True)
        for img in ["ml_rigor_cv.png","ml_rigor_ablation.png","ml_rigor_calibration.png","ml_rigor_importance.png","ml_rigor_shap_bar.png","ml_rigor_shap_beeswarm.png","ml_rigor_shap_waterfall.png"]:
            p=ROOT/"results"/img
            if p.exists(): st.image(str(p),caption=img.replace("_"," ").replace(".png",""),use_container_width=True)
    else: st.info("Run `python scripts/train_ml_rigor.py` to generate reproducible ML-rigor artifacts.")
    st.markdown("**Why accuracy can mislead:** the supplied target is imbalanced (Critical 554, High 405, Moderate 211, Low 30). A model can score well by favoring majority classes while missing the rare Low class. Macro-F1 gives each class equal weight; weighted-F1 reflects class frequency.")

elif nav == "Virtual Scenarios":
    st.markdown('<div class="hero"><h1>Virtual Hazard Scenarios</h1><p>Offline animated conceptual demonstrations with response checklists.</p></div>', unsafe_allow_html=True)
    scenario_options={"Heavy rainfall":"rainfall","Blasting vibration":"blasting","Accelerating deformation":"deformation","Equipment surcharge":"equipment","Groundwater rise":"groundwater","Stable baseline":"baseline"}
    choice=st.selectbox("Choose a scenario",list(scenario_options))
    scenario=api_get("/scenario/"+scenario_options[choice])
    frames=24; t=np.arange(frames); yv=np.maximum(0,1-t/(frames*1.15)) if choice!="Stable baseline" else np.ones(frames)*.92
    fig=go.Figure(data=[go.Scatter(x=np.linspace(0,10,80),y=np.sin(np.linspace(0,6,80))*0.05+yv[0],mode="lines",line=dict(color="#ff5b67",width=4),name="Conceptual slope profile")],frames=[go.Frame(data=[go.Scatter(x=np.linspace(0,10,80),y=np.sin(np.linspace(0,6,80))*0.05+yv[i],mode="lines",line=dict(color="#ff5b67",width=4))],name=str(i)) for i in range(frames)])
    fig.update_layout(title=f"{scenario['title']} · conceptual animation",xaxis_title="Slope profile distance (arbitrary units)",yaxis_title="Relative displacement (illustrative)",height=380,updatemenus=[dict(type="buttons",buttons=[dict(label="Play",method="animate",args=[None,{"frame":{"duration":120,"redraw":True},"fromcurrent":True}]),dict(label="Pause",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0}}])])])
    st.plotly_chart(plot_layout(fig),use_container_width=True)
    st.markdown("### Precautions & response")
    for item in scenario["precautions"]: st.markdown(f"- {item}")
    st.caption(scenario["mechanism"]+" This animation is conceptual; it is not a calibrated physics simulation or an evacuation plan.")

elif nav == "TALUS Voice":
    st.markdown('<div class="hero"><h1>TALUS Voice</h1><p>Rule-based voice assistant using browser speech recognition and speech synthesis when supported.</p></div>', unsafe_allow_html=True)
    voice_html = """<div style='font-family:Arial;color:#eaf2f8;background:#101923;padding:18px;border-radius:14px;border:1px solid #2b3a4c'><div style='display:flex;gap:10px;flex-wrap:wrap'><button id='enable'>Enable voice</button><button id='listen'>Push to talk</button><button id='wake'>Hey TALUS (listen)</button><select id='lang'><option value='en-IN'>English (India)</option><option value='ta-IN'>Tamil</option><option value='hi-IN'>Hindi</option></select></div><p id='status'>Ready. Speech recognition support depends on browser and installed language packs.</p><input id='q' style='width:100%;padding:12px;background:#0b1118;color:white;border:1px solid #456;border-radius:8px' placeholder='Ask: Which sector is most critical?'/><button id='ask' style='margin-top:10px'>Ask TALUS</button><div id='answer' style='margin-top:14px;line-height:1.5'></div><div id='history' style='margin-top:10px;color:#9eb0c2'></div></div><script>const api='""" + API_URL + """';let enabled=false;const status=document.getElementById('status'),ans=document.getElementById('answer'),q=document.getElementById('q'),lang=document.getElementById('lang');document.getElementById('enable').onclick=()=>{enabled=true;status.textContent='Voice enabled. Browser may ask for microphone permission.'};async function ask(text){if(!text.trim())return;status.textContent='Thinking…';try{const r=await fetch(api+'/assistant/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:text})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Request failed');ans.textContent=d.text;document.getElementById('history').textContent='Intent: '+d.intent+' · '+new Date().toLocaleTimeString();if(enabled&&'speechSynthesis'in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(d.speak);u.lang=lang.value;speechSynthesis.speak(u)}status.textContent='Answered by rule-based parser.'}catch(e){status.textContent='Could not reach TALUS API: '+e.message}}document.getElementById('ask').onclick=()=>ask(q.value);q.addEventListener('keydown',e=>{if(e.key==='Enter')ask(q.value)});function listen(){const SR=window.SpeechRecognition||window.webkitSpeechRecognition;if(!SR){status.textContent='Speech recognition unavailable. Use text fallback.';return}const rec=new SR();rec.lang=lang.value;rec.interimResults=false;rec.onresult=e=>{q.value=e.results[0][0].transcript;ask(q.value)};rec.onerror=e=>status.textContent='Mic issue: '+e.error+'. Use text fallback.';rec.start();status.textContent='Listening…'}document.getElementById('listen').onclick=listen;document.getElementById('wake').onclick=listen;</script>"""
    st.components.v1.html(voice_html,height=330,scrolling=False)
    st.caption("Browser speech recognition may require online browser services or installed language packs. Text fallback and rule-based answers work through the local FastAPI. No LLM/API key is required.")

elif nav == "Alert Console":
    st.markdown('<div class="hero"><h1>Alert Console</h1><p>Prototype TARP levels, acknowledgement log and demo-only alert testing.</p></div>', unsafe_allow_html=True)
    try:
        active_alerts=api_get("/alerts/active").get("alerts",[])
        level3=[a for a in active_alerts if int(a.get("level",0))==3]
        if level3: st.markdown(f'<div style="background:#541923;border:2px solid #f04455;padding:14px;border-radius:12px;color:white;font-weight:800;animation:pulse 1.4s infinite">LEVEL 3 DEMO ALERT · {", ".join(a["sector"] for a in level3)} · FOLLOW APPROVED SITE TARP</div>',unsafe_allow_html=True)
    except Exception: pass
    left,right=st.columns([1,2])
    with left:
        sector=st.selectbox("Test sector",[s.get("sector_id") for s in overview],index=min(3,max(0,len(overview)-1)))
        level=st.select_slider("Alert level",options=[0,1,2,3],value=3,format_func=lambda x:{0:"L0 Normal",1:"L1 Advisory",2:"L2 Warning",3:"L3 Evacuate"}[x])
        reason=st.text_input("Reason","Demo scenario: simulated high-risk condition")
        if st.button("TRIGGER DEMO ALERT",type="primary",use_container_width=True):
            try:
                r=api_post("/alerts/test",{"sector":sector,"level":level,"reason":reason,"demo":True}); st.session_state["last_alert_id"]=r["alert"]["id"]; st.success(f"Created alert #{r['alert']['id']}")
            except Exception as exc: st.error(str(exc))
        if st.button("RUN S4 SEVERE DEMO",use_container_width=True):
            try:
                r=api_post("/demo/critical",{}); st.session_state["last_alert_id"]=r["alert"]["id"]
                st.success(f"S4 severe demo saved. Physics FoS: {r['prediction']['factor_of_safety']:.2f}; model class: {r['prediction']['ml_risk_label']}. Alert #{r['alert']['id']} created.")
                st.warning(r["notice"])
            except Exception as exc: st.error(str(exc))
    with right:
        alerts=api_get("/alerts/history").get("alerts",[])
        if alerts: st.dataframe(pd.DataFrame(alerts),use_container_width=True,hide_index=True)
        else: st.info("No alerts have been recorded yet.")
        aid=st.number_input("Alert ID to acknowledge",min_value=1,value=int(st.session_state.get("last_alert_id",1)),step=1)
        who=st.text_input("Acknowledged by","Engineer")
        if st.button("ACKNOWLEDGE ALERT"):
            try: st.json(api_post(f"/alerts/{int(aid)}/ack",{"acknowledged_by":who})); st.cache_data.clear()
            except Exception as exc: st.error(str(exc))
        st.components.v1.html("<div style='background:#111923;padding:12px;border-radius:10px;color:#eef'><button id='testSound'>Test alert tone</button><label> Volume <input id='vol' type='range' min='0' max='1' step='0.05' value='0.35'></label><script>document.getElementById('testSound').onclick=()=>{const C=window.AudioContext||window.webkitAudioContext;if(!C)return;const c=new C(),g=c.createGain();g.gain.value=Number(document.getElementById('vol').value);g.connect(c.destination);[660,880,660,880].forEach((f,i)=>{const o=c.createOscillator();o.frequency.value=f;o.connect(g);o.start(c.currentTime+i*.18);o.stop(c.currentTime+i*.18+.12)})}</script></div>",height=70)
    st.warning("Sound is browser-side and requires a user click. No external notification is sent. The 60-second unattended escalation is not a background daemon in this prototype.")

elif nav == "Advanced Analytics":
    st.markdown('<div class="hero"><h1>Advanced Risk Analytics</h1><p>Illustrative Monte Carlo FoS sensitivity, inverse-velocity indicator and response surface.</p></div>', unsafe_allow_html=True)
    sid=st.selectbox("Sector for analysis",[s.get("sector_id") for s in overview],key="advanced_sector")
    base=next(s for s in overview if s.get("sector_id")==sid)
    payload={k:float(base.get(k,default)) for k,default in {"slope_height_m":60,"slope_angle_deg":40,"cohesion_kpa":35,"friction_angle_deg":30,"unit_weight_kn_m3":24,"rainfall_mm":25,"groundwater_level":10,"pore_pressure_kpa":50,"deformation_rate_mm_day":3.5,"displacement_mm":25,"blast_intensity":0,"equipment_load":10}.items()}
    payload.update({"sector_id":sid,"recent_blasting":int(base.get("recent_blasting",0)),"scenario_name":"Advanced analytics"})
    if st.button("RUN 5,000-SAMPLE MONTE CARLO",type="primary"):
        try:
            result=api_post("/simulate/pof",payload); a,b,c=st.columns(3); a.metric("Illustrative PoF",f"{result['probability_of_failure']:.1%}"); b.metric("Reliability index β",f"{result['reliability_index_beta']:.2f}"); c.metric("Mean FoS",f"{result['fos_mean']:.2f} ± {result['fos_std']:.2f}")
            edges=result["histogram"]["edges"]; counts=result["histogram"]["counts"]
            st.plotly_chart(plot_layout(go.Figure(go.Bar(x=[(edges[i]+edges[i+1])/2 for i in range(len(counts))],y=counts)).update_layout(title="Monte Carlo FoS distribution",xaxis_title="Factor of Safety",yaxis_title="Simulation count")),use_container_width=True)
            st.caption(result["assumption"])
        except Exception as exc: st.error(f"Monte Carlo failed: {exc}")
    try:
        ttf=api_get(f"/sector/{sid}/ttf")
        st.subheader("Inverse-velocity indicator")
        if ttf.get("available"): st.metric("Extrapolated observation steps",f"{ttf['estimated_observation_steps']:.1f}")
        st.info(ttf.get("message", "No estimate available."))
    except Exception as exc: st.caption(f"TTF indicator unavailable: {exc}")
    try:
        surfaces=api_post("/physics/response-surfaces",payload)
        angles=np.asarray(surfaces["angles"]); pressures=np.asarray(surfaces["pressures"]); aa,pp=np.meshgrid(angles,pressures); zz=np.asarray(surfaces["fos_angle_pressure"])
        surf=go.Figure(data=[go.Surface(x=aa,y=pp,z=zz,colorscale="Viridis",colorbar=dict(title="FoS"))]); surf.update_layout(title="FoS response surface · angle × pore pressure",scene=dict(xaxis_title="Slope angle (°)",yaxis_title="Pore pressure (kPa)",zaxis_title="FoS"),height=560)
        st.plotly_chart(plot_layout(surf),use_container_width=True)
        cohesion=np.asarray(surfaces["cohesion"]); friction=np.asarray(surfaces["friction"]); cc,ff=np.meshgrid(cohesion,friction); zz2=np.asarray(surfaces["fos_cohesion_friction"])
        surf2=go.Figure(data=[go.Surface(x=cc,y=ff,z=zz2,colorscale="Viridis",colorbar=dict(title="FoS"))]); surf2.update_layout(title="FoS response surface · cohesion × friction",scene=dict(xaxis_title="Cohesion (kPa)",yaxis_title="Friction angle (°)",zaxis_title="FoS"),height=560)
        st.plotly_chart(plot_layout(surf2),use_container_width=True)
    except Exception as exc: st.error(f"Physics response surfaces unavailable: {exc}")
    st.caption("PoF distributions use assumed variability, not empirically fitted parameter distributions. Response surfaces follow the simplified physics equation. Inverse-velocity estimates are indicative only.")

elif nav == "System Health":
    st.markdown('<div class="hero"><h1>System Health</h1><p>Runtime, dataset and persistence checks.</p></div>', unsafe_allow_html=True)
    try:
        detail=api_get("/system/health-detail"); c=st.columns(4); c[0].metric("API",detail.get("status","unknown")); c[1].metric("Model",detail.get("model","unknown")); c[2].metric("Dataset rows",detail.get("dataset_records",0)); c[3].metric("Response time",f"{detail.get('response_ms',0)} ms")
        st.json(detail)
        db_path=ROOT/"database/talus.db"
        if db_path.exists(): st.metric("Local database size",f"{db_path.stat().st_size/1024:.1f} KB")
    except Exception as exc: st.error(str(exc))
    st.markdown("**Architecture:** Streamlit API client → FastAPI → physics assessment + saved model → SQLite. Offline operation requires dependencies installed and both local processes running.")

elif nav == "Scenario Simulator":
    st.markdown('<div class="hero"><h1>What-if Hazard Simulator</h1><p>Change one hazard driver at a time and observe the complete physics + ML response.</p></div>',unsafe_allow_html=True)
    sid=st.selectbox("Base sector",[s["sector_id"] for s in overview],key="sim_sector")
    base=next(s for s in overview if s["sector_id"]==sid)
    st.caption("Baseline values are loaded from the live API sector state. They are not hard-coded demo numbers.")
    c1,c2,c3=st.columns(3)
    with c1: rain=st.slider("Rainfall (mm)",0.0,250.0,float(base.get("rainfall_mm",25)),5.0); pp=st.slider("Pore pressure (kPa)",0.0,400.0,float(base.get("pore_pressure_kpa",50)),5.0)
    with c2: dr=st.slider("Deformation (mm/day)",0.0,80.0,float(base.get("deformation_rate_mm_day",3.5)),1.0); blast=st.slider("Blast intensity",0.0,10.0,float(base.get("blast_intensity",0)),0.5)
    with c3: load=st.slider("Equipment load",0.0,150.0,float(base.get("equipment_load",10)),5.0); blasting=st.checkbox("Recent blasting",bool(base.get("recent_blasting",0)))
    payload={"sector_id":sid,"slope_height_m":float(base.get("slope_height_m",60)),"slope_angle_deg":float(base.get("slope_angle_deg",40)),"cohesion_kpa":float(base.get("cohesion_kpa",35)),"friction_angle_deg":float(base.get("friction_angle_deg",30)),"unit_weight_kn_m3":float(base.get("unit_weight_kn_m3",24)),"rainfall_mm":rain,"groundwater_level":float(base.get("groundwater_level",10)),"pore_pressure_kpa":pp,"deformation_rate_mm_day":dr,"displacement_mm":float(base.get("displacement_mm",25)),"recent_blasting":int(blasting),"blast_intensity":blast,"equipment_load":load,"scenario_name":"Interactive what-if"}
    if st.button("SIMULATE SCENARIO",type="primary",use_container_width=True):
        result=api_post("/predict",payload)
        a,b,c=st.columns(3)
        with a: metric("Predicted risk",result["ml_risk_label"],f'{result["ml_risk_probability"]*100:.1f}%')
        with b: metric("Factor of Safety",f'{result["factor_of_safety"]:.2f}',result["physics_category"])
        with c: metric("Decision",result["saved_db_id"],"Saved assessment ID")
        st.markdown(f'<div class="insight">{result["decision_recommendation"]}</div>',unsafe_allow_html=True)
    else:
        st.info("Adjust the drivers and run the simulation. Each run is persisted so the Risk Timeline can show the experiment.")

# -----------------------------------------------------------------------------
# System / Viva
# -----------------------------------------------------------------------------
elif nav == "System & Viva":
    st.markdown('<div class="hero"><h1>System, Deployment & Viva</h1><p>Everything needed to demonstrate the project as an end-to-end industrial-style ML prototype.</p></div>',unsafe_allow_html=True)
    tabs=st.tabs(["Architecture","Deployment","Viva","Limitations"])
    with tabs[0]:
        st.code("""Data sources\n   ↓\nPreprocessing + EDA\n   ↓\nPhysics Engine → Factor of Safety\n   ↓\nML Model → Risk class + probability\n   ↓\nXAI → local contributors\n   ↓\nSQLite history\n   ↓\nFastAPI REST API\n   ↓\nStreamlit Control Room""",language="text")
        st.markdown("**Industrial design principle:** the UI does not import the predictor for live inference. It communicates with FastAPI over HTTP, making the boundary deployable and testable.")
    with tabs[1]:
        st.markdown("1. Build backend image from `Dockerfile.backend`\n2. Build frontend image from `Dockerfile.frontend`\n3. Start both with `docker compose up --build`\n4. For Render, use `render.yaml` and set the frontend `TALUS_API_URL` to the deployed backend URL.")
        st.code("docker compose up --build",language="powershell")
        st.markdown("API documentation: `/docs` · health: `/health` · data summary: `/data-summary`")
    with tabs[2]:
        qs=[("Why physics-informed ML?","Physics gives an engineering constraint; ML captures nonlinear patterns in telemetry."),("Why FastAPI?","It creates a clean API boundary between inference services and the UI."),("Why not call the model directly from Streamlit?","Separation improves deployment, testing, security boundaries and integration with other clients."),("What is the Factor of Safety?","A physics-based stability indicator comparing resisting and driving effects; lower values indicate less stable conditions."),("Is this production mine-safety software?","No. It is a research/academic decision-support prototype using the supplied dataset and requires real-site validation."),("Why can the number of sectors change?","The sector registry is derived from processed telemetry, so the UI follows the dataset rather than a hard-coded S1–S6 list.")]
        for q,a in qs: st.markdown(f"**Q: {q}**\n\nA: {a}\n")
    with tabs[3]:
        st.warning("Current model evaluation and telemetry are based on the supplied project dataset. Real deployment requires site-specific calibration, sensor QA/QC, uncertainty analysis, geotechnical validation, drift monitoring, access controls and formal safety approval.")
        st.markdown("**Recommended next research step:** external validation on unseen mine-site data and temporal/leave-one-sector-out evaluation to measure generalization beyond the synthetic training distribution.")
