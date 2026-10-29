import os,hmac
import pandas as pd
import plotly.express as px
import streamlit as st
from core import fetch,aggregate,TARGET
st.set_page_config(page_title="Sheaf Run Monitor",page_icon="📈",layout="wide")
st.title("Sheaf Run Monitor")
st.caption("adavit/sheaf-mpnn-runs | state=finished | user=fs395 | target=250")
pwd=os.environ.get("DASHBOARD_PASSWORD")
if not pwd:st.error("Set DASHBOARD_PASSWORD on the server.");st.stop()
if not st.session_state.get("ok"):
 with st.form("login"):
  attempt=st.text_input("Dashboard password",type="password")
  submitted=st.form_submit_button("Unlock")
 if submitted and hmac.compare_digest(attempt,pwd):st.session_state.ok=True;st.rerun()
 st.stop()
@st.cache_data(ttl=105,show_spinner=False)
def read():return fetch()
with st.sidebar:
 st.code("Project: adavit/sheaf-mpnn-runs\nState: finished\nUser: fs395\nGroup: state,model,dataset,transport_type")
 auto=st.toggle("Auto refresh",value=True)
 if st.button("Refresh now"):read.clear();st.rerun()
 if st.button("Log out"):st.session_state.ok=False;st.rerun()
@st.fragment(run_every="120s" if auto else None)
def dashboard():
 try:data=read()
 except Exception as e:st.error("W&B API connection failed: "+str(e));return
 rows=data["runs"];groups=aggregate(rows)
 if data["unknown"]:st.warning(f'{data["unknown"]} finished runs lacked creator information and were excluded.')
 if not groups:st.info("No matching runs found.");return
 df=pd.DataFrame(groups)
 a,b,c,d=st.columns(4)
 a.metric("Finished runs",len(rows));b.metric("Combinations",len(df))
 c.metric("At 250",sum(df.finished>=TARGET));d.metric("Still needed",int(df.remaining.sum()))
 filtered=df.copy()
 cols=st.columns(3)
 for box,key in zip(cols,["model","dataset","transport_type"]):
  selected=box.multiselect(key,sorted(df[key].unique()))
  if selected:filtered=filtered[filtered[key].isin(selected)]
 if filtered.empty:st.info("No matching groups.");return
 st.dataframe(filtered,use_container_width=True,hide_index=True,column_config={"progress":st.column_config.ProgressColumn("Progress %",min_value=0,max_value=100,format="%.0f%%")})
 st.download_button("Download group counts",filtered.to_csv(index=False),"counts.csv")
 chart=filtered.copy();chart["Group"]=chart["model"]+" / "+chart["dataset"]+" / "+chart["transport_type"]
 fig=px.bar(chart.sort_values("finished",ascending=False).head(25),x="finished",y="Group",orientation="h",hover_data=["remaining","progress"])
 fig.update_layout(yaxis={"categoryorder":"total ascending"},height=max(350,min(len(chart),25)*28))
 fig.add_vline(x=TARGET,line_dash="dash")
 st.plotly_chart(fig,use_container_width=True)
 selection=st.selectbox("Inspect group",filtered.index.tolist(),format_func=lambda i:f'{df.loc[i,"model"]} / {df.loc[i,"dataset"]} / {df.loc[i,"transport_type"]}')
 g=df.loc[selection];details=[r for r in rows if all(r[k]==g[k] for k in ("model","dataset","transport_type"))]
 st.dataframe(pd.DataFrame(details),hide_index=True,use_container_width=True)
dashboard()
