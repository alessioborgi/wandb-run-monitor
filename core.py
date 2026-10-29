import os,json
from collections import Counter
PROJECT="adavit/sheaf-mpnn-runs"
TARGET=250
KEYS=("state","model","dataset","transport_type")
def fetch():
 import wandb
 key=os.environ["WANDB_API_KEY"]
 api=wandb.Api(api_key=key,overrides={"base_url":os.getenv("WANDB_BASE_URL","https://api.forge.coreweave.com")})
 rows=[];unknown=0;examined=0
 for r in api.runs(PROJECT,filters={"state":"finished"},per_page=200):
  examined+=1
  user=(getattr(r,"_attrs",{}) or {}).get("user")
  username=user.get("username") if isinstance(user,dict) else getattr(user,"username",None)
  if not username:
   unknown+=1;continue
  if username.casefold()!="fs395" or getattr(r,"state",None)!="finished":continue
  conf=r.config or {}
  vals={}
  for key in KEYS[1:]:
   v=conf.get(key,"(missing)")
   if isinstance(v,dict) and "value" in v:v=v["value"]
   vals[key]=json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else str(v)
  rows.append(dict(state="finished",**vals,run_id=r.id,name=r.name,url=getattr(r,"url","")))
 return {"runs":rows,"unknown":unknown,"examined":examined}
def aggregate(runs):
 counts=Counter(tuple(r[k] for k in KEYS) for r in runs)
 return sorted([dict(zip(KEYS,k),finished=n,remaining=max(0,TARGET-n),progress=min(100,100*n/TARGET),status="Complete" if n>=TARGET else "Almost there" if n>=200 else "In progress") for k,n in counts.items()],key=lambda r:-r["finished"])
def group_id(row):return json.dumps([row[k] for k in KEYS])
