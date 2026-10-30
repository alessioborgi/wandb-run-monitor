import os,json,smtplib,ssl
from email.message import EmailMessage
from pathlib import Path
from core import fetch,group_id,TARGET
path=Path("data/notified.json")
data=fetch()
groups=data["groups"]
old=set(json.loads(path.read_text()).get("notified",[])) if path.exists() else set()
new=[r for r in groups if r["finished"]>=TARGET and group_id(r) not in old]
print(f"Matched {data['matched']} finished runs; {len(new)} newly completed groups")
if new:
 message=EmailMessage()
 message["From"]=os.environ["SMTP_USER"];message["To"]=os.environ["ALERT_TO"]
 message["Subject"]=f"W&B: {len(new)} configurations reached {TARGET}"
 message.set_content("\n".join(f'{r["model"]} / {r["dataset"]} / {r["transport_type"]}: {r["finished"]}/250' for r in new)+"\nhttps://forge.coreweave.com/wandb/adavit/sheaf-mpnn-runs/table")
 with smtplib.SMTP_SSL("smtp.gmail.com",465,context=ssl.create_default_context()) as s:
  s.login(os.environ["SMTP_USER"],os.environ["SMTP_PASSWORD"])
  s.send_message(message)
 old.update(map(group_id,new))
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps({"notified":sorted(old)},indent=2)+"\n")
