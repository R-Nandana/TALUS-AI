"""Smoke-test existing and newly added TALUS API endpoints locally."""
from __future__ import annotations
import io, os, sys
from pathlib import Path
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.main import app

client=TestClient(app)
__test__ = False

def main():
    checks=[]
    def check(name, response, expected=200):
        ok=response.status_code==expected
        checks.append((name,ok,response.status_code))
        print(("PASS" if ok else "FAIL"),name,response.status_code)
        if not ok: print(response.text[:600])

    check("health",client.get("/health"))
    check("model info",client.get("/model-info"))
    check("data summary",client.get("/data-summary"))
    check("mine overview",client.get("/mine/overview"))
    check("assistant query",client.post("/assistant/query",json={"query":"Which sector is most critical?"}))
    check("severe S4 demo",client.post("/demo/critical",json={}))
    check("alerts active",client.get("/alerts/active"))
    check("alerts history",client.get("/alerts/history"))
    alert=client.post("/alerts/test",json={"sector":"S4","level":3,"reason":"Smoke test only","demo":True})
    check("alerts test",alert)
    if alert.status_code==200:
        aid=alert.json()["alert"]["id"]
        check("alert acknowledgement",client.post(f"/alerts/{aid}/ack",json={"acknowledged_by":"Smoke Test"}))
    check("scenario guidance",client.get("/scenario/rainfall"))
    check("sector TTF",client.get("/sector/S1/ttf"))
    check("system health detail",client.get("/system/health-detail"))
    payload={"sector_id":"S1","slope_height_m":60,"slope_angle_deg":40,"cohesion_kpa":35,"friction_angle_deg":30,"unit_weight_kn_m3":24,"rainfall_mm":25,"groundwater_level":10,"pore_pressure_kpa":50,"deformation_rate_mm_day":3.5,"displacement_mm":25,"recent_blasting":0,"blast_intensity":0,"equipment_load":10,"scenario_name":"Smoke test"}
    check("Monte Carlo PoF",client.post("/simulate/pof",json=payload))
    check("physics response surfaces",client.post("/physics/response-surfaces",json=payload))
    check("PDF report",client.get("/report/S1.pdf"))
    # Upload one project image for the material endpoint.
    image_path=next((ROOT/"data/raw/images").rglob("*.jpg"),None)
    if image_path:
        with open(image_path,"rb") as f:
            check("image classification",client.post("/image/classify",files={"file":(image_path.name,f,"image/jpeg")}))
    else:
        print("SKIP image classification: no bundled sample image found")
    failures=[x for x in checks if not x[1]]
    print(f"\n{len(checks)-len(failures)}/{len(checks)} endpoint checks passed")
    if failures: raise SystemExit(1)

if __name__ == "__main__":
    main()
