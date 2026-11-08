"""Capture execution screenshots of the latest Prefect flow run and of MLflow (for the report appendix).

Optional tooling - needs `pip install playwright` (it uses the installed Google Chrome; otherwise run
`playwright install chromium` and pass --bundled-chromium). Both UIs must be running locally:

    make prefect-server        # http://127.0.0.1:4210
    make mlflow-ui             # http://127.0.0.1:5000
    python -m scripts.capture_screenshots

Writes to reports/figures/stages/:
    flow_graph.png, flow_artifacts.png      flow-run DAG and its artifacts tab
    task_<stage>.png                        logs of every task run (one per pipeline stage)
    artifact_<key>.png                      table artifacts (validation / model comparison / backtest)
    mlflow_runs.png, mlflow_compare.png     experiment tracking
    manifest.json                           flow-run name, task states and durations
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

LEFT = 235  # px of Prefect's navigation sidebar that is cropped away


def _post(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def _get(url: str):
    return json.load(urllib.request.urlopen(url))


def main(prefect: str, mlflow: str, out: Path, experiment: str, bundled: bool) -> None:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    api = f"{prefect}/api"
    flow_run = _post(f"{api}/flow_runs/filter", {"sort": "START_TIME_DESC", "limit": 1})[0]
    tasks = _post(f"{api}/task_runs/filter", {"flow_runs": {"id": {"any_": [flow_run["id"]]}},
                                             "sort": "EXPECTED_START_TIME_ASC", "limit": 50})
    artifacts = _post(f"{api}/artifacts/filter", {"artifacts": {"flow_run_id": {"any_": [flow_run["id"]]}}, "limit": 20})
    registry = json.loads((Path(__file__).resolve().parents[1] / "models" / "registry.json").read_text())
    exp = _get(f"{mlflow}/api/2.0/mlflow/experiments/get-by-name?experiment_name={experiment}")["experiment"]

    manifest = {"flow_run": flow_run["name"], "state": flow_run["state_type"],
                "start": flow_run["start_time"], "seconds": flow_run["total_run_time"], "tasks": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel=None if bundled else "chrome",
                                    args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = browser.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=1.5)

        pg.goto(f"{prefect}/runs/flow-run/{flow_run['id']}", wait_until="domcontentloaded")
        pg.wait_for_timeout(5000)
        for label in ("Skip", "Close"):  # first-visit community dialog
            btn = pg.get_by_role("button", name=label)
            if btn.count():
                btn.first.click()
                pg.wait_for_timeout(800)
                break
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(4000)
        pg.screenshot(path=str(out / "flow_graph.png"), clip={"x": LEFT, "y": 0, "width": 1265, "height": 470})
        tab = pg.get_by_text("Artifacts", exact=True)
        if tab.count():
            tab.last.click()
            pg.wait_for_timeout(3000)
        pg.mouse.wheel(0, 420)
        pg.wait_for_timeout(1200)
        pg.screenshot(path=str(out / "flow_artifacts.png"), clip={"x": LEFT, "y": 0, "width": 1265, "height": 900})

        for t in tasks:
            stage = re.sub(r"-[0-9a-f]{3}$", "", t["name"])
            pg.goto(f"{prefect}/runs/task-run/{t['id']}", wait_until="domcontentloaded")
            pg.wait_for_timeout(3500)
            bottom = pg.evaluate("""() => {
                const els = [...document.querySelectorAll('*')].filter(e => e.children.length === 0 && /Finished in state/.test(e.textContent));
                return els.length ? els[els.length - 1].getBoundingClientRect().bottom : 900; }""")
            pg.screenshot(path=str(out / f"task_{stage}.png"),
                          clip={"x": LEFT, "y": 0, "width": 995, "height": min(900, bottom + 40)})
            manifest["tasks"].append({"stage": stage, "state": t["state_type"], "seconds": t["total_run_time"]})

        for a in artifacts:
            if a.get("type") != "table":
                continue
            pg.goto(f"{prefect}/artifacts/artifact/{a['id']}", wait_until="domcontentloaded")
            pg.wait_for_timeout(3500)
            bottom = pg.evaluate("() => { const t = document.querySelector('table'); return t ? t.getBoundingClientRect().bottom : 700; }")
            pg.screenshot(path=str(out / f"artifact_{a['key']}.png"),
                          clip={"x": LEFT, "y": 0, "width": 1265, "height": min(900, bottom + 30)})

        # MLflow: runs table and the five model runs side by side
        pg.set_viewport_size({"width": 1700, "height": 760})
        pg.goto(f"{mlflow}/#/experiments/{exp['experiment_id']}/runs", wait_until="domcontentloaded")
        pg.wait_for_timeout(6000)
        if pg.get_by_text("MLflow Assistant").count():  # close the assistant side panel if it is open
            pg.mouse.click(1663, 36)
            pg.wait_for_timeout(1000)
        pg.screenshot(path=str(out / "mlflow_runs.png"))
        ids = [registry[m]["run_id"] for m in ("popularity", "item_knn", "content", "mf", "hybrid") if m in registry]
        q = urllib.parse.quote(json.dumps(ids))
        pg.set_viewport_size({"width": 1700, "height": 1450})
        pg.goto(f"{mlflow}/#/compare-runs?runs={q}&experiments=%5B%22{exp['experiment_id']}%22%5D",
                wait_until="domcontentloaded")
        pg.wait_for_timeout(6000)
        if pg.get_by_text("MLflow Assistant").count():
            pg.mouse.click(1663, 36)
            pg.wait_for_timeout(800)
        for header in ("Visualizations", "Parameters"):
            loc = pg.get_by_text(header, exact=True)
            if loc.count():
                loc.first.click()
                pg.wait_for_timeout(600)
        pg.screenshot(path=str(out / "mlflow_compare.png"))
        browser.close()

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest['tasks'])} task screenshots for flow run {flow_run['name']} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefect", default="http://127.0.0.1:4210")
    ap.add_argument("--mlflow", default="http://127.0.0.1:5000")
    ap.add_argument("--experiment", default="foodcom-recsys")
    ap.add_argument("--out", default="reports/figures/stages")
    ap.add_argument("--bundled-chromium", action="store_true")
    a = ap.parse_args()
    main(a.prefect.rstrip("/"), a.mlflow.rstrip("/"), Path(a.out), a.experiment, a.bundled_chromium)
