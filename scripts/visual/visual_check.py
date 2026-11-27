"""Screenshot the running app on real public cases (visual regression / review).

Usage: python scripts/visual/visual_check.py --url http://localhost:8501 --out shots \
           --cases case_00015 case_00000 --mobile-case case_00015

Each case: overview, surgical plan, 3D plan, interactive 3D (desktop 1440 px wide).
One case is also captured at phone width (390 px). Console errors are logged per page.
Needs: pip install playwright && playwright install --with-deps chromium
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

LAUNCH_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


def wait_idle(page: Page, timeout_s: float = 900.0, quiet_s: float = 3.0) -> None:
    """Wait until Streamlit has stopped running (no status widget) for ``quiet_s``."""

    deadline = time.time() + timeout_s
    quiet_since = None
    while time.time() < deadline:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        spinner = page.locator('[data-testid="stSpinner"]').count() > 0
        if running or spinner:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since >= quiet_s:
            return
        time.sleep(0.5)
    raise TimeoutError("Streamlit did not become idle")


def click_button(page: Page, label: str) -> bool:
    button = page.get_by_role("button", name=label)
    if button.count() == 0:
        return False
    button.first.scroll_into_view_if_needed()
    button.first.click()
    wait_idle(page)
    return True


def click_tab(page: Page, label: str) -> None:
    page.get_by_role("tab", name=label).first.click()
    wait_idle(page, quiet_s=1.5)


def shoot(page: Page, path: Path, full: bool = True) -> None:
    page.wait_for_timeout(2500)  # WebGL paint
    page.screenshot(path=str(path), full_page=full)


def last_chart(page: Page, path: Path) -> None:
    charts = page.locator(".js-plotly-plot")
    if charts.count():
        chart = charts.last
        chart.scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        chart.screenshot(path=str(path))


def capture_case(page: Page, url: str, case_id: str, out: Path, prefix: str) -> dict:
    record: dict = {"case": case_id, "files": [], "errors": []}
    page.on("console", lambda m: record["errors"].append(m.text) if m.type == "error" else None)

    def save(name: str, **kwargs) -> None:
        path = out / f"{prefix}{case_id}_{name}.png"
        shoot(page, path, **kwargs)
        record["files"].append(path.name)

    page.goto(f"{url}/?case={case_id}", wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="stApp"]', timeout=120_000)
    wait_idle(page)
    save("01_overview")

    click_tab(page, "Surgical plan")
    if click_button(page, "Compute surgical plan"):
        save("02_plan")
        if click_button(page, "Build 3D plan"):
            save("03_plan_3d")
            chart = out / f"{prefix}{case_id}_04_plan_3d_chart.png"
            last_chart(page, chart)
            if chart.exists():
                record["files"].append(chart.name)

    click_tab(page, "Interactive 3D")
    if click_button(page, "Build interactive 3D"):
        save("05_interactive_3d")
        chart = out / f"{prefix}{case_id}_06_interactive_3d_chart.png"
        last_chart(page, chart)
        if chart.exists():
            record["files"].append(chart.name)
    exceptions = page.locator('[data-testid="stException"]')
    record["exceptions"] = [
        exceptions.nth(i).inner_text()[:2000] for i in range(exceptions.count())
    ]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default="shots")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--mobile-case", default=None)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=LAUNCH_ARGS)
        for case_id in args.cases:
            context = browser.new_context(viewport={"width": 1440, "height": 4200})
            page = context.new_page()
            try:
                results.append(capture_case(page, args.url, case_id, out, args.prefix))
            except Exception as exc:  # noqa: BLE001 - record and continue with other cases
                page.screenshot(
                    path=str(out / f"{args.prefix}{case_id}_failed.png"), full_page=True
                )
                results.append({"case": case_id, "failure": repr(exc)})
            context.close()
        if args.mobile_case:
            context = browser.new_context(
                viewport={"width": 390, "height": 6000}, device_scale_factor=1.5, is_mobile=True
            )
            page = context.new_page()
            try:
                results.append(
                    capture_case(page, args.url, args.mobile_case, out, f"{args.prefix}mobile_")
                )
            except Exception as exc:  # noqa: BLE001
                results.append({"case": args.mobile_case, "mobile": True, "failure": repr(exc)})
            context.close()
        browser.close()
    (out / f"{args.prefix}results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    problems = [item for item in results if item.get("failure") or item.get("exceptions")]
    if problems:
        raise SystemExit(f"{len(problems)} page(s) failed or showed a Streamlit exception")


if __name__ == "__main__":
    main()
