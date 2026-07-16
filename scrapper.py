"""
scrapper.py — Live Playwright scraper for Hospital Dashboard
"""

import json
import os
import time
from typing import Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_DIR, "hospital_cache.json")

HOSPITAL_URL = "https://stagingapis.edoovihms.com/admin/api/doctor/get_all_doctors_without_auth?hospitalId=1&consultationType=both"


def normalize_hospital_data(raw_data: dict) -> dict:
    if not isinstance(raw_data, dict):
        return {}

    normalized = dict(raw_data)

    # Map 'list' to 'doctors' if 'doctors' is not in data
    if "list" in normalized and "doctors" not in normalized:
        normalized["doctors"] = normalized["list"]

    # Normalize doctor fields
    if "doctors" in normalized and isinstance(normalized["doctors"], list):
        processed_doctors = []
        seen_depts = set()
        for doc in normalized["doctors"]:
            if not isinstance(doc, dict):
                continue
            doc_copy = dict(doc)
            
            # Map doctorName -> name
            if "doctorName" in doc_copy and "name" not in doc_copy:
                doc_copy["name"] = doc_copy["doctorName"]
                
            # Normalize specialization / department
            spec = doc_copy.get("specialization")
            if isinstance(spec, str):
                spec = spec.strip().title()
                doc_copy["specialization"] = spec
                doc_copy["department"] = spec
            elif "specialization" in doc_copy and "department" not in doc_copy:
                doc_copy["department"] = doc_copy["specialization"]
            
            processed_doctors.append(doc_copy)
            
            # Collect unique department names
            dept_name = doc_copy.get("department")
            if dept_name:
                seen_depts.add(dept_name)
                
        normalized["doctors"] = processed_doctors

        # Dynamically build departments list
        if "departments" not in normalized:
            normalized["departments"] = [{"name": d} for d in sorted(seen_depts)]

    # Do not hardcode or generate services if missing from source data
    if "services" not in normalized:
        normalized["services"] = []

    return normalized


def capture_api_data(headless: bool = True, attempt: int = 1) -> dict:
    """
    Fetch data live. Attempts a direct HTTP fetch using httpx first,
    and falls back to Playwright if needed.
    """
    # Attempt 1: Direct HTTP fetch (extremely fast and lightweight)
    try:
        import httpx
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(HOSPITAL_URL)
            if resp.status_code == 200:
                body = resp.json()
                if body:
                    normalized = normalize_hospital_data(body)
                    save_cache(normalized)
                    return normalized
            else:
                print(f"Direct fetch failed with status code {resp.status_code}. Falling back to Playwright.")
    except Exception as httpx_exc:
        print(f"Direct fetch failed: {httpx_exc}. Falling back to Playwright.")

    # Attempt 2: Playwright fallback (intercepting network requests)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed")
        return load_cache(max_age_seconds=None) or {}

    api_responses: list[dict] = []

    def handle_response(response):
        try:
            if "json" in response.headers.get("content-type", ""):
                body = response.json()
                if body:
                    api_responses.append({"url": response.url, "data": body})
        except Exception:
            pass

    def block_resources(route, request):
        if request.resource_type in ("image", "stylesheet", "font", "media"):
            route.abort()
        else:
            route.continue_()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--blink-settings=imagesEnabled=false",
                ],
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()
            page.route("**/*", block_resources)
            page.on("response", handle_response)

            page.goto(HOSPITAL_URL, wait_until="domcontentloaded", timeout=30_000)
            
            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass
            
            page.wait_for_timeout(2_000)
            browser.close()

    except Exception as e:
        print(f"Playwright error (attempt {attempt}): {e}")
        if attempt < 2:
            return capture_api_data(headless=headless, attempt=2)
        cached = load_cache(max_age_seconds=None)
        return cached if cached else {}

    # Merge intercepted JSON data — skip any payload that isn't a dict (e.g. an
    # endpoint whose JSON root is a list or a bare value) instead of letting a
    # single malformed/unexpected response crash the whole scrape.
    merged = {}
    for entry in api_responses:
        data = entry.get("data", {})
        if isinstance(data, dict):
            merged.update(data)

    if merged:
        normalized = normalize_hospital_data(merged)
        save_cache(normalized)
        return normalized
    
    cached = load_cache(max_age_seconds=None)
    return cached if cached else {}

def save_cache(data: dict) -> None:
    payload = {
        "data": data,
        "saved_at": time.time(),
        "saved_at_human": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)

def load_cache(max_age_seconds: Optional[float] = 7200.0) -> dict | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            payload = json.load(f)
        if max_age_seconds is not None:
            age = time.time() - payload.get("saved_at", 0)
            if age > max_age_seconds:
                return None
        return payload.get("data")
    except Exception:
        return None

def fetch_hospital_data() -> dict:
    return capture_api_data(headless=True)

if __name__ == "__main__":
    data = fetch_hospital_data()
    if data:
        print("Successfully fetched live data via Playwright.")
        print(f"Found keys: {list(data.keys())}")
    else:
        print("Failed to fetch data.")