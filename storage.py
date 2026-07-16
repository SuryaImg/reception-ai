"""
storage.py — Redis-backed onboarding state + call transcript + seller URL lookup.

Key schema:
  conversation:{session_id}     → JSON blob of OnboardingState/AppointmentState fields
  booking_status:{session_id}   → booking status string
  booking_lock:{doctor_id}:{date}:{time} → lock to prevent duplicate slot bookings
  confirmed_appointment:{appointment_id} → JSON blob of confirmed appointments
"""

import asyncio
import json
import logging
import os
import re
import time
import random
from typing import Optional

from dotenv import load_dotenv
from tools import AppointmentState, CallTranscript, normalize_time_slot

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_URL                = os.environ.get("REDIS_URL", "redis://localhost:6379")
# When REDIS_REQUIRED=true, booking-critical operations (slot locking, confirmed
# appointment writes) will RAISE instead of silently falling back to per-process
# in-memory storage. In a multi-process deployment (num_idle_processes > 1) the
# in-memory fallback is NOT shared across workers, so silently using it lets two
# concurrent callers on different processes double-book the same doctor + slot.
# Set this to "true" in production so a misconfigured/unreachable Redis fails
# LOUDLY rather than silently corrupting bookings. Defaults to "false" so local
# dev without Redis keeps working.
REDIS_REQUIRED           = os.environ.get("REDIS_REQUIRED", "false").strip().lower() == "true"
REDIS_STATE_TTL_SEC      = int(os.environ.get("REDIS_STATE_TTL_DAYS", "30")) * 86400
REDIS_TRANSCRIPT_TTL_SEC = 90 * 86400   # 90 days
REDIS_SELLER_URL_TTL_SEC = 7 * 86400    # 7 days
KEY_PREFIX               = "conversation:"
SELLER_URL_PREFIX        = "seller_url:"
TRANSCRIPT_PREFIX        = "transcript:"

# ── Connection singleton ──────────────────────────────────────────────────────
_redis = None
_use_memory: bool = False
_memory_store: dict[str, AppointmentState] = {}
_memory_transcripts: dict[str, dict] = {}
_memory_seller_urls: dict[str, str] = {}

_connect_lock: Optional[asyncio.Lock] = None
_connect_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_connect_lock() -> asyncio.Lock:
    global _connect_lock, _connect_lock_loop

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        if _connect_lock is None:
            _connect_lock = asyncio.Lock()
        return _connect_lock

    if _connect_lock is None or _connect_lock_loop is not current_loop:
        _connect_lock = asyncio.Lock()
        _connect_lock_loop = current_loop

    return _connect_lock


def _reset_redis_state() -> None:
    global _redis, _use_memory, _connect_lock, _connect_lock_loop
    _redis = None
    _use_memory = False
    _connect_lock = None
    _connect_lock_loop = None
    logger.debug("Redis state reset for new worker process")


async def _execute_with_retry(func, *args, **kwargs):
    max_attempts = 3
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                f"Executing Redis operation attempt {attempt}/{max_attempts} "
                f"for function '{getattr(func, '__name__', str(func))}'"
            )
            res = await asyncio.wait_for(func(*args, **kwargs), timeout=2.0)
            logger.debug(f"Redis operation attempt {attempt}/{max_attempts} succeeded.")
            return res
        except Exception as e:
            last_err = e
            logger.warning(
                f"Redis operation attempt {attempt}/{max_attempts} failed with: {e}. "
                + (f"Retrying..." if attempt < max_attempts else "No more retries.")
            )
    raise last_err


async def _get_redis():
    global _redis, _use_memory

    if _use_memory:
        return None
    if _redis is not None:
        return _redis

    async with _get_connect_lock():
        if _redis is not None:
            return _redis
        if _use_memory:
            return None

        try:
            import redis.asyncio as aioredis

            # Guard against a common misconfig: port 6380 almost always means a
            # TLS-only Redis endpoint (Azure/Upstash/managed). Using the plaintext
            # "redis://" scheme against it makes ping() fail, which silently drops
            # us into the in-memory fallback below and defeats all cross-process
            # locking. Warn loudly so it gets caught before it hits production.
            if REDIS_URL.startswith("redis://") and ":6380" in REDIS_URL:
                logger.warning(
                    "REDIS_URL uses plaintext 'redis://' on port 6380 — that port "
                    "is usually TLS-only. If connection fails, switch to 'rediss://'."
                )

            client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=False,
            )
            await asyncio.wait_for(client.ping(), timeout=2.0)
            _redis = client
            logger.info("Redis connected: %s", REDIS_URL)
            return _redis

        except Exception as e:
            logger.warning(
                "Redis unavailable (%s) — falling back to in-memory storage", e
            )
            _use_memory = True
            return None


def _key(user_id: str) -> str:
    return f"conversation:{user_id}"


def _status_key(user_id: str) -> str:
    return f"booking_status:{user_id}"


def _seller_url_key(phone: str) -> str:
    phone = phone.strip().lstrip("+")
    return f"{SELLER_URL_PREFIX}{phone}"


def _transcript_key(user_id: str, call_ts: str) -> str:
    return f"{TRANSCRIPT_PREFIX}{user_id}:{call_ts}"


def _state_to_dict(state: AppointmentState) -> dict:
    return state.model_dump()


def _dict_to_state(data: dict) -> AppointmentState:
    return AppointmentState(**data)


def _resolve_doctor_id(doctor_name: str) -> str:
    """Resolve a doctor name to a doctor ID string. Fallbacks to normalized name if not found."""
    if not doctor_name:
        return "unknown"
    if str(doctor_name).isdigit():
        return str(doctor_name)
    
    from tools import clean_doctor_name, load_cache_data
    doc_cleaned = clean_doctor_name(doctor_name).strip().lower()
    
    cache = load_cache_data()
    if cache:
        # Check both 'list' (raw API format) and 'doctors' (normalized format)
        doctor_lists = []
        if "list" in cache:
            doctor_lists.append(cache["list"])
        if "doctors" in cache:
            doctor_lists.append(cache["doctors"])
        for doctor_list in doctor_lists:
            if not isinstance(doctor_list, list):
                continue
            for doc in doctor_list:
                # Support both 'doctorName' and 'name' field formats
                d_name = doc.get("doctorName") or doc.get("name") or ""
                d_id = doc.get("doctorId")
                if d_id is not None:
                    if doc_cleaned == clean_doctor_name(d_name).strip().lower():
                        return str(d_id)
                
    return doc_cleaned.replace(" ", "_")


# ── Onboarding State ──────────────────────────────────────────────────────────

async def check_doctor_slot_conflict(doctor_id: str, appointment_date: str, appointment_time: str) -> Optional[dict]:
    """
    Checks if a doctor already has a booked appointment at the same date and time.
    Returns the conflicting appointment data if found, else None.
    """
    if not doctor_id or not appointment_date or not appointment_time:
        return None
        
    r = await _get_redis()
    results = []
    
    target_date = appointment_date.strip()
    target_time = appointment_time.strip()
    
    if r is None:
        for k, val in list(_memory_store.items()):
            if k.startswith("confirmed_appointment:"):
                app = None
                if isinstance(val, AppointmentState):
                    app = val
                elif isinstance(val, dict):
                    d = dict(val)
                    if "status" in d and "booking_status" not in d:
                        d["booking_status"] = d["status"]
                    app = AppointmentState(**d)
                elif isinstance(val, str):
                    d = json.loads(val)
                    if "status" in d and "booking_status" not in d:
                        d["booking_status"] = d["status"]
                    app = AppointmentState(**d)
                if app:
                    results.append(app)
    else:
        try:
            async def do_scan_confirmed():
                results_local = []
                pattern = "confirmed_appointment:*"
                async for key in r.scan_iter(pattern):
                    raw = await r.get(key)
                    if raw:
                        data = json.loads(raw)
                        if "status" in data and "booking_status" not in data:
                            data["booking_status"] = data["status"]
                        results_local.append(AppointmentState(**data))
                return results_local
            results = await _execute_with_retry(do_scan_confirmed)
        except Exception as e:
            logger.warning("Redis scan failed in check_doctor_slot_conflict: %s", e)
            for k, val in list(_memory_store.items()):
                if k.startswith("confirmed_appointment:"):
                    app = None
                    if isinstance(val, AppointmentState):
                        app = val
                    elif isinstance(val, dict):
                        d = dict(val)
                        if "status" in d and "booking_status" not in d:
                            d["booking_status"] = d["status"]
                        app = AppointmentState(**d)
                    elif isinstance(val, str):
                        d = json.loads(val)
                        if "status" in d and "booking_status" not in d:
                            d["booking_status"] = d["status"]
                        app = AppointmentState(**d)
                    if app:
                        results.append(app)
                        
    for app in results:
        app_doc_id = str(app.doctor_id) if app.doctor_id is not None else ""
        if not app_doc_id and app.doctor_name:
            app_doc_id = _resolve_doctor_id(app.doctor_name)
            
        cur_doc_id = str(doctor_id)
        if app_doc_id != cur_doc_id:
            continue
            
        app_date = (app.appointment_date or "").strip()
        if app_date != target_date:
            continue
            
        app_time = normalize_time_slot(app.appointment_time or "")
        t_time = normalize_time_slot(target_time)
        if app_time != t_time:
            continue
            
        # NOTE: AppointmentState has no `status` attribute (only `booking_status`);
        # the old `app.status` fallback was a latent AttributeError masked only by
        # booking_status defaulting to a truthy "DRAFT". Use booking_status directly.
        app_status = str(app.booking_status or "").upper()
        if app_status in ("BOOKED", "CONFIRMED"):
            return {
                "doctor_name": app.doctor_name,
                "appointment_date": app.appointment_date,
                "appointment_time": app.appointment_time
            }
            
    return None


async def lock_appointment_slot(doctor_name: str, date: str, time_slot: str, user_id: str) -> bool:
    """
    Attempts to lock a specific appointment slot for a specific doctor using Redis SETNX.
    The key includes the doctor ID so that booking one doctor's slot does NOT block
    the same time slot for other doctors.
    Returns True if successfully locked, False if already taken.
    """
    doc_id = _resolve_doctor_id(doctor_name)
    d = date.strip().lower().replace(" ", "_")
    t = normalize_time_slot(time_slot).strip().lower().replace(" ", "_")
    key = f"booking_lock:{doc_id}:{d}:{t}"

    r = await _get_redis()
    if r is None:
        if REDIS_REQUIRED:
            # Fail-closed: refuse to lock via non-shared per-process memory when
            # Redis is mandatory, otherwise concurrent workers would double-book.
            raise RuntimeError(
                f"Redis unavailable and REDIS_REQUIRED=true — refusing to lock slot "
                f"{key} via in-memory fallback (would allow cross-process double-booking)."
            )
        # In-memory fallback (single-process dev only)
        existing = _memory_store.get(key)
        if existing and existing != user_id:
            logger.info("Slot already locked in memory: %s by %s (requested by %s)", key, existing, user_id)
            return False
        _memory_store[key] = user_id
        logger.info("Locked slot in memory: %s for user=%s", key, user_id)
        return True

    try:
        async def do_lock():
            # Atomic SET key value NX EX 86400 — sets only if absent AND applies
            # the 24h expiry in a SINGLE round-trip. The previous setnx()+expire()
            # pair was non-atomic: a crash between the two left a permanent lock.
            was_set = await r.set(key, user_id, nx=True, ex=86400)
            if was_set:
                return True
            # Existing lock — idempotent re-lock if this session owns it
            existing_val = await r.get(key)
            return existing_val == user_id
        result = await _execute_with_retry(do_lock)
        logger.info("Lock result for %s user=%s: %s", key, user_id, result)
        return result
    except Exception as e:
        if REDIS_REQUIRED:
            logger.error("Redis lock failed and REDIS_REQUIRED=true — aborting lock for %s: %s", key, e)
            raise
        logger.warning("Redis SET-NX failed during slot lock (%s) — memory fallback", e)
        global _use_memory
        _use_memory = True
        existing = _memory_store.get(key)
        if existing and existing != user_id:
            return False
        _memory_store[key] = user_id
        return True


async def is_slot_locked(doctor_name: str, date: str, time_slot: str) -> bool:
    """
    Checks if a slot is already locked/booked for a specific doctor.
    Uses a doctor-specific key so that different doctors' slots are independent.
    Also checks confirmed appointments in storage.
    """
    doc_id = _resolve_doctor_id(doctor_name)
    d = date.strip().lower().replace(" ", "_")
    t = normalize_time_slot(time_slot).strip().lower().replace(" ", "_")
    key = f"booking_lock:{doc_id}:{d}:{t}"

    r = await _get_redis()
    if r is None:
        # In-memory fallback: check lock key
        if key in _memory_store:
            return True
        # Also check confirmed appointments for this doctor + date + time
        conflict = await check_doctor_slot_conflict(str(doc_id), date.strip(), time_slot.strip())
        return conflict is not None

    try:
        async def do_check():
            val = await r.get(key)
            return val is not None
        locked = await _execute_with_retry(do_check)
        if locked:
            return True
        # Also check confirmed appointments for this doctor + date + time
        conflict = await check_doctor_slot_conflict(str(doc_id), date.strip(), time_slot.strip())
        return conflict is not None
    except Exception as e:
        logger.warning("Redis GET failed during slot check (%s) — memory fallback", e)
        if key in _memory_store:
            return True
        conflict = await check_doctor_slot_conflict(str(doc_id), date.strip(), time_slot.strip())
        return conflict is not None


async def get_appointment_state(user_id: str) -> Optional[AppointmentState]:
    """Fetch patient appointment state. Returns None if no prior state exists."""
    r = await _get_redis()
    if r is None:
        state = _memory_store.get(user_id)
        if state:
            status = _memory_store.get(_status_key(user_id))
            if status:
                state.booking_status = status
        return state

    try:
        async def do_read():
            raw = await r.get(_key(user_id))
            status = await r.get(_status_key(user_id))
            return raw, status

        raw, status = await _execute_with_retry(do_read)
        if raw is None:
            return None
        data = json.loads(raw)
        state = _dict_to_state(data)
        if status:
            state.booking_status = status
        else:
            state.booking_status = "DRAFT"
        logger.debug(
            "Loaded state from Redis for user=%s step=%s, status=%s",
            user_id,
            state.current_step(),
            state.booking_status,
        )
        return state
    except Exception as e:
        logger.warning("Redis GET failed (%s) — memory fallback", e)
        state = _memory_store.get(user_id)
        if state:
            status = _memory_store.get(_status_key(user_id))
            if status:
                state.booking_status = status
        return state


async def save_appointment_state(user_id: str, state: AppointmentState) -> None:
    """Upsert appointment state."""
    if state.booking_status == "BOOKED" or state.booking_status == "confirmed":
        await save_confirmed_appointment(state)
    r = await _get_redis()
    if r is None:
        _memory_store[user_id] = state
        _memory_store[_status_key(user_id)] = state.booking_status
        logger.info(
            "Saved state in memory for user=%s step=%s",
            user_id,
            state.current_step(),
        )
        return

    async def do_write_and_verify():
        payload = json.dumps(_state_to_dict(state))
        await r.set(_key(user_id), payload, ex=REDIS_STATE_TTL_SEC)
        await r.set(_status_key(user_id), state.booking_status, ex=REDIS_STATE_TTL_SEC)
        
        # Verify both
        verify_raw = await r.get(_key(user_id))
        verify_status = await r.get(_status_key(user_id))
        if verify_raw is None:
            raise ValueError(f"Write verification failed: key {_key(user_id)} not found after write")
        if verify_status is None:
            raise ValueError(f"Write verification failed: key {_status_key(user_id)} not found after write")
        
        verify_data = json.loads(verify_raw)
        if verify_status != state.booking_status:
            raise ValueError(f"Write verification failed: booking_status mismatch (expected {state.booking_status}, got {verify_status})")

    try:
        await _execute_with_retry(do_write_and_verify)
        logger.info(
            "Saved state in Redis for user=%s step=%s",
            user_id,
            state.current_step(),
        )
    except Exception as e:
        logger.warning("Redis SET failed after retries (%s) — falling back to memory", e)
        global _use_memory
        _use_memory = True
        _memory_store[user_id] = state
        _memory_store[_status_key(user_id)] = state.booking_status


# ── Seller URL Lookup ─────────────────────────────────────────────────────────

async def get_seller_url(phone: str) -> Optional[str]:
    r = await _get_redis()
    phone_normalized = phone.strip().lstrip("+")

    variants = [
        phone_normalized,
        phone_normalized[-10:],
        "91" + phone_normalized[-10:],
    ]
    variants = list(dict.fromkeys(variants))

    if r is None:
        for v in variants:
            url = _memory_seller_urls.get(v)
            if url:
                logger.info("Seller URL (memory) for phone=%s → %s", phone, url)
                return url
        logger.info("No seller URL found in memory for phone=%s", phone)
        return None

    try:
        for v in variants:
            url = await _execute_with_retry(r.get, _seller_url_key(v))
            if url:
                logger.info("Seller URL from Redis for phone=%s → %s", phone, url)
                return url
        logger.info("No seller URL in Redis for phone=%s (tried %s)", phone, variants)
        return None
    except Exception as e:
        logger.warning("Redis seller_url GET failed (%s)", e)
        return None


async def set_seller_url(phone: str, url: str) -> None:
    r = await _get_redis()
    phone_normalized = phone.strip().lstrip("+")

    if r is None:
        _memory_seller_urls[phone_normalized] = url
        logger.info("Seller URL saved in memory: %s → %s", phone_normalized, url)
        return

    try:
        await _execute_with_retry(
            r.set,
            _seller_url_key(phone_normalized),
            url,
            ex=REDIS_SELLER_URL_TTL_SEC,
        )
        logger.info("Seller URL saved in Redis: %s → %s", phone_normalized, url)
    except Exception as e:
        logger.warning("Redis seller_url SET failed (%s)", e)
        _memory_seller_urls[phone_normalized] = url


# ── Call Transcript ───────────────────────────────────────────────────────────

async def save_call_transcript(user_id: str, transcript: "CallTranscript") -> str:
    r = await _get_redis()
    call_ts = transcript.call_start_time or str(int(time.time()))
    key = _transcript_key(user_id, call_ts)

    payload = json.dumps(transcript.model_dump(), default=str)

    if r is None:
        _memory_transcripts[key] = json.loads(payload)
        logger.info("Transcript saved in memory for user=%s key=%s", user_id, key)
        return key

    try:
        await _execute_with_retry(r.set, key, payload, ex=REDIS_TRANSCRIPT_TTL_SEC)
        logger.info("Transcript saved in Redis for user=%s key=%s", user_id, key)
    except Exception as e:
        logger.warning("Redis transcript SET failed (%s) — memory fallback", e)
        _memory_transcripts[key] = json.loads(payload)

    return key


async def get_call_transcript(user_id: str, call_ts: str) -> Optional[dict]:
    r = await _get_redis()
    key = _transcript_key(user_id, call_ts)

    if r is None:
        return _memory_transcripts.get(key)

    try:
        raw = await _execute_with_retry(r.get, key)
        if raw is None:
            return _memory_transcripts.get(key)
        return json.loads(raw)
    except Exception as e:
        logger.warning("Redis transcript GET failed (%s)", e)
        return _memory_transcripts.get(key)


async def get_latest_transcript(user_id: str) -> Optional[dict]:
    r = await _get_redis()
    pattern = _transcript_key(user_id, "*")

    if r is None:
        matching = {k: v for k, v in _memory_transcripts.items() if k.startswith(f"{TRANSCRIPT_PREFIX}{user_id}:")}
        if not matching:
            return None
        latest_key = max(matching.keys())
        return matching[latest_key]

    try:
        async def do_scan_and_get():
            keys = []
            async for key in r.scan_iter(pattern):
                keys.append(key)
            if not keys:
                return None
            latest_key = max(keys)
            return await r.get(latest_key)

        raw = await _execute_with_retry(do_scan_and_get)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning("Redis transcript scan failed (%s)", e)
        return None


# ── Connection teardown ───────────────────────────────────────────────────────

async def close_pool() -> None:
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:
            pass
        _redis = None


async def generate_unique_appointment_id() -> str:
    """
    Generate a reference ID for a confirmed appointment.

    IMPORTANT: this hospital's backend integration (see scrapper.py) is
    read-only — it only exposes a doctors list, with no appointment-creation
    endpoint that could return an authoritative ID. So this ID is NOT issued
    by the hospital's own system; it is a locally-generated reference number,
    unique only within this system's own `confirmed_appointment:*` registry.
    Callers should present it as a local reference/booking number rather
    than implying it's an ID from the hospital's own records.

    Collision-checked against existing confirmed appointments — a bare
    `random.randint` with no uniqueness check could silently hand two
    different patients the same reference number.
    """
    r = await _get_redis()
    for _ in range(20):
        candidate = f"APT{random.randint(100000, 999999)}"
        key = f"confirmed_appointment:{candidate}"
        if r is not None:
            try:
                exists = await _execute_with_retry(r.exists, key)
                if not exists:
                    return candidate
                continue
            except Exception:
                break  # Redis unreachable — fall through to memory check below
        else:
            if key not in _memory_store:
                return candidate
    # Extremely unlikely fallback (20 collisions in a row, or Redis became
    # unreachable mid-check): derive from a high-resolution timestamp instead
    # of another blind retry, so it's still effectively unique.
    return f"APT{int(time.time() * 1000) % 900000 + 100000}"


async def save_confirmed_appointment(state: AppointmentState) -> None:
    """Save confirmed appointment to a global registry to support search/updates and prevent overwrite."""
    if not state.patient_phone or not state.patient_name:
        return

    if not state.appointment_id:
        state.appointment_id = await generate_unique_appointment_id()

    confirmed_data = {
        "appointment_id": state.appointment_id,
        "patient_name": state.patient_name,
        "phone_number": state.phone_number or state.patient_phone,
        "doctor_id": str(state.doctor_id) if state.doctor_id is not None else _resolve_doctor_id(state.doctor_name or ""),
        "doctor_name": state.doctor_name,
        "department": state.department,
        "appointment_date": state.appointment_date,
        "appointment_time": state.appointment_time,
        "reason": state.reason,
        "status": state.booking_status,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    confirmed_key = f"confirmed_appointment:{state.appointment_id}"
    payload = json.dumps(confirmed_data)
    
    r = await _get_redis()
    if r is None:
        if REDIS_REQUIRED:
            # A confirmed appointment written only to per-process memory is
            # invisible to other workers' duplicate/conflict checks — fail-closed.
            raise RuntimeError(
                f"Redis unavailable and REDIS_REQUIRED=true — refusing to persist "
                f"confirmed appointment {confirmed_key} to non-shared memory."
            )
        _memory_store[confirmed_key] = confirmed_data
        logger.info("Confirmed appointment saved in memory: %s", confirmed_key)
        return
        
    async def do_write_and_verify():
        await r.set(confirmed_key, payload, ex=REDIS_STATE_TTL_SEC)
        verify_raw = await r.get(confirmed_key)
        if verify_raw is None:
            raise ValueError(f"Write verification failed: key {confirmed_key} not found after write")

    try:
        await _execute_with_retry(do_write_and_verify)
        logger.info("Confirmed appointment saved in Redis: %s", confirmed_key)
    except Exception as e:
        if REDIS_REQUIRED:
            logger.error("Redis confirmed save failed and REDIS_REQUIRED=true — aborting: %s", e)
            raise
        logger.warning("Redis confirmed save failed after retries (%s) — memory fallback", e)
        global _use_memory
        _use_memory = True
        _memory_store[confirmed_key] = confirmed_data


async def delete_confirmed_appointment(state: AppointmentState) -> None:
    """Delete a confirmed appointment from the global registry."""
    if not state.appointment_id:
        return
    confirmed_key = f"confirmed_appointment:{state.appointment_id}"
    
    r = await _get_redis()
    if r is None:
        if confirmed_key in _memory_store:
            del _memory_store[confirmed_key]
        logger.info("Confirmed appointment deleted from memory: %s", confirmed_key)
        return
        
    try:
        await _execute_with_retry(r.delete, confirmed_key)
        logger.info("Confirmed appointment deleted from Redis: %s", confirmed_key)
    except Exception as e:
        logger.warning("Redis confirmed delete failed (%s) — memory fallback", e)
        global _use_memory
        _use_memory = True
        if confirmed_key in _memory_store:
            del _memory_store[confirmed_key]


async def search_confirmed_appointments(
    patient_phone: str,
    patient_name: Optional[str] = None,
    appointment_date: Optional[str] = None,
    appointment_time: Optional[str] = None,
    department: Optional[str] = None,
    doctor_name: Optional[str] = None,
    doctor_id: Optional[str] = None
) -> list[AppointmentState]:
    """
    Search for existing confirmed appointments using phone and optionally name/date/time/department/doctor.
    """
    clean_search_phone = re.sub(r"\D", "", patient_phone)[-10:]
    results = []
    
    r = await _get_redis()
    if r is None:
        for k, val in _memory_store.items():
            if k.startswith("confirmed_appointment:"):
                app = None
                if isinstance(val, AppointmentState):
                    app = val
                elif isinstance(val, dict):
                    d = dict(val)
                    if "status" in d and "booking_status" not in d:
                        d["booking_status"] = d["status"]
                    app = AppointmentState(**d)
                elif isinstance(val, str):
                    d = json.loads(val)
                    if "status" in d and "booking_status" not in d:
                        d["booking_status"] = d["status"]
                    app = AppointmentState(**d)
                if app:
                    results.append(app)
    else:
        try:
            async def do_scan_confirmed():
                results_local = []
                pattern = "confirmed_appointment:*"
                async for key in r.scan_iter(pattern):
                    raw = await r.get(key)
                    if raw:
                           data = json.loads(raw)
                           if "status" in data and "booking_status" not in data:
                               data["booking_status"] = data["status"]
                           results_local.append(AppointmentState(**data))
                return results_local
            
            results = await _execute_with_retry(do_scan_confirmed)
        except Exception as e:
            logger.warning("Redis confirmed search scan failed (%s) — memory fallback", e)
            for k, val in _memory_store.items():
                if k.startswith("confirmed_appointment:"):
                    app = None
                    if isinstance(val, AppointmentState):
                        app = val
                    elif isinstance(val, dict):
                        d = dict(val)
                        if "status" in d and "booking_status" not in d:
                            d["booking_status"] = d["status"]
                        app = AppointmentState(**d)
                    elif isinstance(val, str):
                        d = json.loads(val)
                        if "status" in d and "booking_status" not in d:
                            d["booking_status"] = d["status"]
                        app = AppointmentState(**d)
                    if app:
                        results.append(app)
                        
    # Filter the results
    filtered = []
    for app in results:
        # 1. Filter phone
        app_phone = re.sub(r"\D", "", app.phone_number or app.patient_phone or "")[-10:]
        if app_phone != clean_search_phone:
            continue
            
        # 2. Filter patient name
        if patient_name and app.patient_name:
            n1 = patient_name.lower().strip()
            n2 = app.patient_name.lower().strip()
            if n1 not in n2 and n2 not in n1:
                continue
                
        # 3. Filter date
        if appointment_date and app.appointment_date != appointment_date:
            continue
            
        # 4. Filter time
        if appointment_time and normalize_time_slot(app.appointment_time) != normalize_time_slot(appointment_time):
            continue
            
        # 5. Filter department
        if department and app.department:
            if department.lower().strip() != app.department.lower().strip():
                continue
                
        # 6. Filter doctor name
        if doctor_name and app.doctor_name:
            from tools import clean_doctor_name
            if clean_doctor_name(doctor_name).lower().strip() != clean_doctor_name(app.doctor_name).lower().strip():
                continue
                
        # 7. Filter doctor_id
        if doctor_id and app.doctor_id:
            if str(doctor_id) != str(app.doctor_id):
                continue
                
        # Reconstruct fields to ensure AppointmentState is correct
        if app.patient_phone is None and app.phone_number is not None:
             app.patient_phone = app.phone_number
        elif app.phone_number is None and app.patient_phone is not None:
             app.phone_number = app.patient_phone
             
        filtered.append(app)
        
    return filtered


async def release_appointment_slot(doctor_name: str, date: str, time_slot: str, user_id: str) -> bool:
    """Release a previously locked appointment slot."""
    r = await _get_redis()
    doc_id = _resolve_doctor_id(doctor_name)
    d = date.strip().lower().replace(" ", "_")
    t = normalize_time_slot(time_slot).strip().lower().replace(" ", "_")
    key = f"booking_lock:{doc_id}:{d}:{t}"
    
    if r is None:
        if key in _memory_store and _memory_store[key] == user_id:
            del _memory_store[key]
        logger.info("Released slot in memory: %s for user=%s", key, user_id)
        return True
        
    try:
        async def do_release():
            val = await r.get(key)
            if val == user_id:
                await r.delete(key)
            return True
        await _execute_with_retry(do_release)
        logger.info("Released slot in Redis: %s for user=%s", key, user_id)
        return True
    except Exception as e:
        logger.warning("Redis delete failed during slot release (%s)", e)
        global _use_memory
        _use_memory = True
        if key in _memory_store and _memory_store[key] == user_id:
            del _memory_store[key]