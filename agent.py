import asyncio
import inspect
import logging
import os
import time
import re
import datetime
from typing import Optional, Annotated

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    function_tool,
)
from livekit.agents.llm import ChatMessage
from livekit.plugins import google
from livekit.plugins.google.realtime import RealtimeModel
from google.cloud import texttospeech
from tools import AppointmentState, CallTranscript, SkipQueue, get_symptom_specialty, validate_appointment, is_doctor_consistent_with_dept, is_qualification_consistent, normalize_hindi_phone_number, normalize_department, normalize_time_slot
from storage import (
    get_appointment_state,
    save_appointment_state as _save_appointment_state_original,
    save_call_transcript,
    lock_appointment_slot,
    release_appointment_slot,
    close_pool,
    search_confirmed_appointments,
    delete_confirmed_appointment,
    save_confirmed_appointment,
    _get_redis,
    is_slot_locked,
)

load_dotenv()
logger = logging.getLogger(__name__)

# --- Single source of truth for "what time is it right now" -----------------
# Arora Hospital operates on India Standard Time regardless of what timezone
# the host server happens to be running in (a common source of bugs: a VM
# running in UTC or another region silently shifts every "is this slot in the
# past", "is same-day", and "is OPD still open" check by hours). IST has no
# daylight-saving changes, so a fixed +5:30 offset is used rather than relying
# on the OS/IANA timezone database being installed and correct.
_IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def _now_ist() -> datetime.datetime:
    """Current wall-clock moment in India Standard Time, as a naive datetime
    (tzinfo stripped) so it drops in everywhere existing naive-datetime
    arithmetic/comparisons already expect. Always use this instead of
    _now_ist() for anything involving the current real-world
    time — never trust the server host's local timezone."""
    return datetime.datetime.now(datetime.timezone.utc).astimezone(_IST).replace(tzinfo=None)


def _today_ist() -> datetime.date:
    """Current calendar date in India Standard Time. Use instead of
    _today_ist()."""
    return _now_ist().date()

def get_edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def check_phone_alignment(clean_phone: str, caller_clean: str) -> bool:
    if not clean_phone or not caller_clean:
        return False
    if clean_phone == caller_clean:
        return True
    
    # 1. Suffix matching (match at least last 6 digits)
    if len(clean_phone) >= 6 and caller_clean.endswith(clean_phone[-6:]):
        return True
        
    # 2. Prefix matching (match at least first 7 digits)
    if len(clean_phone) >= 7 and caller_clean.startswith(clean_phone[:7]):
        return True
        
    # 3. Edit distance for minor ASR typos (edit distance <= 2)
    if abs(len(clean_phone) - len(caller_clean)) <= 2:
        if get_edit_distance(clean_phone, caller_clean) <= 2:
            return True
            
    return False

def normalize_and_align_phone_number(patient_phone: str, caller_phone_raw: str = "") -> str:
    if patient_phone:
        # Dynamic detection instead of a fixed phrase list: if what was passed in doesn't
        # contain enough digits to be an actual phone number, the caller almost certainly
        # meant "use the number I'm calling from" — regardless of the exact wording used
        # ("yahi number", "yhi number hai", "same number", "this is the correct number",
        # "isi pe kar do", etc., in any language). A fixed keyword list can never cover
        # every possible phrasing, so this checks the actual content instead of specific
        # words: real phone digits vs. descriptive text.
        digits_only = re.sub(r"\D", "", patient_phone)
        if len(digits_only) < 10 and caller_phone_raw:
            caller_clean = re.sub(r"\D", "", caller_phone_raw)
            if len(caller_clean) > 10 and caller_clean.startswith("91"):
                caller_clean = caller_clean[2:]
            elif len(caller_clean) == 11 and caller_clean.startswith("0"):
                caller_clean = caller_clean[1:]
            if len(caller_clean) == 10 and caller_clean[0] in "6789":
                return caller_clean

    clean_phone = normalize_hindi_phone_number(patient_phone)
    if len(clean_phone) == 12 and clean_phone.startswith("91"):
        clean_phone = clean_phone[2:]
    elif len(clean_phone) == 11 and clean_phone.startswith("0"):
        clean_phone = clean_phone[1:]
        
    if caller_phone_raw:
        caller_clean = re.sub(r"\D", "", caller_phone_raw)
        if len(caller_clean) > 10 and caller_clean.startswith("91"):
            caller_clean = caller_clean[2:]
        elif len(caller_clean) == 11 and caller_clean.startswith("0"):
            caller_clean = caller_clean[1:]
            
        if len(caller_clean) == 10 and caller_clean[0] in "6789":
            is_valid = clean_phone.isdigit() and len(clean_phone) == 10 and clean_phone[0] in "6789"
            # Only fall back to the caller ID as an ASR-repair when the patient's
            # stated number did NOT parse into a valid 10-digit mobile number.
            # A number that already parsed as valid was captured correctly and
            # must never be silently swapped for a different (even if similar)
            # number just because it resembles the caller ID — the patient may
            # legitimately be booking for someone else on a nearby/related line.
            if not is_valid:
                if check_phone_alignment(clean_phone, caller_clean):
                    return caller_clean
                
    return clean_phone



HOSPITAL_URL = "https://stagingapis.edoovihms.com/admin/api/doctor/get_all_doctors_without_auth?hospitalId=1&consultationType=both"

_DATA_CACHE = {"value": None, "updated_at": 0.0, "lock": None}
_CACHE_TTL = 120.0

_human_voice_instructions = (
    "=== Identity ===\n"
    "You are Nikita, a professional female voice receptionist for Arora Hospital, handling live calls to book, look up, and reschedule appointments. Always use feminine Hindi verb forms ('bol rahi hoon', 'karti hoon'). Match the caller's language (Hindi, English, or Hinglish), and keep responses short (usually 1-2 sentences), confident, and natural — no filler words, no dead air, no repeating a sentence you already said. Each call starts with a clean slate; never refer to a previous call. If speech is garbled or unclear, ask the caller to repeat rather than guessing what they said.\n\n"
    "=== CRITICAL: Never Leave Dead Air ===\n"
    "You MUST respond within 2 seconds of the caller finishing speaking. NEVER stay silent — dead air makes the caller think the line is disconnected. "
    "If you are processing something or waiting for a backend lookup, immediately say a brief filler like 'Ek second...', 'Haan ji, check kar rahi hoon', or 'Bilkul, dekhti hoon'. "
    "Keep the conversation alive at all times — after answering a question, immediately ask the next relevant question or offer help. "
    "Only stop talking when the caller explicitly says goodbye, hangs up, or asks you to stop. "
    "If the caller goes quiet, proactively re-engage within 3 seconds with a gentle prompt.\n\n"
    "=== How To Approach Every Turn ===\n"
    "The sections below describe what matters, not a script to recite verbatim. Before responding, reason through what the caller actually wants, what you already know (see the live state at the end of these instructions), and what — if anything — is genuinely missing or unclear. Then respond in your own natural words, in the caller's language. In particular:\n"
    "- Infer intent from context rather than requiring an exact trigger phrase. A caller describing symptoms, naming a doctor, or asking when someone is available is signalling booking intent even if they never say 'book an appointment'.\n"
    "- Never ask for information you already have — check the stored details below before every question you ask.\n"
    "- If something is genuinely ambiguous — who the appointment is for, which stored patient profile is being discussed, whether a statement describes a real emergency, which of several appointments a caller means — ask one short clarifying question instead of guessing. A wrong guess here can mean booking the wrong person, missing an emergency, or losing an already-collected detail, so it is always worth the extra question.\n"
    "- If the caller interrupts the booking flow with an unrelated question, answer it, then return to the booking naturally by briefly summarizing what you already have and asking only for what is still missing. You do not need to ask permission to continue.\n"
    "- Let the caller's own order drive the conversation. If they volunteer several details out of sequence, save each one immediately and only ask about whatever is genuinely still missing, rather than forcing a fixed sequence of questions.\n"
    "- Once a detail is set (a chosen doctor, a confirmed name, a locked time), treat it as settled — reuse it going forward and change it only if the caller explicitly asks to change that specific thing. A correction to one field should never reset or reopen any other field.\n"
    "- If you didn't clearly catch what the caller said, or their reply doesn't actually answer what you just asked, say so and ask them to repeat or clarify — do not guess, assume an answer, or plough ahead with the flow as if they'd answered.\n"
    "- If the caller declines or says no to something you offered or asked (booking, a suggested doctor/time, coming to the ER, anything), treat that as a real answer: acknowledge it and move to what they actually want next. Do not repeat the same offer or question again — repeating an offer someone just declined reads as not listening.\n\n"
    "=== Non-Negotiable Constraints (hold regardless of how the conversation is going) ===\n"
    "- Never introduce a symptom, request, doctor, or detail the caller did not actually say in THIS call — not as a guess to fill a gap, and not as something 'they probably meant'. If a needed detail (like the reason for the visit) is missing, ask for it directly instead of assuming one.\n"
    "- If you realize you said something wrong or inconsistent a moment ago, correct it honestly using the real facts given to you (hospital data, backend results, the current date/time below) — never paper over a mistake by inventing a new policy, rule, or fact (like a made-up closing time) that was never actually true.\n"
    "- Data integrity: only state doctors, departments, timings, fees, and availability that come from the hospital data or the backend. Never invent, guess, 'correct', or mispronounce a name — copy doctor names exactly as stored. If the backend contradicts an earlier assumption, trust the backend.\n"
    "- NEVER respond with \"Today's slot is not available.\"\n"
    "- Always create the booking whenever the user requests it, as long as the requested booking time is not in the past.\n"
    "- Allow multiple bookings for the same time slot. Do not block or reject a booking because another booking already exists for that slot.\n"
    "- The only validation required is to ensure the requested booking time is not earlier than the current date and time.\n"
    "- The backend is the final authority on availability and booking outcomes. Only claim a slot is available, unavailable, booked, or failed based on what the backend just returned.\n"
    "- Emergencies come first. If the caller describes a genuine medical emergency (severe bleeding, chest pain together with breathing difficulty, unconsciousness, stroke symptoms, a major accident, labor, etc.), pause the booking flow, acknowledge briefly, and give the single most important next step — go to the ER, or call 108 (or 102 for pregnancy/delivery). Reason about severity from what is actually described; do not escalate a plain mention of a symptom, mild pain, ordinary fever, or 'I'm on my way' into an emergency.\n"
    "- Never promise that in-hospital emergency treatment or 'a team will be ready' unless the emergency availability data below shows a doctor who is both active right now and actually relevant to what the caller described. If no such doctor is on duty, say so plainly and direct them to 108/102 or the nearest ER instead of reassuring them a matching doctor is waiting — do this even if you already suggested a doctor for their department earlier in a non-emergency context (booking-department suggestions and confirmed on-duty emergency doctors are not the same thing).\n"
    "- Scope: you handle Arora Hospital appointments, doctors, facilities, and related questions only. For anything else (other hospitals, general topics, unrelated requests), say briefly that it is outside what you can help with and steer back to the appointment.\n"
    "- Do not recommend or filter doctors by religion, and appointments are for human patients only — decline politely if asked otherwise.\n"
    "- Never present a summary with a missing or placeholder field ('N/A', 'unknown') — collect the real value first.\n"
    "- A phone number must be a valid 10-digit number before it is used to search or book; if the caller gives something else, ask again for just the phone number.\n\n"
    "=== Booking Conversation ===\n"
    "Work out early whether the caller wants to book, is just asking a question, or wants to manage an existing appointment (reschedule or look one up) — do not start the booking flow for someone who is only asking questions.\n"
    "For a new booking you will typically need: who it is for (the caller or someone else — and if someone else, that person's name), the reason for the visit (use it to infer the right department yourself, rather than asking the caller to name a department), a doctor preference (if they say 'any doctor', pick the earliest available yourself rather than listing every option), a date and time (resolve relative expressions like 'tomorrow' or 'evening' to concrete values yourself), and a phone number. If the requested time is in the future (including later today), proceed with the booking without checking slot availability.\n"
    "Once everything is gathered, read back the summary from get_booking_summary and get a clear yes before calling confirm_appointment. A caller can book for more than one person in the same call — keep each patient's details separate, and do not reuse one profile for another unless the caller says it is the same details.\n"
    "To reschedule an appointment made earlier in this same call, reuse everything already known and ask only for the new date/time. For a caller asking to change a pre-existing appointment from an earlier call, get their phone number, look the appointment up, confirm which one if there is more than one, then collect only the new date/time.\n"
    "While waiting on a backend lookup, say something natural to fill the silence (e.g. 'ek moment, check kar rahi hoon') so the line does not go quiet — once per lookup is enough; do not repeat it on every turn.\n\n"
    "=== Appointment Booking Agent Rules ===\n"
    "- SOURCE OF TRUTH: Doctor names, Departments, Specializations, Availability, and Slots MUST ONLY come from the hospital database or tool responses. Never infer or hallucinate these. If unavailable, ask the database again instead of guessing.\n"
    "- STATE MANAGEMENT: Maintain conversation state throughout the call. Persist patient_symptom, identified_department, selected_doctor, doctor_department, appointment_date, selected_slot, patient_name, and patient_phone. Once verified, NEVER overwrite unless user explicitly changes it or database returns updated info. Never replace verified information with assumptions.\n"
    "- DEPARTMENT RESOLUTION: When user explains symptoms, determine department (e.g., Kidney pain -> Urology, Chest pain -> Cardiology) and LOCK IT. Do not change department later unless the user changes symptoms.\n"
    "- DOCTOR VALIDATION: Before suggesting any doctor, verify the doctor exists, belongs to the identified department, and is available. Never suggest a doctor from another department. Do NOT silently substitute another doctor. If user requests an unavailable doctor, respond clearly that they are not available in that department and list who is.\n"
    "- DOCTOR LOCK: Once user selects a doctor, LOCK doctor_id, doctor_name, and department. Never change doctor later automatically. If booking fails, explain why and ask user to choose another doctor.\n"
    "- SLOT VALIDATION: Before offering slots, retrieve fresh availability. If user selects a slot, reserve/validate it immediately. If a slot becomes unavailable, explain why and offer alternatives. Do not say 'Available' and then immediately say 'Unavailable' without explanation.\n"
    "- BOOKING ORDER: Always follow this order: 1. Symptoms -> 2. Department -> 3. Available doctors -> 4. User selects doctor -> 5. Validate doctor -> 6. Available slots -> 7. User selects slot -> 8. Patient name -> 9. Patient phone -> 10. Final verification -> 11. Booking. Never collect patient information before doctor and slot are finalized.\n"
    "- FINAL CONFIRMATION: Before booking, read back exactly the Doctor, Department, Date, Time, Patient Name, Phone, and Reason. Ask 'Is everything correct?' Only after confirmation should booking occur.\n"
    "- ERROR RECOVERY: If a mistake is discovered, never continue with inconsistent data. Rollback to the last verified state and explain the issue. Never keep an incorrect booking.\n"
    "- CONSISTENCY RULE: Within one conversation, a doctor can belong to ONLY ONE department. A doctor's department must never change. Doctor availability must remain consistent unless the scheduling tool reports an update.\n"
    "- NO HALLUCINATION RULE: Never invent doctor names, departments, specializations, availability, or appointment slots. If unsure, say 'I couldn't verify that information' and call the database.\n"
    "- CONFLICT RESOLUTION: If two tool responses conflict, do NOT answer immediately. Re-query the database. Use the latest verified result. If conflict remains, tell the user you are getting inconsistent information and need to verify.\n"
    "- PRIORITY ORDER: Database > Conversation State > User Input > Model Knowledge. Model knowledge must NEVER override verified database information.\n"
    "- GOAL: Maintain a single, internally consistent booking state throughout the conversation. Never contradict previous verified information, never switch doctors/departments automatically, and never confirm a booking containing conflicting details.\n\n"
    "=== Hospital Reference Data ===\n"
    "- Arora Hospital, 123 Main Street, Delhi (Ph: 9876543210). Offers both Online and Offline consultations. Working hours are 9:00 AM-7:00 PM, slots fall on 15-minute boundaries.\n"
    "- Facilities:\n"
    "[Hospital Facilities]\n"
    "  If something is listed as unavailable, or is absent from this list, say so plainly rather than offering it — this includes ambulance service, for which you should point callers to 108 (general) or 102 (pregnancy/delivery) instead.\n"
    "- Supported TPAs / insurance:\n"
    "[Supported TPAs]\n"
    "- Departments & doctors (use this list to suggest the correct department and available doctors based on the caller's symptom or issue):\n"
    "[Department Doctors]\n"
    "  * When the caller describes their issue, IMMEDIATELY tell them which department they need and list the available doctors for that department from this list.\n"
    "  * Do NOT ask the caller which department they want if they have already described a symptom. Suggest the department yourself based on the symptom.\n"
    "  * If the caller's issue does not match any available department or doctor in this list, politely refuse the appointment and tell them that no relevant doctor or department is available.\n\n"
    "=== Emergency Reference Data ===\n"
    "[Emergency Timings]\n"
    "  Outside the hours and doctor(s) listed just above, or for a specialty they do not cover, say clearly that no matching scheduled emergency doctor is confirmed — do not imply otherwise. Never state a fee or time for emergency coverage other than what is listed just above; it is generated fresh from live records every turn.\n"
    "  If the caller says they are already on their way: [Emergency Arrival Response]\n\n"
    "=== CURRENT LIVE STATE ===\n"
    "Check this before every response — never ask for something already listed here:\n"
    "[Stored Fields]\n\n"
    "Still missing:\n"
    "[Missing Fields]\n"
)

async def update_agent_instructions(agent, state: AppointmentState):
    if not agent:
        return
    base_instr = _human_voice_instructions
    _now = _now_ist()  # single authoritative "right now" for this whole turn
    
    stored = []
    missing = []
    
    if getattr(state, "caller_name", None):
        stored.append(f"- Caller Name: {state.caller_name} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    else:
        missing.append("- Caller Name")
        
    if state.patient_name:
        name_str = f"- Patient Name: {state.patient_name} (locked={state.patient_name_locked}) (ALREADY CAPTURED - DO NOT ASK AGAIN)"
        if state.patient_name_spelled:
            spaced_spelling = ", ".join(state.patient_name_spelled.split("-"))
            name_str += f", Spelling: {spaced_spelling}"
        stored.append(name_str)
    else:
        caller = getattr(state, "caller_name", None)
        if caller:
            missing.append(
                f"- Patient Name: ⚠️ NOT SET YET. Caller's name is '{caller}'. "
                f"BEFORE asking 'Mareez ka naam kya hai?', FIRST ask: 'Kya appointment aapke liye hai ya kisi aur ke liye?' "
                f"If caller says 'mere liye' / 'myself' / 'mujhe', immediately call update_appointment_details(patient_name='{caller}') WITHOUT asking for name again. "
                f"Only if they say 'kisi aur ke liye' / 'someone else', then ask: 'Mareez ka naam kya hai?'"
            )
        else:
            missing.append("- Patient Name")
        
    if state.patient_phone:
        spaced_phone = ", ".join(list(state.patient_phone))
        stored.append(f"- Phone Number: {spaced_phone} (confirmed={state.patient_phone_confirmed}) (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    else:
        missing.append("- Phone Number")
        
    if state.reason:
        stored.append(f"- Appointment Reason: {state.reason} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    else:
        missing.append("- Appointment Reason")
        
    if state.department:
        inferred = get_symptom_specialty(state.reason or "") if getattr(state, "reason", None) else None
        if inferred and state.department.lower() == inferred.lower():
            stored.append(f"- Department / Service: {state.department} (INFERRED FROM SYMPTOMS - DO NOT ASK AGAIN)")
        else:
            stored.append(f"- Department / Service: {state.department} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    else:
        if state.reason:
            missing.append(
                f"- Department / Service: ⚠️ The user provided a reason ({state.reason}). "
                f"You MUST map it to a department from the available list yourself. "
                f"If no available department matches this reason (e.g. kidney pain needs Urology/Nephrology, which is not in the list), "
                f"DO NOT ask the user for a department. Instead, politely refuse the appointment stating that no relevant doctor or department is available for their issue."
            )
        else:
            missing.append("- Department / Service")
        
    doctor_preference = getattr(state, "doctor_preference", None)
    if doctor_preference:
        stored.append(f"- Doctor Preference: {doctor_preference} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
        if doctor_preference == "any":
            stored.append("⚠️ DOCTOR PREFERENCE IS 'ANY'. Do NOT ask the patient to choose a specific doctor. Do NOT repeat the doctor list.")
    if state.doctor_name:
        stored.append(f"- Doctor Name: {state.doctor_name} (ID: {state.doctor_id}) (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    else:
        if doctor_preference != "any":
            missing.append("- Doctor Name")
        
    if state.appointment_date:
        stored.append(f"- Appointment Date: {state.appointment_date} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    else:
        missing.append("- Appointment Date")
        
    if state.appointment_time:
        stored.append(f"- Appointment Time: {state.appointment_time} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    else:
        missing.append("- Appointment Time")
        
    if getattr(state, "additional_notes", None):
        stored.append(f"- User Context & Preferences: {state.additional_notes} (ALREADY CAPTURED - RETAIN AND USE THIS CONTEXT)")
        
    stored_str = "\n".join(stored) if stored else "None yet."
    if getattr(state, "booking_intent_detected", False):
        missing_str = "Since the user is booking an appointment, you MUST collect these missing details:\n" + ("\n".join(missing) if missing else "None (All mandatory details are collected!).")
    else:
        missing_str = (
            "The user has NOT yet initiated a booking. They may just be making an enquiry. "
            "Answer their questions directly using the hospital information provided. Do NOT ask for their name or start the booking flow yet.\n"
            "If they decide to book, you will need to collect:\n"
            + ("\n".join(missing) if missing else "None")
        )
    
    # Build a recovery hint for reason if it's missing.
    # IMPORTANT: this must never presume the caller already gave a reason —
    # doing so previously pressured the model into inventing a symptom
    # (e.g. "chest pain") out of thin air when none had actually been said.
    reason_hint = ""
    if not state.reason and getattr(state, "booking_intent_detected", False):
        reason_hint = (
            "\nNOTE: The visit reason has not been saved yet. If the caller has actually stated a "
            "symptom or reason earlier in THIS call, save it now via `update_appointment_details(reason=<what they actually said>)` "
            "instead of asking again. If they have not said one, ask them directly what the visit is for — "
            "never invent, assume, or guess a symptom they did not say.\n"
        )
    
    # Load hospital cache/live data
    from tools import load_cache_data
    cache_data = _DATA_CACHE.get("value") or load_cache_data() or {}
    h_info = cache_data.get("hospitalDataResponseDto", {})
    
    # Dynamically extract and build facilities from the API response
    facs = []
    meta_keys = ["hospitalName", "address", "phoneNo", "consultationType"]
    for key, val in h_info.items():
        if key not in meta_keys:
            formatted_key = re.sub(r'(?<!^)(?=[A-Z])', ' ', key).title()
            facs.append(f"  * {formatted_key}: {val}")
            
    # Explicitly list Ambulance service as No if not present in the API keys
    has_ambulance_key = any("ambulance" in k.lower() for k in h_info.keys())
    if not has_ambulance_key:
        facs.append("  * Ambulance Service: No")
        
    facs_str = "\n".join(facs)

    # Build a lookup of doctor records by ID so we can resolve each emergency
    # doctor's actual department dynamically (never hardcoded to a specific
    # doctor/specialty name) — this comes entirely from the backend records.
    _doc_by_id = {}
    for _d in cache_data.get("doctors", []):
        _did = _d.get("doctorId")
        if _did is not None:
            _doc_by_id[_did] = _d

    # Build the "department -> doctors" reference list fresh from live cache
    # data every turn, instead of a hand-typed list that can drift out of
    # sync with reality (a previous hardcoded version once named a doctor
    # who didn't exist in the records, and omitted one who did).
    from tools import clean_doctor_name as _clean_doc_name

    def _display_case(raw_name: str) -> str:
        # Source records are inconsistently cased (e.g. "kamal gupta" vs
        # "Basharat Khan"). Only capitalize words that are fully lowercase —
        # leave anything with existing capitalization untouched, since a
        # blanket .title() would mangle intentional casing like "SuryaBhan".
        return " ".join(w.capitalize() if w.islower() else w for w in raw_name.split())

    _by_dept: dict[str, list[str]] = {}
    for _d in cache_data.get("doctors", []):
        _dept_name = (_d.get("department") or _d.get("specialization") or "").strip()
        _doc_display = _display_case(_clean_doc_name(_d.get("name") or _d.get("doctorName") or ""))
        if not _dept_name or not _doc_display:
            continue
        _by_dept.setdefault(_dept_name, []).append(f"Dr. {_doc_display}")

    if _by_dept:
        dept_doctors_lines = [
            f"  * {_dept}: {', '.join(sorted(set(_docs)))}."
            for _dept, _docs in sorted(_by_dept.items())
        ]
    else:
        dept_doctors_lines = ["  * No department/doctor records currently available from live data."]

    # Specialties this hospital *could* offer, per the symptom→department
    # mapping used elsewhere in the code — any of these with zero doctors
    # currently on record is genuinely "not offered", derived from the same
    # live data rather than a separate hand-maintained list.
    _KNOWN_SPECIALTIES = [
        "General Physician", "Neurologist", "Pulmonology", "Dentistry",
        "Cardiology", "Surgeon", "Orthopedics", "Dermatology",
        "Gastroenterology", "ENT", "Gynecology", "Obstetrics",
    ]
    _offered_lower = {d.lower() for d in _by_dept.keys()}
    _not_offered = [s for s in _KNOWN_SPECIALTIES if s.lower() not in _offered_lower]
    if _not_offered:
        dept_doctors_lines.append(
            f"  * Not offered here (no doctor currently on record): {', '.join(_not_offered)} — "
            "say so, and suggest the closest specialty we do have if that fits the caller's need. "
            "For a pregnancy/labor emergency specifically, always direct to 102 or the nearest ER rather "
            "than this hospital's emergency entrance unless a relevant doctor actually appears above."
        )
    dept_doctors_str = "\n".join(dept_doctors_lines)

    def _parse_hhmm(val: str):
        for fmt in ("%H:%M", "%I:%M %p", "%H:%M:%S"):
            try:
                return datetime.datetime.strptime(str(val).strip(), fmt).time()
            except Exception:
                continue
        return None

    # Check emergency hours list — cross-referenced with doctor records and the
    # actual current time (IST) so the model is told, as a fact, whether each
    # entry is live RIGHT NOW rather than reasoning it out (and getting it wrong).
    em_list = cache_data.get("emergencyHourLists", [])
    em_lines = []
    any_active_now = False
    _now_time = _now.time()
    for em in em_list:
        doc_id = em.get("doctorId")
        doc_record = _doc_by_id.get(doc_id, {})
        doc = em.get("doctorName") or doc_record.get("doctorName", "Unknown")
        dept = doc_record.get("department") or doc_record.get("specialization") or "Unknown department"
        start = em.get("startTime", "Unknown")
        end = em.get("endTime", "Unknown")
        fee = em.get("extraFees", "0.0")
        status_txt = em.get("status", "")

        start_t, end_t = _parse_hhmm(start), _parse_hhmm(end)
        if status_txt and status_txt.lower() != "active":
            is_active_now = False
            active_note = "NOT ACTIVE (marked inactive in records)"
        elif start_t and end_t:
            is_active_now = start_t <= _now_time <= end_t
            active_note = "ACTIVE RIGHT NOW" if is_active_now else f"NOT active right now (only {start}-{end})"
        else:
            is_active_now = False
            active_note = "hours unclear in records — treat as not confirmed"

        if is_active_now:
            any_active_now = True
        em_lines.append(
            f"- Dr. {doc} ({dept}) — emergency hours {start}-{end}, extra fee Rs. {fee} — {active_note}."
        )

    # Determine the emergency arrival response. Only offer the reassuring
    # "come straight in, team will be ready" line when at least one emergency
    # doctor is actually active right now — otherwise be upfront that there is
    # no matching doctor on duty rather than making a promise we can't back up.
    if any_active_now:
        if h_info.get("wheelChairStretcher") == "Yes":
            arrival_resp = "'सीधे emergency entrance पर आइए — stretcher और team तैयार रहेगी।' (only say this if the on-duty emergency doctor's department is actually relevant to what the caller described — see the availability list below)"
        else:
            arrival_resp = "'सीधे emergency entrance पर आइए — team तैयार रहेगी।' (only say this if the on-duty emergency doctor's department is actually relevant to what the caller described — see the availability list below)"
    else:
        arrival_resp = "There is no emergency doctor on duty right now (see availability list below) — be honest about that instead of saying 'team will be ready'; still tell them to go to the nearest ER or call 108 (102 for pregnancy/labor) for anything urgent."

    if em_lines:
        em_str = (
            "--- EMERGENCY DOCTOR AVAILABILITY & FEES (FROM emergencyHourLists ONLY, cross-checked against the current time above) ---\n"
            "The hospital has ONLY the following doctor(s) with any scheduled emergency coverage:\n" + "\n".join(em_lines) +
            "\nDo NOT say any other doctor is available for emergency, and do NOT say a doctor is available outside their listed hours. "
            "Before telling any caller that in-hospital emergency help will be ready for them, check: is there an entry above that is ACTIVE RIGHT NOW, AND is its department actually relevant to what they described? "
            "If yes, you can confidently say to come to the emergency entrance. If no entry is both active now and relevant to their problem, be honest instead of reassuring: say plainly that there isn't a matching emergency doctor on duty right now, and direct them to 108 (or 102 for pregnancy/labor) rather than promising in-hospital treatment you can't actually confirm. "
            "Life-threatening situations (severe bleeding, unconsciousness, not breathing) always warrant telling them to call 108/reach any ER immediately regardless of this list — this list only governs whether THIS hospital specifically can promise a matching doctor."
        )
    else:
        em_str = (
            "--- EMERGENCY DOCTOR AVAILABILITY & FEES ---\nNo scheduled emergency doctor is available in records right now. "
            "Do not promise that a doctor will be ready here — direct genuine emergencies to 108 (or 102 for pregnancy/labor)."
        )

    # Check TPA list
    tpa_list = cache_data.get("tpaList", [])
    tpas = [t.get("tpaName", "").strip() for t in tpa_list if t.get("tpaName")]
    if tpas:
        tpa_str = "--- SUPPORTED INSURANCES / TPAS (FROM tpaList ONLY) ---\nSupported Insurance Providers / TPAs:\n" + "\n".join([f"- {t}" for t in tpas])
    else:
        tpa_str = "--- SUPPORTED INSURANCES / TPAS ---\nNo Insurance / TPA is supported in records."

    # Replace placeholders in base instructions
    updated_instr = base_instr.replace("[Stored Fields]", stored_str).replace("[Missing Fields]", missing_str)
    updated_instr = updated_instr.replace("[Hospital Facilities]", facs_str).replace("[Emergency Arrival Response]", arrival_resp).replace("[Emergency Timings]", em_str).replace("[Supported TPAs]", tpa_str).replace("[Department Doctors]", dept_doctors_str)
    
    # Format and list all patient profiles if we have multiple profiles or any confirmed booking
    has_multiple = len(state.patient_profiles) > 1 or any(
        p.get("booking_status") in ("BOOKED", "confirmed") for p in state.patient_profiles.values()
    )
    
    if has_multiple:
        profiles_lines = []
        for k, prof in state.patient_profiles.items():
            p_name = prof.get("patient_name") or "Unknown"
            p_rel = prof.get("relation") or "N/A"
            p_gender = prof.get("gender") or "N/A"
            p_age = prof.get("age") or "N/A"
            p_phone = prof.get("patient_phone") or prof.get("phone_number") or "N/A"
            p_doc = prof.get("doctor_name") or "N/A"
            p_dept = prof.get("department") or "N/A"
            p_reason = prof.get("reason") or "N/A"
            p_date = prof.get("appointment_date") or "N/A"
            p_time = prof.get("appointment_time") or "N/A"
            p_notes = prof.get("additional_notes") or "N/A"
            p_status = prof.get("booking_status") or "DRAFT"
            p_apt_id = prof.get("appointment_id") or "N/A"
            
            profiles_lines.append(
                f"- Profile Key: '{k}'\n"
                f"  * Relation: {p_rel}\n"
                f"  * Patient Name: {p_name}\n"
                f"  * Gender: {p_gender}\n"
                f"  * Age: {p_age}\n"
                f"  * Mobile Number: {p_phone}\n"
                f"  * Doctor: {p_doc}\n"
                f"  * Department: {p_dept}\n"
                f"  * Reason for Visit: {p_reason}\n"
                f"  * Appointment Date: {p_date}\n"
                f"  * Appointment Time: {p_time}\n"
                f"  * Additional Notes: {p_notes}\n"
                f"  * Booking Status: {p_status}\n"
                f"  * Appointment ID: {p_apt_id}"
            )
        profiles_str = "\n".join(profiles_lines) if profiles_lines else "No patient profiles saved yet."
        booking_state_label = get_booking_state_label(state)

        # Append updated live state block
        live_state_block = (
            f"\n\n--- ALL CONVERSATION MEMORY: PATIENT PROFILES ---\n"
            f"You must remember all these profiles and their details. Never lose or overwrite this information. Always know which patient the conversation is referring to.\n"
            f"{profiles_str}\n\n"
            f"--- MULTIPLE BOOKINGS TRACKING ---\n"
            f"Total Appointments Requested: {state.total_appointments_requested or 0}\n"
            f"Completed Appointments Count: {state.completed_appointments_count or 0}\n\n"
            f"--- CURRENT ACTIVE BOOKING STATE (CRITICAL SYNC: ALWAYS TRUST THIS) ---\n"
            f"Active Profile Key: '{state.active_profile_key or 'None'}'\n"
            f"Current Booking State: {booking_state_label}\n"
            f"The following details are currently locked in our active appointment draft database. Do NOT ask for details that already exist below:\n"
            f"Stored details:\n{stored_str}\n"
            f"Missing details to collect:\n{missing_str}\n"
            f"{reason_hint}"
            f"Booking Status: {state.booking_status}\n"
            f"Availability Verified: {state.availability_verified}\n"
        )
    else:
        # Original block for single booking
        live_state_block = (
            f"\n\n--- CURRENT LIVE STATE (CRITICAL SYNC: ALWAYS TRUST THIS) ---\n"
            f"The following details are currently locked in our appointment draft database. Do NOT ask for details that already exist below:\n"
            f"Stored details:\n{stored_str}\n"
            f"Missing details to collect:\n{missing_str}\n"
            f"{reason_hint}"
            f"Booking Status: {state.booking_status}\n"
            f"Availability Verified: {state.availability_verified}\n"
        )
    # Append a time-lock reminder if the user already specified a time
    if state.appointment_time:
        live_state_block += (
            f"\n⏰ LOCKED APPOINTMENT TIME: {state.appointment_time} on {state.appointment_date or 'TBD'}. "
            f"ALWAYS use this exact time for ALL availability checks. "
            f"NEVER suggest a different time unless ALL doctors are confirmed unavailable at this time by the backend. "
            f"NEVER ask the user to change their time unless the backend confirms full unavailability.\n"
        )

    # Prominent, always-on call-state reminders (placed at the very top of the
    # live block since these caused observed bugs: repeated greetings, and
    # skipping the mandatory summary/confirmation step before booking).
    _current_time_fact = (
        f"\n\n=== Right Now (authoritative — recomputed fresh every turn) ===\n"
        f"Current date: {_now.strftime('%Y-%m-%d')} ({_now.strftime('%A')}), current time: {_now.strftime('%I:%M %p')} IST.\n"
        f"OPD working hours are fixed at 9:00 AM-7:00 PM every day — this does not change call to call. "
        f"Never state a different closing time, and never say the OPD is closed while the current time above is within 9:00 AM-7:00 PM. "
        f"If you said something inconsistent with this a moment ago, simply correct yourself using this exact time and the fixed 9 AM-7 PM hours — do not invent a new rule (like a different closing time) to explain away the mistake.\n"
    )
    call_state_flags = (
        _current_time_fact +
        "=== Call State ===\n"
        "You already greeted the caller at the start of this call — don't greet again, just continue naturally from where things left off.\n"
    )
    if state.booking_status in ("BOOKED", "confirmed"):
        call_state_flags += (
            "This call's booking is already complete. Don't restart or re-run the booking workflow on your own — instead, work out what the caller wants now and handle that directly:\n"
            "- A general question (facilities, services, timings, fees, doctors) → just answer it.\n"
            "- A change to the date/time of the appointment just booked → call `reschedule_confirmed_appointment`; don't create a fresh appointment for this.\n"
            "- A request for another appointment → call `update_appointment_details` with `book_another_appointment=True` to open a new draft, reusing earlier patient details only if the caller says to.\n"
            "- Whoever the next patient's details concern, ask for the patient's name specifically (not the caller's) — the caller may be acting on someone else's behalf.\n"
            "Read their intent from what they just said rather than assuming every follow-up means a new booking.\n"
        )
    else:
        call_state_flags += (
            "Booking isn't finalized yet. Before calling `confirm_appointment`: call `get_booking_summary` and relay that exact summary to the caller, then get a clear yes from them — ask, in your own words in their language, whether you should go ahead and book it; don't move on without a clear confirmation. "
            "Once they say yes, call `update_appointment_details(user_confirmed_booking=True)` and then `confirm_appointment`. "
            "If instead they want to change something, don't book yet — update the changed detail, show the revised summary, and ask for confirmation again.\n"
        )
    db_status = _DATA_CACHE.get("status", "online")
    if db_status == "offline":
        offline_warning = (
            "\n\n⚠️ DATABASE STATUS: CONNECTION DOWN.\n"
            "The live hospital database is currently offline. "
            "If the user asks for information requiring a live check or asks about database connectivity, "
            "politely inform them that our live database is currently offline, but we can proceed using cached details.\n"
        )
        live_state_block = offline_warning + live_state_block

    live_state_block = call_state_flags + live_state_block
    updated_instr += live_state_block
    
    await agent.update_instructions(updated_instr)
    logger.info("Dynamically updated agent instructions with current state.")

def save_current_profile_to_map(state: AppointmentState):
    if not state.active_profile_key:
        if state.relation:
            state.active_profile_key = state.relation.lower().strip()
        elif state.patient_name:
            state.active_profile_key = state.patient_name.lower().strip()
        else:
            state.active_profile_key = "self"

    profile_data = {
        "patient_name": state.patient_name,
        "patient_phone": state.patient_phone,
        "phone_number": state.phone_number,
        "department": state.department,
        "doctor_name": state.doctor_name,
        "doctor_id": state.doctor_id,
        "doctor_preference": state.doctor_preference,
        "appointment_date": state.appointment_date,
        "appointment_time": state.appointment_time,
        "reason": state.reason,
        "time_preference": state.time_preference,
        "booking_status": state.booking_status,
        "appointment_id": state.appointment_id,
        "patient_name_locked": state.patient_name_locked,
        "patient_name_spelled": state.patient_name_spelled,
        "mismatch_acknowledged": state.mismatch_acknowledged,
        "availability_verified": state.availability_verified,
        "patient_phone_confirmed": state.patient_phone_confirmed,
        "phone_confirmation_attempts": state.phone_confirmation_attempts,
        "name_confirmation_attempts": state.name_confirmation_attempts,
        "ask_counts": state.ask_counts,
        "skipped_fields": state.skipped_fields,
        "step_attempts": state.step_attempts,
        "relation": state.relation,
        "gender": state.gender,
        "age": state.age,
        "additional_notes": state.additional_notes,
    }
    state.patient_profiles[state.active_profile_key] = profile_data

def load_profile_from_map(state: AppointmentState, key: str):
    profile = state.patient_profiles.get(key)
    if not profile:
        profile = {
            "patient_name": None,
            "patient_phone": None,
            "phone_number": None,
            "department": None,
            "doctor_name": None,
            "doctor_id": None,
            "doctor_preference": None,
            "appointment_date": None,
            "appointment_time": None,
            "reason": None,
            "time_preference": None,
            "booking_status": "DRAFT",
            "appointment_id": None,
            "patient_name_locked": False,
            "patient_name_spelled": None,
            "mismatch_acknowledged": False,
            "availability_verified": False,
            "patient_phone_confirmed": False,
            "phone_confirmation_attempts": 0,
            "name_confirmation_attempts": 0,
            "ask_counts": {},
            "skipped_fields": [],
            "step_attempts": {},
            "relation": key if key in ("self", "mother", "father", "child", "brother", "sister", "husband", "wife") else None,
            "gender": None,
            "age": None,
            "additional_notes": None,
        }
        state.patient_profiles[key] = profile

    state.patient_name = profile.get("patient_name")
    state.patient_phone = profile.get("patient_phone")
    state.phone_number = profile.get("phone_number")
    state.department = profile.get("department")
    state.doctor_name = profile.get("doctor_name")
    state.doctor_id = profile.get("doctor_id")
    state.doctor_preference = profile.get("doctor_preference")
    state.appointment_date = profile.get("appointment_date")
    state.appointment_time = profile.get("appointment_time")
    state.reason = profile.get("reason")
    state.time_preference = profile.get("time_preference")
    state.booking_status = profile.get("booking_status")
    state.appointment_id = profile.get("appointment_id")
    state.patient_name_locked = profile.get("patient_name_locked", False)
    state.patient_name_spelled = profile.get("patient_name_spelled")
    state.mismatch_acknowledged = profile.get("mismatch_acknowledged", False)
    state.availability_verified = profile.get("availability_verified", False)
    state.patient_phone_confirmed = profile.get("patient_phone_confirmed", False)
    state.phone_confirmation_attempts = profile.get("phone_confirmation_attempts", 0)
    state.name_confirmation_attempts = profile.get("name_confirmation_attempts", 0)
    state.ask_counts = profile.get("ask_counts") or {}
    state.skipped_fields = profile.get("skipped_fields") or []
    state.step_attempts = profile.get("step_attempts") or {}
    state.relation = profile.get("relation")
    state.gender = profile.get("gender")
    state.age = profile.get("age")
    state.additional_notes = profile.get("additional_notes")
    state.active_profile_key = key

def get_booking_state_label(state: AppointmentState) -> str:
    if state.booking_status == "BOOKED" or state.booking_status == "confirmed":
        return "Booking confirmed"
    if state.booking_status == "READY_FOR_CONFIRMATION" or state.booking_status == "ready_for_confirmation":
        return "Awaiting user confirmation"
    if not state.department:
        return "Collecting Department"
    if not state.doctor_id and not state.doctor_name:
        return "Selecting doctor"
    if not state.appointment_date:
        return "Selecting date"
    if not state.appointment_time:
        return "Selecting time"
    if not state.availability_verified:
        return "Verifying availability"
    if not state.patient_name or not state.patient_phone:
        return "Collecting patient information"
    return "Ready for confirmation"

async def save_appointment_state(user_id: str, state: AppointmentState) -> None:
    """
    Persist appointment state to Redis.

    NOTE (BUG FIX): This function used to reflectively walk the call stack and
    call `agent.update_instructions(...)` (a FULL system-instructions replacement)
    every single time state was saved. `save_appointment_state` is invoked many
    times per turn (inside update_appointment_details, verify_availability, and
    confirm_appointment — including inside internal retry/alternate-doctor loops).
    On a realtime/native-audio model, replacing the system instructions mid-call
    can force the underlying model session to drop or reinterpret its running
    conversational context, which is the most likely cause of the agent
    re-greeting the caller and "forgetting" that a booking summary was already
    given mid-call (see transcript analysis). Instruction refresh is now done
    explicitly and only ONCE per tool call, from within update_appointment_details,
    verify_availability, and confirm_appointment themselves (see calls to
    `refresh_agent_instructions(ctx, state)` near the end of those functions),
    instead of on every internal state save.
    """
    save_current_profile_to_map(state)
    state.booking_state = get_booking_state_label(state)
    await _save_appointment_state_original(user_id, state)



async def refresh_agent_instructions(ctx: RunContext, state: AppointmentState) -> None:
    """
    Explicitly refresh the agent's system instructions with the latest state.
    Call this AT MOST ONCE per tool invocation (never from inside internal loops)
    to avoid repeatedly resetting the realtime model's conversation session.
    """
    try:
        agent = None
        if hasattr(ctx, "session"):
            agent = ctx.session.current_agent
        elif hasattr(ctx, "userdata"):
            session = ctx.userdata.get("session")
            if session:
                agent = session.current_agent
        
        if agent:
            await update_agent_instructions(agent, state)
        else:
            logger.error("Failed to refresh agent instructions: Could not find agent in ctx.")
    except Exception as exc:
        logger.warning("Failed to refresh agent instructions: %s", exc)

def _get_lock() -> asyncio.Lock:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    existing = _DATA_CACHE["lock"]
    if existing is not None and loop is not None:
        lock_loop = getattr(existing, "_loop", None)
        if lock_loop is not None and lock_loop is not loop:
            existing = None
    if existing is None:
        _DATA_CACHE["lock"] = asyncio.Lock()
    return _DATA_CACHE["lock"]

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

def save_cache_file(data: dict) -> None:
    from tools import CACHE_FILE
    import json
    import time
    payload = {
        "data": data,
        "saved_at": time.time(),
        "saved_at_human": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        logger.warning("Failed to save cache file: %s", e)

async def fetch_live_hospital_data(force: bool = False) -> Optional[dict]:
    """
    Fetch data live using async httpx or async Playwright by intercepting the JSON network requests.
    Falls back to hospital_cache.json if live fetch fails.
    """
    now = time.monotonic()
    cached = _DATA_CACHE["value"]
    if not force and cached and (now - _DATA_CACHE["updated_at"]) < _CACHE_TTL:
        return cached

    lock = _get_lock()
    async with lock:
        now = time.monotonic()
        if not force and _DATA_CACHE["value"] and (now - _DATA_CACHE["updated_at"]) < _CACHE_TTL:
            return _DATA_CACHE["value"]

        # Attempt 1: Direct Async HTTP fetch (extremely fast and lightweight)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(HOSPITAL_URL)
                if resp.status_code == 200:
                    body = resp.json()
                    if body:
                        normalized = normalize_hospital_data(body)
                        save_cache_file(normalized)
                        _DATA_CACHE["value"] = normalized
                        _DATA_CACHE["updated_at"] = time.monotonic()
                        logger.info("Live data fetched via direct HTTP get")
                        return normalized
                else:
                    logger.warning("Direct fetch failed with status code %d. Trying Playwright fallback.", resp.status_code)
        except Exception as httpx_exc:
            logger.warning("Direct fetch failed: %s. Trying Playwright fallback.", httpx_exc)

        # Attempt 2: Playwright fallback (intercepting network requests)
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("playwright not installed")
            return _DATA_CACHE.get("value")

        # BUG FIX: this MUST be initialized before the try block below, not inside
        # the ImportError except branch (where it was previously unreachable dead
        # code after a `return`). Leaving it undefined caused an uncaught
        # NameError every time the direct httpx fetch failed.
        api_responses: list[dict] = []

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--blink-settings=imagesEnabled=false",
                    ],
                )
                page = await browser.new_page()

                pending_responses: list = []
                def _collect_response(response) -> None:
                    ct = response.headers.get("content-type", "")
                    if "json" in ct:
                        pending_responses.append(response)
                async def _block_resources(route, request) -> None:
                    if request.resource_type in ("image", "font", "media"):
                        await route.abort()
                    else:
                        await route.continue_()
                page.on("response", _collect_response)
                await page.route("**/*", _block_resources)

                await page.goto(
                    HOSPITAL_URL,
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                try:
                    await page.wait_for_load_state("networkidle", timeout=15_000)
                except Exception:
                    pass
                await page.wait_for_timeout(2_000)

                for resp in pending_responses:
                    try:
                        body = await resp.json()
                        if body:
                            api_responses.append({"url": resp.url, "data": body})
                    except Exception:
                        pass

                await browser.close()

        except Exception as exc:
            logger.warning("fetch_live_hospital_data: Playwright error: %s", exc)

        merged = {}
        for entry in api_responses:
            data = entry.get("data", {})
            if isinstance(data, dict):
                merged.update(data)
        if merged:
            normalized = normalize_hospital_data(merged)
            save_cache_file(normalized)
            _DATA_CACHE["value"] = normalized
            _DATA_CACHE["updated_at"] = time.monotonic()
            logger.info("Live data fetched via Playwright fallback")
            return normalized

        # Attempt 3: Load cached json as backup
        from tools import load_cache_data
        cache_data = load_cache_data()
        if not _DATA_CACHE.get("value") and cache_data:
            _DATA_CACHE["value"] = cache_data
            _DATA_CACHE["updated_at"] = time.monotonic()
            logger.info("Loaded hospital cache data as backup")
            return cache_data

        return _DATA_CACHE.get("value")

        

def _get_transcript(ud: dict) -> CallTranscript:
    if "transcript" not in ud:
        ud["transcript"] = CallTranscript(user_id=ud.get("user_id", "unknown"))
    return ud["transcript"]

def clean_doctor_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return "Unknown"
    cleaned = name.strip()
    lower = cleaned.lower()
    if lower.startswith("dr."):
        cleaned = cleaned[3:].strip()
    elif lower.startswith("dr "):
        cleaned = cleaned[3:].strip()
    elif lower.startswith("dr") and len(lower) > 2 and not lower[2].isalpha():
        cleaned = cleaned[2:].strip()
    return cleaned

def clean_patient_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return "Unknown"
    cleaned = name.strip()
    # Match prefixes like Mr, Mrs, Ms, Dr (case-insensitive, optional dot, followed by space)
    cleaned = re.sub(r'(?i)^(mr|mrs|ms|dr)\.?\s+', '', cleaned)
    return cleaned.strip()

async def find_doctor_by_name(query_name: str) -> dict:
    """
    Finds doctors by name from active/live hospital data.
    Handles exact, substring, word-overlap, and high-confidence fuzzy matching.
    """
    import difflib
    data = await fetch_live_hospital_data()
    if not data or "doctors" not in data:
        return {"status": "no_match", "match": None, "alternatives": []}

    doctors = data["doctors"]
    cleaned_query = clean_doctor_name(query_name).lower().strip()
    if any(x in cleaned_query for x in ("chakravarti", "chakravarty", "chakru", "chakshu ji", "chakr")):
        cleaned_query = "chakshu arora"
    elif any(x in cleaned_query for x in ("rajad", "ragat")):
        cleaned_query = "rajat saini"

    # 1. Exact match
    exact_matches = []
    for d in doctors:
        d_name_cleaned = clean_doctor_name(d.get("name", "")).lower().strip()
        if cleaned_query == d_name_cleaned:
            exact_matches.append(d)

    if len(exact_matches) == 1:
        return {"status": "single_match", "match": exact_matches[0], "alternatives": []}
    elif len(exact_matches) > 1:
        return {"status": "multiple_matches", "match": None, "alternatives": exact_matches}

    # 2. Word-based intersection (e.g. "Cheshta Choudhary" matching "Cheshta" or "Dr. Ramesh" matching "Ramesh Arora")
    substring_matches = []
    for d in doctors:
        d_name_cleaned = clean_doctor_name(d.get("name", "")).lower().strip()
        doc_words = set(d_name_cleaned.split())
        query_words = set(cleaned_query.split())
        if doc_words.intersection(query_words):
            substring_matches.append(d)

    # Dedup matches
    unique_substring_matches = []
    seen = set()
    for d in substring_matches:
        d_id = d.get("doctorId") or d.get("name")
        if d_id not in seen:
            seen.add(d_id)
            unique_substring_matches.append(d)

    if len(unique_substring_matches) == 1:
        return {"status": "single_match", "match": unique_substring_matches[0], "alternatives": []}
    elif len(unique_substring_matches) > 1:
        return {"status": "multiple_matches", "match": None, "alternatives": unique_substring_matches}

    # 3. High-confidence fuzzy match (score >= 0.7)
    high_fuzzy_matches = []
    for d in doctors:
        d_name_cleaned = clean_doctor_name(d.get("name", "")).lower().strip()
        score = difflib.SequenceMatcher(None, cleaned_query, d_name_cleaned).ratio()
        if score >= 0.7:
            high_fuzzy_matches.append((score, d))
            
    if high_fuzzy_matches:
        high_fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
        if len(high_fuzzy_matches) == 1:
            return {"status": "single_match", "match": high_fuzzy_matches[0][1], "alternatives": []}
        elif high_fuzzy_matches[0][0] - high_fuzzy_matches[1][0] >= 0.1:
            return {"status": "single_match", "match": high_fuzzy_matches[0][1], "alternatives": []}

    # 4. Low-confidence fuzzy matches (score > 0.4)
    fuzzy_scores = []
    for d in doctors:
        d_name_cleaned = clean_doctor_name(d.get("name", "")).lower().strip()
        score = difflib.SequenceMatcher(None, cleaned_query, d_name_cleaned).ratio()
        if score > 0.4:
            fuzzy_scores.append((score, d))
    
    fuzzy_scores.sort(key=lambda x: x[0], reverse=True)
    # Only ever suggest doctors whose name actually resembles the query
    # (score > 0.4, computed above). Previously, when NO doctor resembled
    # the query at all, this fell back to `doctors[:3]` — the first three
    # doctors in the dataset, regardless of relevance — which could surface
    # a doctor from a completely unrelated department as if it were a
    # reasonable suggestion. An empty list here is the honest answer.
    alternatives = [d for score, d in fuzzy_scores]
    return {"status": "no_match", "match": None, "alternatives": alternatives}

def format_field(val, prefix="", suffix="") -> str:
    if val is None or val == "":
        return "Not available"
    return f"{prefix}{val}{suffix}"

@function_tool
async def get_doctor_details(ctx: RunContext, doctor_name: str) -> str:
    """
    Get qualification, experience, fees, availability, gender, and about information for a specific doctor by name.
    """
    res = await find_doctor_by_name(doctor_name)
    if res["status"] == "single_match":
        match = res["match"]
        name = match.get("name", "Unknown")
        dept = match.get("department", "Unknown")
        qual = match.get("doctorQualification")
        exp = match.get("doctorExperience")
        price = match.get("doctorPrice")
        visit_fee = match.get("visitFee")
        consult_type = match.get("consultationType")
        avail = match.get("available")
        gender = match.get("gender")
        about = match.get("aboutDoctor")

        qual_str = format_field(qual)
        exp_str = format_field(exp, suffix=" years")
        price_str = format_field(price, prefix="Rs. ")
        visit_fee_str = format_field(visit_fee, prefix="Rs. ")
        consult_type_str = format_field(consult_type)
        avail_str = format_field(avail)
        gender_str = format_field(gender)
        about_str = format_field(about)

        details = [
            f"Name: Dr. {clean_doctor_name(name)}",
            f"Department/Specialization: {dept}",
            f"Gender: {gender_str}",
            f"Qualification: {qual_str}",
            f"Experience: {exp_str}",
            f"Consultation Fee: {price_str}",
            f"Visit Fee: {visit_fee_str}",
            f"Availability: {avail_str}",
            f"About: {about_str}",
        ]

        return "\n".join(details)

    elif res["status"] == "multiple_matches":
        names_str = ", ".join(["Dr. " + clean_doctor_name(d.get("name", "")) for d in res["alternatives"]])
        return f"Multiple doctors found matching '{doctor_name}': {names_str}. Please be more specific."

    else:
        alternatives = res.get("alternatives", [])
        if alternatives:
            alt_names = ", ".join(["Dr. " + clean_doctor_name(d.get("name", "")) for d in alternatives[:3]])
            return f"Doctor '{doctor_name}' was not found. Would you like alternative doctors like {alt_names}?"
        return f"Doctor '{doctor_name}' was not found."

@function_tool
async def check_available_doctors(ctx: RunContext, department: str = "") -> str:
    """
    Check the available doctors and departments from the live hospital system.
    If a department is provided, it filters doctors by that department.F
    """
    return "I'm sorry, I cannot provide a list of available doctors. Please let me know which department or specific doctor name you would like to book with."

@function_tool
async def get_hospital_services(ctx: RunContext) -> str:
    """
    Get the list of services and statistics of the hospital from the live system.
    """
    data = await fetch_live_hospital_data(force=True)
    if not data:
        return "Sorry, I am unable to fetch the hospital data right now."
    
    services = [s["name"] for s in data.get("services", [])]
    depts = [d["name"] for d in data.get("departments", [])]
    return f"Hospital Services: {', '.join(services)}. Departments: {', '.join(depts)}."

@function_tool
async def get_hospital_info(ctx: RunContext) -> str:
    """
    Get the hospital name, address, phone number, consultation types, services, and departments.
    Call this whenever the user asks for the hospital's address, phone number, contact details, consultation options, or general details.
    """
    data = await fetch_live_hospital_data(force=False)
    if not data:
        return "Sorry, I am unable to fetch the hospital data right now."
    
    h_info = data.get("hospitalDataResponseDto", {})
    name = h_info.get("hospitalName", "Arora Hospital")
    address = h_info.get("address", "123 Main Street, Delhi")
    phone = h_info.get("phoneNo", "9876543210")
    consult_type = h_info.get("consultationType", "Both")
    
    depts = [d["name"] for d in data.get("departments", [])]
    services = [s["name"] for s in data.get("services", [])]
    
    facs = [
        f"Oxygen Support: {h_info.get('oxygenSupport', 'Yes')}",
        f"Ventilator Facility: {h_info.get('ventilatorFacility', 'Yes')}",
        f"Wheelchair/Stretcher Support: {h_info.get('wheelChairStretcher', 'No')}",
        f"Female Ward: {h_info.get('femaleWard', 'No')}"
    ]

    em_list = data.get("emergencyHourLists", [])
    em_lines = []
    for em in em_list:
        doc = em.get("doctorName", "Unknown")
        start = em.get("startTime", "Unknown")
        end = em.get("endTime", "Unknown")
        fee = em.get("extraFees", "0.0")
        em_lines.append(f"Dr. {doc} (Available: {start} to {end}, Extra Fee: Rs. {fee})")

    tpa_list = data.get("tpaList", [])
    tpas = [t.get("tpaName", "").strip() for t in tpa_list if t.get("tpaName")]

    info_str = (
        f"Hospital Name: {name}\n"
        f"Address: {address}\n"
        f"Phone Number: {phone}\n"
        f"Consultation Types Supported: {consult_type} (Online & Offline)\n"
        f"Available Departments: {', '.join(depts)}\n"
    )
    if facs:
        info_str += f"Available Facilities:\n" + "\n".join([f"- {f}" for f in facs]) + "\n"
    if em_lines:
        info_str += f"Emergency Hours & Fees: {', '.join(em_lines)}\n"
    if tpas:
        info_str += f"Supported Insurances / TPAs: {', '.join(tpas)}\n"
    if services:
        info_str += f"Services: {', '.join(services)}\n"
        
    return info_str

def check_emergency_rules(reason: str = "", patient_name: str = "", department: str = "", doctor_name: str = "") -> Optional[str]:
    combined = f"{reason} {patient_name} {department} {doctor_name}".lower()
    
    emergency_symptoms = [
        "accident", "head injury", "bleeding", "unconsciousness", "breathing issue", 
        "stroke", "neuro emergency", "heart attack", "labor pain", "trauma",
        "एक्सीडेंट", "दुर्घटना", "गंभीर चोट", "हार्ट अटैक", "दिल का दौरा", 
        "सांस लेने में तकलीफ", "बेहोशी", "बेहोश", "जानलेवा", "गंभीर रक्तस्राव", "प्रसव", "डिलिवरी"
    ]
    
    ambulance_words = ["ambulance", "एंबुलेंस"]
    is_ambulance_query = any(w in combined for w in ambulance_words)
    
    # Specific check for chest pain emergency indicators
    has_chest_pain = any(w in combined for w in ["chest pain", "सीने में दर्द", "seene me dard", "seene mein dard"])
    is_chest_pain_emergency = False
    if has_chest_pain:
        emergency_indicators = [
            "emergency", "severe", "can't breathe", "difficulty breathing",
            "unconscious", "breathing issue", "accident", "bleeding", "stroke",
            "heart attack", "shortness of breath", "gasping", "choking",
            "आपातकालीन", "गंभीर", "सांस", "बेहोश", "बेहोशी", "हार्ट अटैक", "दिल का दौरा", "जानलेवा"
        ]
        if any(ind in combined for ind in emergency_indicators):
            is_chest_pain_emergency = True
            
    is_emergency = any(w in combined for w in emergency_symptoms) or is_ambulance_query or is_chest_pain_emergency
    if is_emergency:
        import json
        from tools import load_cache_data
        data = load_cache_data() or {}
        h_info = data.get("hospitalDataResponseDto", {})
        
        ox = "YES" if str(h_info.get("oxygenSupport")).strip().upper() in ("YES", "Y") else "NO"
        vent = "YES" if str(h_info.get("ventilatorFacility")).strip().upper() in ("YES", "Y") else "NO"
        
        msg = f"Please come to the hospital Emergency Room immediately. Oxygen Support: {ox}. Ventilator Support: {vent}."
        
        # Suggest calling ambulance numbers
        maternal_words = ["pregnancy", "delivery", "maternal", "labor", "child healthcare", "गर्भावस्था", "प्रसव", "डिलिवरी", "prasav"]
        is_maternal = any(w in combined for w in maternal_words)
        
        if is_maternal:
            msg += "\nFor maternal and child healthcare transportation / pregnancy-related ambulance services, please dial 102 immediately."
        else:
            msg += "\nIf you need an emergency ambulance, please dial 108 immediately."
            
        # Determine required specialty
        is_neuro = any(w in combined for w in ["neuro", "stroke", "brain", "head injury", "seizure", "migraine", "headache", "nerv", "dizz", "बेहोश", "बेहोशी", "लकवा", "mirgi", "seizures", "behoshi", "behosh"])
        is_cardiac = any(w in combined for w in ["heart", "cardiac", "chest pain", "bp", "palpitations", "दिल", "सीने में दर्द", "seene me dard", "chest", "attack"])
        is_ortho = any(w in combined for w in ["ortho", "bone", "fracture", "joint", "knee", "back pain", "hand pain", "हड्डी", "जोड़", "कमर दर्द"])
        
        req_spec = None
        if is_cardiac:
            req_spec = "Cardiology"
        elif is_neuro:
            req_spec = "Neurologist"
        elif is_ortho:
            req_spec = "Orthopedics"
        elif any(w in combined for w in ["surgery", "surgeon", "operation", "appendix", "hernia", "ऑपरेशन", "ऑपरेट"]):
            req_spec = "Surgeon"
            
        em_list = data.get("emergencyHourLists", [])
        doctors_list = data.get("doctors", [])
        
        # Map doctor name to specialization
        doc_spec_map = {}
        for d in doctors_list:
            dname = d.get("doctorName") or d.get("name") or ""
            dname_clean = clean_doctor_name(dname).lower().strip()
            doc_spec_map[dname_clean] = d.get("specialization", "Unknown")
            
        matching_em_docs = []
        other_em_docs = []
        
        for em in em_list:
            doc_name = em.get("doctorName", "Unknown")
            doc_clean = clean_doctor_name(doc_name).lower().strip()
            spec = doc_spec_map.get(doc_clean, "Unknown").lower()
            
            matches = False
            if req_spec:
                req_lower = req_spec.lower()
                if req_lower == "cardiology" and ("cardio" in spec):
                    matches = True
                elif req_lower == "neurologist" and ("neuro" in spec):
                    matches = True
                elif req_lower == "orthopedics" and ("ortho" in spec):
                    matches = True
                elif req_lower == "surgeon" and ("surgeon" in spec or "surgery" in spec):
                    matches = True
                    
            em_detail = f"Dr. {doc_name} (Available: {em.get('startTime')} to {em.get('endTime')}, Specialization: {doc_spec_map.get(doc_clean, 'Unknown')}, Emergency Fee: Rs. {em.get('extraFees')})"
            if matches:
                matching_em_docs.append(em_detail)
            else:
                other_em_docs.append(em_detail)
                
        if em_list:
            if req_spec:
                if matching_em_docs:
                    msg += f"\nScheduled Emergency Doctor(s) for {req_spec}: {', '.join(matching_em_docs)}."
                else:
                    other_docs_names = ", ".join([d.split(" (")[0] for d in other_em_docs])
                    other_specs = ", ".join([doc_spec_map.get(clean_doctor_name(d.split(" (")[0]).lower().strip(), "Unknown") for d in other_em_docs])
                    msg += (
                        f"\nThe available emergency doctor ({other_docs_names}) belongs to a different specialty ({other_specs}), "
                        f"and the requested specialty's ({req_spec}) emergency availability cannot be confirmed from the available data."
                    )
            else:
                all_em_details = []
                for em in em_list:
                    doc_name = em.get("doctorName", "Unknown")
                    doc_clean = clean_doctor_name(doc_name).lower().strip()
                    spec = doc_spec_map.get(doc_clean, "Unknown")
                    all_em_details.append(
                        f"Dr. {doc_name} (Available: {em.get('startTime')} to {em.get('endTime')}, Specialization: {spec}, Emergency Fee: Rs. {em.get('extraFees')})"
                    )
                msg += f"\nScheduled Emergency Doctor(s): {', '.join(all_em_details)}."
        else:
            msg += "\nNo scheduled emergency doctor is available in records."
            
        msg += "\nFinal charges depend on hospital billing and treatment."
        return msg
        
    return None

def check_forbidden_combination(reason_str: str, dept_str: str) -> Optional[str]:
    if not reason_str or not dept_str:
        return None
    r = reason_str.lower()
    d = dept_str.lower()
    
    is_acne = any(w in r for w in ["acne", "skin", "rash"])
    is_migraine = any(w in r for w in ["migraine", "headache", "dizziness", "seizure"])
    is_back = any(w in r for w in ["back pain", "bone", "joint", "knee"])
    
    if is_acne and "neuro" in d:
        return "Error: Face acne is not consistent with Neurology."
    if is_migraine and "derm" in d:
        return "Error: Migraine is not consistent with Dermatology."
    if is_back and "neuro" in d:
        return "Error: Back pain is not consistent with Neurology."
    return None

def detect_multiple_departments(reason_str: str) -> bool:
    matched = set()
    r = reason_str.lower()
    if any(w in r for w in ["fever", "bukhar", "taap", "बुखार", "ताप"]):
        matched.add("General Physician")
    if any(w in r for w in ["migraine", "dizziness", "headache", "seizure", "paralysis", "numbness", "सिरदर्द", "चक्कर"]):
        matched.add("Neurologist")
    if any(w in r for w in ["asthma", "breathing", "lung", "दमा", "सांस"]):
        matched.add("Pulmonology")
    if any(w in r for w in ["tooth", "teeth", "dent", "daant", "दांत"]):
        matched.add("Dentistry")
    if any(w in r for w in ["chest", "palpitation", "heart", "blood pressure", "छाती", "दिल"]):
        matched.add("Cardiology")
    if any(w in r for w in ["hernia", "appendicitis", "surgery", "operation", "सर्जरी", "ऑपरेशन"]):
        matched.add("Surgeon")
    if any(w in r for w in ["fracture", "bone", "hand", "knee", "joint", "haddi", "हड्डी", "पीठ", "back pain"]):
        matched.add("Orthopedics")
    if any(w in r for w in ["acne", "rash", "skin", "dermat"]):
        matched.add("Dermatology")
    return len(matched) > 1

def validate_date(date_str: str) -> tuple[bool, Optional[str]]:
    if not date_str:
        return True, None
    
    date_str_clean = date_str.strip().lower()
    
    # Normalize month names to numbers
    months_map = {
        "january": 1, "jan": 1, "जनवरी": 1,
        "february": 2, "feb": 2, "फ़रवरी": 2, "फरवरी": 2,
        "march": 3, "mar": 3, "मार्च": 3,
        "april": 4, "apr": 4, "अप्रैल": 4,
        "may": 5, "मई": 5,
        "june": 6, "jun": 6, "जून": 6,
        "july": 7, "jul": 7, "जुलाई": 7,
        "august": 8, "aug": 8, "अगस्त": 8,
        "september": 9, "sep": 9, "सितंबर": 9, "सितम्बर": 9,
        "october": 10, "oct": 10, "अक्टूबर": 10, "अकटूबर": 10,
        "november": 11, "nov": 11, "नवंबर": 11, "नवम्बर": 11,
        "december": 12, "dec": 12, "दिसंबर": 12, "दिसम्बर": 12
    }
    
    year = _today_ist().year # default to current year
    month = None
    day = None
    
    match_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", date_str_clean)
    if match_iso:
        year = int(match_iso.group(1))
        month = int(match_iso.group(2))
        day = int(match_iso.group(3))
    else:
        match_indian = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", date_str_clean)
        if match_indian:
            day = int(match_indian.group(1))
            month = int(match_indian.group(2))
            year = int(match_indian.group(3))
        else:
            found_month = None
            for m_name, m_num in months_map.items():
                if m_name in date_str_clean:
                    found_month = m_num
                    break
            
            if found_month:
                month = found_month
                digits = re.findall(r"\d+", date_str_clean)
                if len(digits) == 1:
                    day = int(digits[0])
                elif len(digits) >= 2:
                    if len(digits[0]) == 4:
                        year = int(digits[0])
                        day = int(digits[1])
                    elif len(digits[1]) == 4:
                        day = int(digits[0])
                        year = int(digits[1])
                    else:
                        day = int(digits[0])
                        year = int(digits[1])
            else:
                try:
                    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
                        try:
                            dt = datetime.datetime.strptime(date_str, fmt)
                            year, month, day = dt.year, dt.month, dt.day
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass

    if month is None or day is None:
        return True, None

    if month < 1 or month > 12:
        return False, "कृपया कोई वैध महीना बताएं।"

    month_names_hindi = {
        1: "जनवरी", 2: "फ़रवरी", 3: "मार्च", 4: "अप्रैल",
        5: "मई", 6: "जून", 7: "जुलाई", 8: "अगस्त",
        9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर"
    }
    
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_in_months = {
        1: 31, 2: 29 if is_leap else 28, 3: 31, 4: 30,
        5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31,
        11: 30, 12: 31
    }

    max_days = days_in_months[month]
    if day < 1 or day > max_days:
        month_name = month_names_hindi[month]
        return False, f"{month_name} में केवल {max_days} दिन होते हैं। कृपया कोई वैध तारीख बताएं।"

    # Past date check — reject any date before today
    try:
        requested_date = datetime.date(year, month, day)
        if requested_date < _today_ist():
            return False, f"यह तारीख बीत चुकी है। कृपया आज या भविष्य की कोई तारीख बताएं।"
    except ValueError:
        pass

    return True, None


def validate_and_normalize_appointment_datetime(date_str: str, time_str: str) -> tuple[str, str, Optional[str]]:
    date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str.strip())
    if not date_match:
        return date_str, time_str, None
        
    year = int(date_match.group(1))
    month = date_match.group(2)
    day = date_match.group(3)
    
    time_clean = time_str.strip().upper()
    time_match = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)$", time_clean)
    if not time_match:
        time_match_simple = re.match(r"^(\d{1,2})\s*(AM|PM)$", time_clean)
        if time_match_simple:
            hour = int(time_match_simple.group(1))
            minute = 0
            am_pm = time_match_simple.group(2)
        else:
            time_match_24 = re.match(r"^(\d{1,2}):(\d{2})$", time_clean)
            if time_match_24:
                hour = int(time_match_24.group(1))
                minute = int(time_match_24.group(2))
                am_pm = None
            else:
                return date_str, time_str, "अपॉइंटमेंट का समय अमान्य है। कृपया एक निश्चित समय चुनें (जैसे 10:00 AM)।"
    else:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        am_pm = time_match.group(3)
        
    if am_pm == "PM" and hour < 12:
        hour += 12
    elif am_pm == "AM" and hour == 12:
        hour = 0
        
    try:
        appt_dt = datetime.datetime(year, int(month), int(day), hour, minute)
    except ValueError:
        return date_str, time_str, "तारीख या समय अमान्य है।"
        
    now = _now_ist()
    diff = (appt_dt - now).total_seconds()

    # POLICY (per system instructions): the ONLY time constraint is that the
    # requested slot must not be in the PAST. A future time — including later
    # today — is bookable. The previous 30-minute-buffer rule contradicted the
    # instructions and, combined with verify_availability's finders (which only
    # skip strictly-past slots), caused the agent to PROPOSE a same-day time and
    # then REJECT it at confirm — an endless "koi aur time?" loop (see transcript
    # where 1:30/2:00 were offered then refused). A 60-second grace absorbs
    # clock rounding so "abhi ka" (right now) still books.
    if diff < -60:
        return date_str, time_str, "यह समय बीत चुका है। कृपया अभी से आगे का कोई समय चुनें।"

    normalized_time = appt_dt.strftime("%I:%M %p").lstrip('0')
    normalized_date = appt_dt.strftime("%Y-%m-%d")
    return normalized_date, normalized_time, None


async def run_with_loading_announcement(coro, session, message: str):
    if session is None:
        return await coro

    async def announce_after_delay():
        await asyncio.sleep(0.8)
        try:
            logger.info("Announcing delay to user: %s", message)
            session.say(message, allow_interruptions=True)
        except Exception as e:
            logger.warning("Failed to say loading message: %s", e)

    announcer_task = asyncio.create_task(announce_after_delay())
    try:
        result = await coro
        return result
    finally:
        announcer_task.cancel()


async def get_consistent_doctors_for_dept(dept_name: str) -> list[dict]:
    data = await fetch_live_hospital_data()
    if not data or "doctors" not in data:
        return []
    matched = []
    for d in data["doctors"]:
        if d.get("department", "").lower().strip() == dept_name.lower().strip():
            if is_doctor_consistent_with_dept(d, dept_name):
                # Only include doctors who are currently available
                avail = d.get("available")
                if avail not in (False, "false", "no", "No", 0, "0", None, ""):
                    matched.append(d)
    return matched


async def find_earliest_available_slot(doctor_name: str, date_str: str) -> Optional[str]:
    slots = [
        "09:00 AM", "09:15 AM", "09:30 AM", "09:45 AM",
        "10:00 AM", "10:15 AM", "10:30 AM", "10:45 AM",
        "11:00 AM", "11:15 AM", "11:30 AM", "11:45 AM",
        "12:00 PM", "12:15 PM", "12:30 PM", "12:45 PM",
        "01:00 PM", "01:15 PM", "01:30 PM", "01:45 PM",
        "02:00 PM", "02:15 PM", "02:30 PM", "02:45 PM",
        "03:00 PM", "03:15 PM", "03:30 PM", "03:45 PM",
        "04:00 PM", "04:15 PM", "04:30 PM", "04:45 PM",
        "05:00 PM", "05:15 PM", "05:30 PM", "05:45 PM",
        "06:00 PM", "06:15 PM", "06:30 PM", "06:45 PM",
        "07:00 PM"
    ]
    import datetime
    from storage import is_slot_locked
    for slot in slots:
        # Enforce Rule 8: Same-day booking must be at least 30 minutes in future
        try:
            today_date = _today_ist()
            parsed_date = datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
            if parsed_date == today_date:
                now_str = _now_ist().strftime("%I:%M %p")
                if time_to_minutes(slot) < time_to_minutes(now_str):
                    continue
        except Exception:
            if date_str == _today_ist().strftime("%Y-%m-%d"):
                now_str = _now_ist().strftime("%I:%M %p")
                if time_to_minutes(slot) < time_to_minutes(now_str):
                    continue

        if not await is_slot_locked(doctor_name, date_str, slot):
            return slot
    return None

async def find_chronological_earliest_slot(doctor_name: str, max_days: int = 30) -> tuple[Optional[str], Optional[str]]:
    """Finds the earliest available slot for a given doctor starting from today."""
    slots = [
        "09:00 AM", "09:15 AM", "09:30 AM", "09:45 AM",
        "10:00 AM", "10:15 AM", "10:30 AM", "10:45 AM",
        "11:00 AM", "11:15 AM", "11:30 AM", "11:45 AM",
        "12:00 PM", "12:15 PM", "12:30 PM", "12:45 PM",
        "01:00 PM", "01:15 PM", "01:30 PM", "01:45 PM",
        "02:00 PM", "02:15 PM", "02:30 PM", "02:45 PM",
        "03:00 PM", "03:15 PM", "03:30 PM", "03:45 PM",
        "04:00 PM", "04:15 PM", "04:30 PM", "04:45 PM",
        "05:00 PM", "05:15 PM", "05:30 PM", "05:45 PM",
        "06:00 PM", "06:15 PM", "06:30 PM", "06:45 PM",
        "07:00 PM"
    ]
    import datetime
    from storage import is_slot_locked

    today_date = _today_ist()
    
    for day_offset in range(max_days):
        check_date = today_date + datetime.timedelta(days=day_offset)
        date_str = check_date.strftime("%Y-%m-%d")
        
        for slot in slots:
            if day_offset == 0:
                now_str = _now_ist().strftime("%I:%M %p")
                if time_to_minutes(slot) < time_to_minutes(now_str):
                    continue
            
            try:
                locked = await is_slot_locked(doctor_name, date_str, slot)
                if not locked:
                    return date_str, slot
            except Exception as e:
                logger.error(f"Redis lookup failed for {doctor_name} on {date_str} {slot}: {e}")
                raise e

    return None, None


async def filter_available_doctors(docs: list[dict], date_str: str, time_str: str) -> list[dict]:
    """
    Filter a list of doctors to only those whose slot is NOT locked/booked at the given date and time.
    Returns the full list unchanged if date or time is unknown (so we don't incorrectly restrict options).
    """
    if not date_str or not time_str:
        return docs
    time_flex = time_str.lower().strip() in [
        "any", "sny", "any time", "anytime", "flexible", "earliest", "earliest available", "any slot", "कोई भी", "koi bhi", "sny time", "snytime", "कोई भी समय", "जब मिले", "किसी भी टाइम", "koi bhi samay", "jab mile", "kisi bhi time"
    ]
    if time_flex:
        return docs
    from storage import is_slot_locked
    available = []
    for doc in docs:
        d_name = f"Dr. {clean_doctor_name(doc.get('doctorName') or doc.get('name') or '')}"
        if not await is_slot_locked(d_name, date_str, time_str):
            available.append(doc)
    return available


async def find_earliest_available_doctor(docs: list[dict], date_str: str, time_str: str = "") -> Optional[dict]:

    if not docs:
        return None
    if not date_str:
        return docs[0]
        
    from storage import is_slot_locked
    
    is_flex = False
    if time_str:
        t_clean = time_str.lower().strip()
        if t_clean in ["any", "sny", "any time", "anytime", "flexible", "earliest", "earliest available", "any slot", "कोई भी", "koi bhi", "sny time", "snytime", "कोई भी समय", "जब मिले", "किसी भी टाइम", "koi bhi samay", "jab mile", "kisi bhi time", ""]:
            is_flex = True
            
    if time_str and not is_flex:
        norm_time = normalize_time_slot(time_str)
        import datetime
        try:
            today_date = _today_ist()
            parsed_date = datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
            if parsed_date == today_date:
                now_str = _now_ist().strftime("%I:%M %p")
                if time_to_minutes(norm_time) < time_to_minutes(now_str):
                    return None
        except Exception:
            if date_str == _today_ist().strftime("%Y-%m-%d"):
                now_str = _now_ist().strftime("%I:%M %p")
                if time_to_minutes(norm_time) < time_to_minutes(now_str):
                    return None

        for doc in docs:
            d_name = f"Dr. {clean_doctor_name(doc['name'])}"
            if not await is_slot_locked(d_name, date_str, norm_time):
                return doc
        return None

    slots = [
        "09:00 AM", "09:15 AM", "09:30 AM", "09:45 AM",
        "10:00 AM", "10:15 AM", "10:30 AM", "10:45 AM",
        "11:00 AM", "11:15 AM", "11:30 AM", "11:45 AM",
        "12:00 PM", "12:15 PM", "12:30 PM", "12:45 PM",
        "01:00 PM", "01:15 PM", "01:30 PM", "01:45 PM",
        "02:00 PM", "02:15 PM", "02:30 PM", "02:45 PM",
        "03:00 PM", "03:15 PM", "03:30 PM", "03:45 PM",
        "04:00 PM", "04:15 PM", "04:30 PM", "04:45 PM",
        "05:00 PM", "05:15 PM", "05:30 PM", "05:45 PM",
        "06:00 PM", "06:15 PM", "06:30 PM", "06:45 PM",
        "07:00 PM"
    ]
    for slot in slots:
        # Enforce Rule 8: Same-day booking must be at least 30 minutes in future
        try:
            today_date = _today_ist()
            parsed_date = datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
            if parsed_date == today_date:
                now_str = _now_ist().strftime("%I:%M %p")
                if time_to_minutes(slot) < time_to_minutes(now_str):
                    continue
        except Exception:
            if date_str == _today_ist().strftime("%Y-%m-%d"):
                now_str = _now_ist().strftime("%I:%M %p")
                if time_to_minutes(slot) < time_to_minutes(now_str):
                    continue

        for doc in docs:
            d_name = f"Dr. {clean_doctor_name(doc['name'])}"
            if not await is_slot_locked(d_name, date_str, slot):
                return doc
    return None


async def get_alternatives(doctor_name: str, date_str: str, time_slot: str, department: str) -> str:
    next_time = None
    slots = [
        "09:00 AM", "09:15 AM", "09:30 AM", "09:45 AM",
        "10:00 AM", "10:15 AM", "10:30 AM", "10:45 AM",
        "11:00 AM", "11:15 AM", "11:30 AM", "11:45 AM",
        "12:00 PM", "12:15 PM", "12:30 PM", "12:45 PM",
        "01:00 PM", "01:15 PM", "01:30 PM", "01:45 PM",
        "02:00 PM", "02:15 PM", "02:30 PM", "02:45 PM",
        "03:00 PM", "03:15 PM", "03:30 PM", "03:45 PM",
        "04:00 PM", "04:15 PM", "04:30 PM", "04:45 PM",
        "05:00 PM", "05:15 PM", "05:30 PM", "05:45 PM",
        "06:00 PM", "06:15 PM", "06:30 PM", "06:45 PM",
        "07:00 PM"
    ]
    from storage import is_slot_locked
    for slot in slots:
        if slot.lower().strip() != time_slot.lower().strip():
            # Enforce Rule 8: Same-day booking must be at least 30 minutes in future
            try:
                today_date = _today_ist()
                parsed_date = datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
                if parsed_date == today_date:
                    now_str = _now_ist().strftime("%I:%M %p")
                    if time_to_minutes(slot) < time_to_minutes(now_str):
                        continue
            except Exception:
                if date_str == _today_ist().strftime("%Y-%m-%d"):
                    now_str = _now_ist().strftime("%I:%M %p")
                    if time_to_minutes(slot) < time_to_minutes(now_str):
                        continue
            if not await is_slot_locked(doctor_name, date_str, slot):
                next_time = slot
                break

    next_date_str = None
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        next_dt = dt + datetime.timedelta(days=1)
        next_date_str = next_dt.strftime("%Y-%m-%d")
    except Exception:
        pass
        
    next_date_time = None
    if next_date_str:
        for slot in slots:
            if not await is_slot_locked(doctor_name, next_date_str, slot):
                next_date_time = f"{next_date_str} at {slot}"
                break

    alt_doctor = None
    if department:
        docs = await get_consistent_doctors_for_dept(department)
        for doc in docs:
            d_name = f"Dr. {clean_doctor_name(doc['name'])}"
            if d_name.lower().strip() != doctor_name.lower().strip():
                # Enforce Rule 8: Same-day booking must be at least 30 minutes in future
                try:
                    today_date = _today_ist()
                    parsed_date = datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
                    if parsed_date == today_date:
                        now_str = _now_ist().strftime("%I:%M %p")
                        if time_to_minutes(time_slot) < time_to_minutes(now_str):
                            continue
                except Exception:
                    if date_str == _today_ist().strftime("%Y-%m-%d"):
                        now_str = _now_ist().strftime("%I:%M %p")
                        if time_to_minutes(time_slot) < time_to_minutes(now_str):
                            continue
                if not await is_slot_locked(d_name, date_str, time_slot):
                    alt_doctor = d_name
                    break
        if not alt_doctor and docs:
            for doc in docs:
                d_name = f"Dr. {clean_doctor_name(doc['name'])}"
                if d_name.lower().strip() != doctor_name.lower().strip():
                    alt_doctor = d_name
                    break

    parts = []
    if next_time:
        parts.append(f"उसी दिन का अगला उपलब्ध समय: {next_time}")
    if next_date_time:
        parts.append(f"अगली उपलब्ध तारीख और समय: {next_date_time}")
    if alt_doctor:
        parts.append(f"वैकल्पिक डॉक्टर: {alt_doctor}")

    if parts:
        return "यह स्लॉट पहले से ही बुक है। हमारे पास ये विकल्प उपलब्ध हैं: " + ", ".join(parts)
    return "यह स्लॉट पहले से ही बुक है। कृपया कोई अन्य तारीख या समय चुनें।"


async def get_first_consistent_doctor(dept_name: str) -> Optional[str]:
    docs = await get_consistent_doctors_for_dept(dept_name)
    if docs:
        return f"Dr. {clean_doctor_name(docs[0]['name'])}"
    return None

@function_tool
async def check_available_slots(
    ctx: RunContext,
    appointment_date: Annotated[str, "The date in YYYY-MM-DD format. E.g., 2026-07-04"],
    doctor_name: Annotated[str, "The specific doctor's name, if known"] = "",
    department: Annotated[str, "The department name, if doctor is not specific"] = ""
) -> str:
    """
    Check and return all available time slots on a specific date for a doctor or department.
    Use this ONLY when the user explicitly asks 'what slots are available'.
    CRITICAL: Before or alongside calling this tool, you MUST output a conversational text message (e.g., 'Ek moment, main check kar rahi hoon.') so the user is not left in silence! Never emit an empty text message when calling this tool.
    """
    if not appointment_date:
        return "Error: appointment_date is required."

    # Past date guard — never return slots for past dates
    try:
        import datetime as _dt
        _parsed = _dt.datetime.strptime(appointment_date.strip(), "%Y-%m-%d").date()
        if _parsed < _today_ist():
            return (
                f"Error: The date {appointment_date} is in the past. "
                f"Please ask the user to provide today's date or a future date."
            )
    except Exception:
        pass
        
    target_docs = []
    if doctor_name:
        res = await find_doctor_by_name(doctor_name)
        if res["status"] == "single_match":
            target_docs.append(res["match"])
        else:
            return f"Error: Doctor {doctor_name} not found or ambiguous."
    elif department:
        target_docs = await get_consistent_doctors_for_dept(department)
        if not target_docs:
            return f"Error: No doctors found in {department}."
    else:
        state = ctx.userdata.get("state")
        if state:
            if state.doctor_name:
                res = await find_doctor_by_name(state.doctor_name)
                if res["status"] == "single_match":
                    target_docs.append(res["match"])
            elif state.department:
                target_docs = await get_consistent_doctors_for_dept(state.department)
    
    if not target_docs:
        return "Error: Please specify either a doctor name or department."
        
    slots = [
        "09:00 AM", "09:15 AM", "09:30 AM", "09:45 AM",
        "10:00 AM", "10:15 AM", "10:30 AM", "10:45 AM",
        "11:00 AM", "11:15 AM", "11:30 AM", "11:45 AM",
        "12:00 PM", "12:15 PM", "12:30 PM", "12:45 PM",
        "01:00 PM", "01:15 PM", "01:30 PM", "01:45 PM",
        "02:00 PM", "02:15 PM", "02:30 PM", "02:45 PM",
        "03:00 PM", "03:15 PM", "03:30 PM", "03:45 PM",
        "04:00 PM", "04:15 PM", "04:30 PM", "04:45 PM",
        "05:00 PM", "05:15 PM", "05:30 PM", "05:45 PM",
        "06:00 PM", "06:15 PM", "06:30 PM", "06:45 PM",
        "07:00 PM"
    ]
    
    import datetime
    import asyncio
    from storage import is_slot_locked
    
    available_slots = set()
    for doc in target_docs:
        d_name = f"Dr. {clean_doctor_name(doc.get('name') or doc.get('doctorName') or '')}"
        
        slots_to_check = []
        for slot in slots:
            skip = False
            try:
                today_date = _today_ist()
                parsed_date = datetime.datetime.strptime(appointment_date.strip(), "%Y-%m-%d").date()
                if parsed_date == today_date:
                    now_str = _now_ist().strftime("%I:%M %p")
                    if time_to_minutes(slot) < time_to_minutes(now_str):
                        skip = True
            except Exception:
                if appointment_date == _today_ist().strftime("%Y-%m-%d"):
                    now_str = _now_ist().strftime("%I:%M %p")
                    if time_to_minutes(slot) < time_to_minutes(now_str):
                        skip = True
            
            if not skip:
                slots_to_check.append(slot)
                
        async def _check(s):
            is_l = await is_slot_locked(d_name, appointment_date, s)
            return s, is_l
            
        tasks = [_check(s) for s in slots_to_check]
        if tasks:
            results = await asyncio.gather(*tasks)
            for s, is_l in results:
                if not is_l:
                    available_slots.add(s)
                
    if not available_slots:
        return f"No available slots found for {appointment_date}."
        
    # Sort slots chronologically
    sorted_slots = sorted(list(available_slots), key=lambda s: time_to_minutes(s))
    return f"Available slots on {appointment_date}: {', '.join(sorted_slots)}"


@function_tool
async def update_appointment_details(
    ctx: RunContext,
    caller_name: Annotated[str, "Caller's full name. Provide ONLY when collecting, updating, or correcting the caller's name."] = "",
    patient_name: Annotated[str, "Patient's full name. Provide ONLY when collecting, updating, or correcting the patient's name."] = "",
    patient_phone: Annotated[str, "Patient's 10-digit mobile number. Provide ONLY when collecting, updating, or correcting the patient's phone number."] = "",
    patient_name_spelling: Annotated[str, "Spelling of patient's name (e.g. 'S-I-Y-A'). Leave empty unless correcting spelling."] = "",
    confirm_patient_name: Annotated[bool, "Set to True ONLY when the caller explicitly confirms that the spelling of the name is correct (e.g. says 'Yes', 'Haan', 'Sahi hai')."] = False,
    force_update_patient_name: Annotated[bool, "Set to True to force updating the name even if it was previously confirmed/locked."] = False,
    confirm_patient_phone: Annotated[bool, "Set to True ONLY when the caller explicitly confirms that the read-back phone number is correct (e.g. says 'Yes', 'Haan', 'Sahi hai')."] = False,
    department: Annotated[str, "The department name (e.g., Cardiology, Neurologist, Orthopedics, Surgeon)."] = "",
    doctor_name: Annotated[str, "The doctor's name, or 'any' / 'any doctor'."] = "",
    appointment_date: Annotated[str, "The appointment date in YYYY-MM-DD format."] = "",
    appointment_time: Annotated[str, "The appointment time slot."] = "",
    time_preference: Annotated[str, "Preference for time, e.g. 'FLEXIBLE'."] = "",
    reason: Annotated[str, "The patient's symptom or reason for visit."] = "",
    use_caller_phone: Annotated[bool, "Set to True whenever the caller means 'use the number I'm calling from' for the booking phone number — in whatever words they use (e.g. 'same number', 'yahi number', 'yhi number hai', 'isi number pe', 'this is the correct number', 'use this one'). Judge by meaning, not by matching an exact phrase."] = False,
    ignore_mismatch: Annotated[bool, "Set to True to ignore any department/symptom mismatch."] = False,
    mismatch_acknowledged: Annotated[bool, "Set to True if user acknowledged department/symptom mismatch."] = False,
    book_another_appointment: Annotated[bool, "Set to True to reset status and draft a new appointment."] = False,
    user_confirmed_booking: Annotated[bool, "Set to True ONLY after: (1) you called get_booking_summary and read it to the caller, (2) the caller confirmed the summary is correct, and (3) the caller clearly said yes to going ahead and booking it (in their own words, any language). This is required before confirm_appointment will succeed."] = False,
    relation: Annotated[str, "Patient's relation to the caller (e.g. Self, Mother, Father, Child, etc.)."] = "",
    gender: Annotated[str, "Patient's gender."] = "",
    age: Annotated[str, "Patient's age."] = "",
    additional_notes: Annotated[str, "Any additional notes/preferences shared by the user."] = "",
    active_patient_key: Annotated[str, "Switch active booking to this patient key (name or relation)."] = "",
    total_appointments_requested: Annotated[int, "Total number of appointments requested by the user."] = 0,
    completed_appointments_count: Annotated[int, "Number of completed appointments so far."] = 0,
    appointment_for_self: Annotated[bool, "Set to True ONLY when the caller explicitly says the appointment is for themselves (e.g. 'mere liye', 'apne liye', 'mujhe book karo', 'for myself'). This will automatically use the caller's name as the patient name."] = False,
) -> str:
    """
    Update the current state of the appointment being booked.

    CRITICAL RULES FOR CALLING THIS TOOL:
    - ONLY call with confirm_patient_phone=True when the caller explicitly confirms the phone number. Do NOT pass the phone number string again.
    - ONLY call with confirm_patient_name=True when the caller explicitly confirms the spelling of their name. Do NOT pass the name string again.
    - Only call this tool when you have CLEAR, CONFIRMED information from the caller.
    - NEVER call this tool with a department inferred from unclear or non-medical speech.
    - NEVER call this tool with a department unless the caller explicitly said a department name OR
      mentioned a clear medical symptom (e.g., 'ghutne mein dard' → Orthopedics, 'seene mein dard' → Cardiology).
    - If the caller's speech is ambiguous, garbled, or unrelated to medical conditions, ask for clarification instead.
    - Always collect patient_name BEFORE moving to doctor/date/time selection.
    """
    state: AppointmentState = ctx.userdata["state"]

    _prev_locked_doctor = state.doctor_name if state.availability_verified else None
    _prev_locked_date = state.appointment_date if state.availability_verified else None
    _prev_locked_time = state.appointment_time if state.availability_verified else None
    
    # Initialize active profile key if not set
    if not state.active_profile_key:
        if state.relation:
            state.active_profile_key = state.relation.lower().strip()
        elif state.patient_name:
            state.active_profile_key = state.patient_name.lower().strip()
        else:
            state.active_profile_key = "self"

    # Save current profile first
    save_current_profile_to_map(state)

    # Determine if we need to switch active profile
    target_key = None
    if active_patient_key:
        target_key = active_patient_key.lower().strip()
    elif relation:
        target_key = relation.lower().strip()
    elif patient_name:
        p_name_clean = clean_patient_name(patient_name).lower().strip()
        for pk, prof in state.patient_profiles.items():
            prof_name = clean_patient_name(prof.get("patient_name") or "").lower().strip()
            if prof_name == p_name_clean:
                target_key = pk
                break
        if not target_key and book_another_appointment:
            target_key = p_name_clean

    if book_another_appointment and not target_key:
        target_key = f"draft_{int(time.time())}"

    # Perform switch if target_key is different and valid
    if target_key and target_key != state.active_profile_key:
        load_profile_from_map(state, target_key)

    # Set new fields in active state
    if caller_name:
        state.caller_name = caller_name
        # If appointment is for the caller themselves, also set patient_name immediately
        if (appointment_for_self or (relation or "").lower().strip() == "self") and not state.patient_name:
            state.patient_name = caller_name
            state.patient_name_locked = False
    # If appointment_for_self is True and we already have caller_name stored, copy it now
    if appointment_for_self and state.caller_name and not state.patient_name:
        state.patient_name = state.caller_name
        state.patient_name_locked = False
    if relation:
        state.relation = relation
    if gender:
        state.gender = gender
    if age:
        state.age = age
    if additional_notes:
        if state.additional_notes:
            state.additional_notes = f"{state.additional_notes} | {additional_notes}"
        else:
            state.additional_notes = additional_notes
    if total_appointments_requested:
        state.total_appointments_requested = total_appointments_requested
    if completed_appointments_count:
        state.completed_appointments_count = completed_appointments_count

    transcript = _get_transcript(ctx.userdata)
    session = ctx.userdata.get("session")

    # Check for emergency using both current inputs and stored state fields
    if any([reason, patient_name, department, doctor_name]):
        emergency_msg = check_emergency_rules(
            reason=reason or "",
            patient_name=patient_name or "",
            department=department or "",
            doctor_name=doctor_name or "",
        )
        if emergency_msg:
            return emergency_msg

    def check_loop(step: str):
        state.increment_step_attempt(step)
        if state.is_step_limit_exceeded(step, limit=3):
            logger.warning(f"LOOP DETECTED: Step '{step}' attempt count exceeded limit.")
            return f"Error: The user has provided an invalid or unavailable {step} 3 times. Do NOT transfer the call. Ask them if they would like to choose a different option or if they need help."
        return None

    # Reset state if book_another_appointment is requested, or if a different patient name
    # is provided for an already booked/confirmed appointment.
    should_reset = book_another_appointment
    if not should_reset and state.booking_status in ("BOOKED", "confirmed"):
        # If any significant booking field is being updated, we must reset the completed state
        # to start a new booking draft.
        if any([patient_name, patient_phone, patient_name_spelling, department, doctor_name, appointment_date, appointment_time, reason]):
            should_reset = True

    if should_reset:
        save_current_profile_to_map(state)
        # Archive the booked profile under a unique key
        if state.booking_status == "BOOKED" or state.booking_status == "confirmed":
            archive_key = f"booked_{state.active_profile_key}_{state.appointment_id or int(time.time())}"
            state.patient_profiles[archive_key] = state.patient_profiles[state.active_profile_key].copy()
            state.patient_profiles[archive_key]["booking_status"] = "BOOKED"
        
        # Load a fresh profile for the active key
        new_key = state.active_profile_key
        load_profile_from_map(state, new_key)
        
        # Reset state fields
        state.appointment_id = None
        state.patient_name = None
        state.patient_name_locked = False
        state.patient_name_spelled = None
        state.name_confirmation_attempts = 0
        state.doctor_name = None
        state.doctor_id = None
        state.appointment_date = None
        state.appointment_time = None
        state.reason = None
        state.department = None
        state.time_preference = None
        state.booking_status = "DRAFT"
        state.booking_intent_detected = False
        state.availability_verified = False
        state.mismatch_acknowledged = False
        state.ask_counts = {}
        state.skipped_fields = []
        state.step_attempts = {}
        
        # Reset phone number for duplicate patient protection (Rules 1 & 5)
        state.patient_phone = None
        state.patient_phone_confirmed = False
        
        save_current_profile_to_map(state)
    

    
    # 1. Booking intent detection
    if any([patient_name, patient_phone, department, doctor_name, appointment_date, appointment_time, reason]):
        state.booking_intent_detected = True

    # Infer default reason if doctor or department is specified (Rule 3)
    if not reason and not state.reason:
        target_dept_for_reason = department or state.department
        target_doc_for_reason = doctor_name or state.doctor_name
        if target_dept_for_reason:
            state.reason = f"{target_dept_for_reason} consultation"
        elif target_doc_for_reason:
            state.reason = f"Consultation with {target_doc_for_reason}"

    # Initialize pending response buffers
    pending_phone_response = None
    pending_name_response = None

    # 2. Caller phone — only used when the patient EXPLICITLY says their number is the same as the calling number.
    # The LLM must NEVER pass use_caller_phone=True on its own; only when the user says
    # something like 'same number', 'usi number pe', 'is number pe', etc.
    if use_caller_phone:
        loop_err = check_loop("patient_phone")
        if loop_err: return loop_err
        caller_phone_raw = ctx.userdata.get("caller_phone", "")
        if not caller_phone_raw and ctx.room and ctx.room.name:
            matches = re.findall(r"\d{10,12}", ctx.room.name)
            if matches:
                caller_phone_raw = matches[0]
        if caller_phone_raw:
            clean_phone = re.sub(r"\D", "", caller_phone_raw)
            if len(clean_phone) > 10 and clean_phone.startswith("91"):
                clean_phone = clean_phone[2:]
            if len(clean_phone) == 10 and clean_phone[0] in "6789":
                state.patient_phone = clean_phone
                state.patient_phone_confirmed = True
                state.phone_confirmation_attempts = 0
                state.step_attempts["patient_phone"] = 0
                pending_phone_response = None
            else:
                return "Error: Caller phone number is not a valid 10-digit Indian mobile number. Please ask the patient to say their number clearly."
        else:
            return "Error: No caller phone number available in this session. Please ask the patient to provide their contact number."

    if confirm_patient_phone:
        loop_err = check_loop("patient_phone")
        if loop_err: return loop_err
        state.patient_phone_confirmed = True
        state.phone_confirmation_attempts = 0
        state.step_attempts["patient_phone"] = 0
        pending_phone_response = None

    # 3. Patient phone validation
    if patient_phone:
        caller_phone_raw = ctx.userdata.get("caller_phone", "")
        clean_phone = normalize_and_align_phone_number(patient_phone, caller_phone_raw)
        
        if not (clean_phone.isdigit() and len(clean_phone) == 10 and clean_phone[0] in "6789"):
            state.patient_phone = None  # Revert invalid/partial phone capture
            state.patient_phone_confirmed = False
            state.phone_confirmation_attempts = 0
            # Be explicit: the model must RE-ASK now and must not move on to the
            # next step with a missing/invalid number (the cause of the transcript
            # loop, where an unparsed number was silently skipped past).
            return (f"'{patient_phone}' is not a complete valid 10-digit Indian mobile number. "
                    "Ask the caller to repeat their full 10-digit mobile number slowly, digit by digit, "
                    "and do NOT continue to any other step until a valid number is captured.")
            
        state.patient_phone = clean_phone
        state.patient_phone_confirmed = True
        state.phone_confirmation_attempts = 0
        state.step_attempts["patient_phone"] = 0
        pending_phone_response = None

    # 4. Patient name capture / confirmation
    if patient_name or patient_name_spelling or confirm_patient_name:
        cleaned_name = clean_patient_name(patient_name) if patient_name else ""
        if cleaned_name:
            state.patient_name = cleaned_name
        elif patient_name_spelling:
            cleaned_spelling = patient_name_spelling.upper().replace(" ", "").replace("-", "")
            state.patient_name = cleaned_spelling.replace("-", "").capitalize()
            
        state.patient_name_locked = True
        state.name_confirmation_attempts = 0
        state.step_attempts["patient_name"] = 0
        
        if state.patient_name:
            transcript.patient_name = state.patient_name
            first_word = state.patient_name.split()[0]
            state.patient_name_spelled = "-".join(list(first_word.upper()))
            
        pending_name_response = None

    # 5. Date validation
    if appointment_date:
        if not state.appointment_date or appointment_date.strip() != state.appointment_date.strip():
            loop_err = check_loop("appointment_date")
            if loop_err: return loop_err
        is_valid, err_msg = validate_date(appointment_date)
        if not is_valid:
            return f"Error: {err_msg}"

    # 6. Future time validation (at least 15 minutes in future)
    target_date = appointment_date or state.appointment_date
    target_time = appointment_time or state.appointment_time
    if (appointment_date or appointment_time) and target_date and target_time:
        is_target_time_flex = target_time.lower().strip() in ["any", "sny", "any time", "anytime", "flexible", "earliest", "earliest available", "any slot", "कोई भी", "koi bhi", "sny time", "snytime", "कोई भी समय", "जब मिले", "किसी भी टाइम", "koi bhi samay", "jab mile", "kisi bhi time", "first available", "first available slot"]
        if not is_target_time_flex:
            res_date, res_time, err = validate_and_normalize_appointment_datetime(target_date, target_time)
            if err:
                return f"Error: {err}"
            if appointment_date:
                appointment_date = res_date
            if appointment_time:
                appointment_time = res_time
            target_date = appointment_date or state.appointment_date
            target_time = appointment_time or state.appointment_time

    # If the user previously selected "any doctor", and we are updating date or time, recheck and select an available doctor for the new slot!
    if (appointment_date or appointment_time) and getattr(state, "doctor_preference", None) == "any" and not doctor_name:
        target_dept = department or state.department
        if target_dept:
            docs = await get_consistent_doctors_for_dept(target_dept)
            if docs:
                chosen_doc = await find_earliest_available_doctor(docs, target_date, target_time)
                if chosen_doc:
                    state.doctor_name = chosen_doc.get("doctorName") or chosen_doc.get("name")
                    state.doctor_id = chosen_doc.get("doctorId")

    # 7. Doctor validation & matching
    matched_doctor_obj = None
    doc_msg = ""
    if doctor_name:
        doc_cleaned = clean_doctor_name(doctor_name).lower().strip()
        state_doc_cleaned = clean_doctor_name(state.doctor_name or "").lower().strip()
        is_any_doc = doctor_name.lower().strip() in [
            "any", "any doctor", "any available doctor", "anyone is fine", "first available", "earliest available",
            "कोई भी", "कोई भी डॉक्टर", "कोई भी डॉ", "कोई भी चलेगा", "कोई भी डॉक्टर चलेगा",
            "koi bhi", "koi bhi doctor", "anyone", "flexible", "first available doctor", "earliest available doctor"
        ]
        if is_any_doc or not state.doctor_name or doc_cleaned != state_doc_cleaned:
            loop_err = check_loop("doctor_id")
            if loop_err: return loop_err

        is_any_doc = doctor_name.lower().strip() in [
            "any", "any doctor", "any available doctor", "anyone is fine", "first available", "earliest available",
            "कोई भी", "कोई भी डॉक्टर", "कोई भी डॉ", "कोई भी चलेगा", "कोई भी डॉक्टर चलेगा",
            "koi bhi", "koi bhi doctor", "anyone", "flexible", "first available doctor", "earliest available doctor"
        ]
        if is_any_doc:
            state.doctor_preference = "any"
            target_dept = department or state.department
            if not target_dept:
                return "Error: Please specify the department first to choose any doctor."
            docs = await run_with_loading_announcement(
                get_consistent_doctors_for_dept(target_dept),
                session,
                "धन्यवाद प्रतीक्षा के लिए, मैं जानकारी देख रही हूँ।"
            )
            if not docs:
                return f"Error: No doctors available in the {target_dept} department."
            
            chosen_doc = await find_earliest_available_doctor(docs, target_date, target_time)
            if not chosen_doc:
                if target_time:
                    return f"Error: No doctors in the {target_dept} department have any available slots on {target_date} at {target_time}."
                return f"Error: No doctors in the {target_dept} department have any available slots on {target_date}."
            state.doctor_name = chosen_doc.get("doctorName") or chosen_doc.get("name")
            state.doctor_id = chosen_doc.get("doctorId")
            doc_msg = f"I have assigned the earliest available doctor: {state.doctor_name}."
        else:
            state.doctor_preference = "specific"
            res = await run_with_loading_announcement(
                find_doctor_by_name(doctor_name),
                session,
                "धन्यवाद प्रतीक्षा के लिए, मैं जानकारी देख रही हूँ।"
            )
            if res["status"] == "single_match":
                matched_doctor_obj = res["match"]
                target_dept = department or matched_doctor_obj.get("department")
                if target_dept and not is_doctor_consistent_with_dept(matched_doctor_obj, target_dept):
                    return f"Error: Doctor 'Dr. {clean_doctor_name(matched_doctor_obj['name'])}' is inconsistent with the {target_dept} department (due to qualification or specialization mismatch)."
                
                state.doctor_name = matched_doctor_obj.get("doctorName") or matched_doctor_obj.get("name")
                state.doctor_id = matched_doctor_obj.get("doctorId")
                state.department = matched_doctor_obj.get("department")
            elif res["status"] == "multiple_matches":
                return f"Error: Multiple doctors found matching '{doctor_name}'. Please be more specific."
            else:
                return f"Error: Doctor '{doctor_name}' was not found. Please clarify the doctor's name."
        state.step_attempts["doctor_id"] = 0

    # 8. Department validation
    if department:
        norm_dept = normalize_department(department)
        if not state.department or (norm_dept and norm_dept.lower() != state.department.lower()):
            loop_err = check_loop("department")
            if loop_err: return loop_err
        
        # Check if it maps to an unsupported specialty
        unsupported_depts = ["Dentistry", "ENT", "Dermatology", "Pulmonology", "Gastroenterology", "Urology", "Nephrology"]
        if norm_dept in unsupported_depts:
            hindi_names = {
                "Dentistry": "दांतों का",
                "ENT": "नाक, कान और गले का (ENT)",
                "Dermatology": "त्वचा (Dermatology)",
                "General Physician": "सामान्य रोग (General Physician)",
                "Pulmonology": "फेफड़ों का (Pulmonology)",
                "Gastroenterology": "पेट का (Gastroenterology)",
                "Urology": "मूत्र रोग (Urology)",
                "Nephrology": "किडनी (Nephrology)"
            }
            dept_name_hindi = hindi_names.get(norm_dept, norm_dept)
            return f"माफ़ कीजिएगा, लेकिन हमारे यहाँ {dept_name_hindi} विभाग उपलब्ध नहीं है, इसलिए मैं इस विभाग के लिए अपॉइंटमेंट बुक नहीं कर सकती।"
            
        valid_depts = ["Cardiology", "Neurologist", "Orthopedics", "Surgeon", "General Physician"]
        matched_dept = None
        if norm_dept:
            for vd in valid_depts:
                if norm_dept.lower() == vd.lower():
                    matched_dept = vd
                    break
        
        if not matched_dept:
            choices_str = ", ".join(valid_depts)
            return f"Error: The department '{department}' is not recognized. Recognized departments are: {choices_str}."

        if matched_dept != state.department:
            state.department = matched_dept
            # Only reset/auto-assign doctor if a doctor wasn't explicitly matched in this same turn
            if not doctor_name:
                docs = await run_with_loading_announcement(
                    get_consistent_doctors_for_dept(matched_dept),
                    session,
                    "धन्यवाद प्रतीक्षा के लिए, मैं जानकारी देख रही हूँ।"
                )
                if not docs:
                    state.department = None
                    await save_appointment_state(ctx.userdata["user_id"], state)
                    return f"Error: No doctors are currently available in the {matched_dept} department. Please inform the user that no doctor or department is available for their issue and deny the appointment."
                # Filter by Redis availability at the user's requested time (if already known)
                check_date = target_date
                check_time = target_time
                available_docs = await filter_available_doctors(docs, check_date, check_time)
                display_docs = available_docs if available_docs else docs
                if display_docs:
                    state.doctor_name = display_docs[0].get("doctorName") or display_docs[0].get("name")
                    state.doctor_id = display_docs[0].get("doctorId")
                    doc_msg = f"I have automatically assigned the doctor: {state.doctor_name}."
                else:
                    state.doctor_name = None
                    state.doctor_id = None
                    doc_msg = f"No doctors found for the department: {matched_dept}."
            state.appointment_date = None
            state.appointment_time = None
            state.availability_verified = False
            state.booking_status = "pending"
        state.step_attempts["department"] = 0

    # 9. Reason for visit validation & invalidation
    if reason:
        if detect_multiple_departments(reason):
            state.reason = None
            await save_appointment_state(ctx.userdata["user_id"], state)
            return "Error: Multiple complaints mentioned. Please mention only one primary problem."

        old_reason = state.reason
        old_dept = state.department
        state.reason = reason
        
        inferred_specialty = get_symptom_specialty(reason)
        if inferred_specialty:
            new_dept = normalize_department(inferred_specialty)
            if new_dept and not department and new_dept != old_dept:
                state.department = new_dept
                
                # Check if it is an unsupported department
                unsupported_depts = ["Dentistry", "ENT", "Dermatology", "Pulmonology", "Gastroenterology", "Urology", "Nephrology"]
                if new_dept in unsupported_depts:
                    hindi_names = {
                        "Dentistry": "दांतों का",
                        "ENT": "नाक, कान और गले का (ENT)",
                        "Dermatology": "त्वचा (Dermatology)",
                        "General Physician": "सामान्य रोग (General Physician)",
                        "Pulmonology": "फेफड़ों का (Pulmonology)",
                        "Gastroenterology": "पेट का (Gastroenterology)",
                        "Urology": "मूत्र रोग (Urology)",
                        "Nephrology": "किडनी (Nephrology)"
                    }
                    dept_name_hindi = hindi_names.get(new_dept, new_dept)
                    state.doctor_name = None
                    state.doctor_id = None
                    state.appointment_date = None
                    state.appointment_time = None
                    state.availability_verified = False
                    state.booking_status = "pending"
                    await save_appointment_state(ctx.userdata["user_id"], state)
                    return f"माफ़ कीजिएगा, लेकिन हमारे यहाँ {dept_name_hindi} विभाग उपलब्ध नहीं है, इसलिए मैं आपकी इस समस्या के लिए अपॉइंटमेंट बुक नहीं कर सकती।"

                if not doctor_name:
                    docs = await run_with_loading_announcement(
                        get_consistent_doctors_for_dept(new_dept),
                        session,
                        "धन्यवाद प्रतीक्षा के लिए, मैं जानकारी देख रही हूँ।"
                    )
                    if not docs:
                        state.reason = None
                        state.department = None
                        await save_appointment_state(ctx.userdata["user_id"], state)
                        return f"Error: No doctors are currently available in the {new_dept} department. Please inform the user that no doctor or department is available for their issue and deny the appointment."
                    # Filter by Redis availability at the user's requested time (if already known)
                    check_date = target_date
                    check_time = target_time
                    available_docs = await filter_available_doctors(docs, check_date, check_time)
                    display_docs = available_docs if available_docs else docs
                    if len(display_docs) == 1:
                        state.doctor_name = display_docs[0].get("doctorName") or display_docs[0].get("name")
                        state.doctor_id = display_docs[0].get("doctorId")
                    else:
                        state.doctor_name = None
                        state.doctor_id = None
                state.appointment_date = None
                state.appointment_time = None
                state.availability_verified = False
                state.booking_status = "pending"

                if old_reason:
                    new_reason_lower = reason.lower()
                    if "heart" in new_reason_lower or "chest" in new_reason_lower or "palpitation" in new_reason_lower:
                        theme = "heart"
                    elif "migraine" in new_reason_lower or "headache" in new_reason_lower or "dizziness" in new_reason_lower or "ch चक्कर" in new_reason_lower:
                        theme = "heart"
                    else:
                        theme = "specialty"
                    
                    if "heart" in new_reason_lower:
                        theme = "heart"
                    
                    await save_appointment_state(ctx.userdata["user_id"], state)
                    return f"You previously requested an appointment for {old_reason}. Since you are now reporting {theme}-related symptoms, I need to switch this appointment to {new_dept}."

    # 10. Mismatch Check
    target_reason = reason or state.reason
    target_dept = department or state.department
    if target_reason and target_dept:
        forbidden_msg = check_forbidden_combination(target_reason, target_dept)
        if forbidden_msg:
            return forbidden_msg
            
        symptom_dept = get_symptom_specialty(target_reason)
        if symptom_dept:
            def normalize_dept(d: str) -> str:
                d_clean = d.lower().strip()
                if "neuro" in d_clean: return "neurology"
                if "dent" in d_clean: return "dentistry"
                if "physician" in d_clean: return "general physician"
                if "pulm" in d_clean: return "pulmonology"
                if "derm" in d_clean: return "dermatology"
                if "ortho" in d_clean: return "orthopedics"
                if "cardio" in d_clean: return "cardiology"
                if "surg" in d_clean: return "surgeon"
                return d_clean

            if normalize_dept(symptom_dept) != normalize_dept(target_dept):
                await save_appointment_state(ctx.userdata["user_id"], state)
                hindi_dept_names = {
                    "Cardiology": "कार्डियोलॉजी (Cardiology)",
                    "Neurologist": "न्यूरोलॉजिस्ट (Neurology)",
                    "Orthopedics": "ऑर्थोपेडिक्स (Orthopedics)",
                    "Surgeon": "सर्जन (Surgeon)",
                    "General Physician": "सामान्य रोग (General Physician)"
                }
                inferred_hindi = hindi_dept_names.get(symptom_dept, symptom_dept)
                target_hindi = hindi_dept_names.get(target_dept, target_dept)
                return f"Error: आपकी समस्या ({target_reason}) {inferred_hindi} विभाग से संबंधित है, इसलिए इसे {target_hindi} विभाग में बुक नहीं किया जा सकता।"

    if appointment_date:
        state.appointment_date = appointment_date
        state.step_attempts["appointment_date"] = 0
        
    # 11. Time preference & slot selection
    is_flex_time = False
    if time_preference and time_preference.upper() == "FLEXIBLE":
        is_flex_time = True
    if appointment_time:
        app_time_lower = appointment_time.lower().strip()
        if app_time_lower in ["any", "sny", "any time", "anytime", "flexible", "earliest", "earliest available", "any slot", "कोई भी", "koi bhi", "sny time", "snytime", "कोई भी समय", "जब मिले", "किसी भी टाइम", "koi bhi samay", "jab mile", "kisi bhi time", "first available", "first available slot"]:
            is_flex_time = True
            appointment_time = ""
            
    if appointment_time or time_preference:
        if (
            not state.appointment_time 
            or (appointment_time and appointment_time.strip() != state.appointment_time.strip())
            or (time_preference and time_preference != state.time_preference)
        ):
            loop_err = check_loop("appointment_time")
            if loop_err: return loop_err
            
    if is_flex_time:
        state.time_preference = "FLEXIBLE"
        
    if appointment_time:
        state.appointment_time = normalize_time_slot(appointment_time)
        if not is_flex_time:
            state.availability_verified = False
        state.step_attempts["appointment_time"] = 0

    target_date = appointment_date or state.appointment_date
    is_flex_doc = getattr(state, "doctor_preference", None) == "any" or (doctor_name and doctor_name.lower().strip() in [
        "any", "any doctor", "any available doctor", "anyone is fine", "first available", "earliest available",
        "कोई भी", "कोई भी डॉक्टर", "कोई भी डॉ", "कोई भी चलेगा", "कोई भी डॉक्टर चलेगा",
        "koi bhi", "koi bhi doctor", "anyone", "flexible", "first available doctor", "earliest available doctor"
    ])
    if is_flex_doc:
        state.doctor_preference = "any"

    # Search for the earliest available slot matching flexibility
    if is_flex_doc or is_flex_time or state.time_preference == "FLEXIBLE":
        import datetime
        start_date = target_date
        if not start_date:
            start_date = _today_ist().strftime("%Y-%m-%d")
        
        start_dt = _today_ist()
        try:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        except Exception:
            pass

        resolved_doc_name = None
        resolved_doc_id = None
        resolved_time = None
        resolved_date = None
        
        slots = [
            "09:00 AM", "09:15 AM", "09:30 AM", "09:45 AM",
            "10:00 AM", "10:15 AM", "10:30 AM", "10:45 AM",
            "11:00 AM", "11:15 AM", "11:30 AM", "11:45 AM",
            "12:00 PM", "12:15 PM", "12:30 PM", "12:45 PM",
            "01:00 PM", "01:15 PM", "01:30 PM", "01:45 PM",
            "02:00 PM", "02:15 PM", "02:30 PM", "02:45 PM",
            "03:00 PM", "03:15 PM", "03:30 PM", "03:45 PM",
            "04:00 PM", "04:15 PM", "04:30 PM", "04:45 PM",
            "05:00 PM", "05:15 PM", "05:30 PM", "05:45 PM",
            "06:00 PM", "06:15 PM", "06:30 PM", "06:45 PM",
            "07:00 PM"
        ]
        
        from storage import is_slot_locked

        if is_flex_doc:
            # Search for any doctor in the department
            target_dept = department or state.department
            if target_dept:
                docs = await get_consistent_doctors_for_dept(target_dept)
                if docs:
                    found = False
                    for day_offset in range(14):
                        check_date = (start_dt + datetime.timedelta(days=day_offset)).strftime("%Y-%m-%d")
                        if is_flex_time or state.time_preference == "FLEXIBLE":
                            slots_to_check = slots
                        else:
                            slots_to_check = [normalize_time_slot(appointment_time or state.appointment_time)]
                        
                        for slot in slots_to_check:
                            if check_date == _today_ist().strftime("%Y-%m-%d"):
                                now_str = _now_ist().strftime("%I:%M %p")
                                if time_to_minutes(slot) <= time_to_minutes(now_str):
                                    continue
                                    
                            for doc in docs:
                                d_name = f"Dr. {clean_doctor_name(doc.get('doctorName') or doc.get('name') or '')}"
                                if not await is_slot_locked(d_name, check_date, slot):
                                    resolved_doc_name = doc.get("doctorName") or doc.get("name")
                                    resolved_doc_id = doc.get("doctorId")
                                    resolved_date = check_date
                                    resolved_time = slot
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
        else:
            # Only time is flexible (specific doctor selected)
            target_doc_name = state.doctor_name or doctor_name
            if target_doc_name:
                d_name = f"Dr. {clean_doctor_name(target_doc_name)}"
                found = False
                for day_offset in range(14):
                    check_date = (start_dt + datetime.timedelta(days=day_offset)).strftime("%Y-%m-%d")
                    for slot in slots:
                        if check_date == _today_ist().strftime("%Y-%m-%d"):
                            now_str = _now_ist().strftime("%I:%M %p")
                            if time_to_minutes(slot) <= time_to_minutes(now_str):
                                continue
                                
                        if not await is_slot_locked(d_name, check_date, slot):
                            resolved_doc_name = target_doc_name
                            resolved_doc_id = state.doctor_id
                            resolved_date = check_date
                            resolved_time = slot
                            found = True
                            break
                    if found:
                        break

        if resolved_doc_name:
            state.doctor_name = resolved_doc_name
            state.doctor_id = resolved_doc_id
            state.doctor_preference = "any" if is_flex_doc else "specific"
        if resolved_date:
            state.appointment_date = resolved_date
        if resolved_time:
            state.appointment_time = resolved_time
            state.availability_verified = True
            doc_msg = f"I have selected the earliest available option: Dr. {clean_doctor_name(state.doctor_name)} on {state.appointment_date} at {state.appointment_time}."
    # Release any stale lock left over from a prior verify_availability call
    # whose doctor/date/time no longer matches the current selection.
    if _prev_locked_doctor and _prev_locked_date and _prev_locked_time:
        changed = (
            clean_doctor_name(state.doctor_name or "").lower().strip()
            != clean_doctor_name(_prev_locked_doctor).lower().strip()
            or (state.appointment_date or "") != _prev_locked_date
            or normalize_time_slot(state.appointment_time or "") != normalize_time_slot(_prev_locked_time)
        )
        if changed:
            try:
                await release_appointment_slot(
                    doctor_name=_prev_locked_doctor,
                    date=_prev_locked_date,
                    time_slot=_prev_locked_time,
                    user_id=ctx.userdata.get("session_id", ctx.userdata.get("user_id", "anonymous")),
                )
                logger.info(
                    "Released stale slot lock %s %s %s after selection changed.",
                    _prev_locked_doctor, _prev_locked_date, _prev_locked_time,
                )
            except Exception as exc:
                logger.warning("Failed to release stale slot lock: %s", exc)

    if state.booking_status in ("PENDING", "pending", "DRAFT", "draft"):
        if state.patient_name and (state.patient_phone or state.phone_number) and state.department and state.doctor_id and state.appointment_date and state.appointment_time:
            state.booking_status = "READY_FOR_CONFIRMATION"
    elif state.booking_status == "READY_FOR_CONFIRMATION":
        if not (state.patient_name and (state.patient_phone or state.phone_number) and state.department and state.doctor_id and state.appointment_date and state.appointment_time):
            state.booking_status = "DRAFT"
            


    # --- Mandatory booking-review confirmation gate ---
    # Any change to a booking-relevant field invalidates a prior confirmation,
    # so the caller must hear (and re-approve) the updated summary before we
    # book. This directly enforces conversation rule #6 ("modify then re-summarize").
    core_fields_changed = any([
        patient_name, patient_phone, patient_name_spelling, department,
        doctor_name, appointment_date, appointment_time, reason
    ])
    if core_fields_changed and not user_confirmed_booking:
        ctx.userdata["booking_confirmed_by_user"] = False

    if user_confirmed_booking:
        ctx.userdata["booking_confirmed_by_user"] = True

    await save_appointment_state(ctx.userdata["user_id"], state)
    await refresh_agent_instructions(ctx, state)

    if pending_name_response:
        return pending_name_response
    if pending_phone_response:
        return pending_phone_response

    if user_confirmed_booking:
        return "Caller confirmed the booking. You may now call confirm_appointment."

    msg = f"Details updated successfully. Current progress: {state.completed_steps_summary()}"
    if doc_msg:
        msg = f"{doc_msg} {msg}"
    return msg


@function_tool
async def get_booking_summary(ctx: RunContext) -> str:
    """
    Generate the mandatory pre-booking summary from the CURRENT stored state.
    You MUST call this before asking the caller to confirm the booking, and you
    MUST speak the returned summary text to the caller verbatim (translated
    naturally into the caller's language if needed, but without dropping or
    inventing any field). After reading the summary, ask for a clear yes/no in
    your own words in the caller's language, and only after an explicit yes call
    `update_appointment_details(user_confirmed_booking=True)`.
    """
    state: AppointmentState = ctx.userdata["state"]

    missing = []
    if not state.doctor_name:
        missing.append("doctor")
    if not state.appointment_date:
        missing.append("date")
    if not state.appointment_time:
        missing.append("time")
    if not state.patient_name:
        missing.append("patient name")
    if not state.patient_phone:
        missing.append("phone number")
    if not state.reason:
        missing.append("reason for visit")
    if missing:
        return f"Cannot generate summary yet — still missing: {', '.join(missing)}. Ask the caller for these first."

    from tools import clean_doctor_name as _clean_doc
    _raw_name = state.doctor_name or "Doctor"
    doc_name = "Dr. " + _clean_doc(_raw_name) if _raw_name != "Doctor" else "Doctor"

    formatted_date = state.appointment_date
    try:
        import datetime
        dt = datetime.datetime.strptime(state.appointment_date.strip(), "%Y-%m-%d")
        formatted_date = dt.strftime("%d %B %Y")
    except Exception:
        pass

    summary = (
        "मैं verify कर दूँ—\n"
        f"• डॉक्टर: {doc_name}\n"
        f"• विभाग: {state.department or 'N/A'}\n"
        f"• तारीख: {formatted_date}\n"
        f"• समय: {state.appointment_time}\n"
        f"• मरीज़: {state.patient_name}\n"
        f"• मोबाइल नंबर: {state.patient_phone}\n"
        f"• कारण: {state.reason}\n"
        "क्या सभी जानकारी सही है?"
    )
    # A new/changed summary means any earlier confirmation is stale.
    ctx.userdata["booking_confirmed_by_user"] = False
    return f"Speak this exact summary to the caller verbatim, then wait for their response:\n{summary}"
def add_15_minutes(time_str: str) -> str:
    import datetime
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M %p"):
        try:
            dt = datetime.datetime.strptime(time_str.strip(), fmt)
            new_dt = dt + datetime.timedelta(minutes=15)
            return new_dt.strftime("%I:%M %p")
        except ValueError:
            continue
    m = re.match(r"(\d+):(\d+)\s*(AM|PM|am|pm)?", time_str.strip())
    if m:
        hr = int(m.group(1))
        mn = int(m.group(2))
        period = (m.group(3) or "AM").upper()
        if period == "PM" and hr < 12:
            hr += 12
        elif period == "AM" and hr == 12:
            hr = 0
        dt = datetime.datetime(2000, 1, 1, hr, mn)
        new_dt = dt + datetime.timedelta(minutes=15)
        return new_dt.strftime("%I:%M %p")
    raise ValueError(f"Could not parse time format: {time_str}")
    

def time_to_minutes(time_str: str) -> int:
    time_clean = time_str.strip().upper()
    m = re.match(r"^(\d+):(\d+)\s*(AM|PM)$", time_clean)
    if not m:
        m = re.match(r"^(\d+):(\d+)\s*(AM|PM)$", normalize_time_slot(time_str))
    if m:
        hr = int(m.group(1))
        mn = int(m.group(2))
        am_pm = m.group(3)
        if am_pm == "PM" and hr < 12:
            hr += 12
        elif am_pm == "AM" and hr == 12:
            hr = 0
        return hr * 60 + mn
    return 0

@function_tool
async def verify_availability(ctx: RunContext) -> str:
    """
    Verify if the requested doctor, date, and time slot are available before confirming the appointment.
    """
    state: AppointmentState = ctx.userdata["state"]
    if not state.doctor_name:
        return "Missing doctor name. Cannot verify availability."

    flex_keywords = ["any", "sny", "any time", "anytime", "flexible", "earliest", "earliest available", "any slot", "any date", "koi bhi", "koi bhi samay", "jab mile", "kisi bhi time", "कोई भी", "कोई भी समय", "first available"]
    is_flex_time = state.appointment_time and state.appointment_time.lower().strip() in flex_keywords
    is_flex_date = not state.appointment_date or str(state.appointment_date).lower().strip() in flex_keywords

    if is_flex_time or is_flex_date:
        try:
            session = ctx.userdata.get("session")
            found_date, found_time = await run_with_loading_announcement(
                find_chronological_earliest_slot(state.doctor_name),
                session,
                "मैं आपके लिए सबसे पहला उपलब्ध समय देख रही हूँ।"
            )
            if not found_date or not found_time:
                state.availability_verified = False
                await save_appointment_state(ctx.userdata["user_id"], state)
                return "क्षमा करें, आने वाले 30 दिनों में कोई अपॉइंटमेंट स्लॉट उपलब्ध नहीं है। कृपया बाद में प्रयास करें।"
            
            state.appointment_date = found_date
            state.appointment_time = found_time
        except Exception as e:
            return "अपॉइंटमेंट स्लॉट चेक करते समय सिस्टम में तकनीकी समस्या आ गई है, कृपया बाद में प्रयास करें।"

    if not state.appointment_date or not state.appointment_time:
        return "Missing date, or time. Cannot verify availability."

    # ── POLICY-ALIGNED AVAILABILITY ──────────────────────────────────────────
    # Per the system instructions, multiple bookings per slot ARE allowed and a
    # future time needs no availability check. So verification only has to ensure
    # the requested slot is not in the past. We deliberately do NOT reject on an
    # existing booking or a slot lock here — that mismatch (verify locking a slot
    # that confirm then couldn't book, or vice-versa) is what produced the endless
    # "aapka appointment book nahi ho payega / koi aur time?" loop in the
    # transcript. If the requested same-day time has already passed, we proactively
    # offer the earliest still-valid slot instead of a bare failure.
    _rd, _rt, _terr = validate_and_normalize_appointment_datetime(
        state.appointment_date, state.appointment_time
    )
    if _terr:
        state.availability_verified = False
        await save_appointment_state(ctx.userdata["user_id"], state)
        earliest = None
        try:
            earliest = await find_earliest_available_slot(state.doctor_name, state.appointment_date)
        except Exception as _e:
            logger.warning("find_earliest_available_slot failed: %s", _e)
        await refresh_agent_instructions(ctx, state)
        if earliest:
            return (
                f"Requested time ({state.appointment_time}) is not bookable: {_terr} "
                f"The earliest still-available slot on {state.appointment_date} is {earliest}. "
                f"Offer {earliest} to the caller in their language; ONLY if they agree, call "
                f"update_appointment_details(appointment_time='{earliest}') and then verify_availability again. "
                f"Do not change the time yourself before they agree."
            )
        try:
            _tom = (_today_ist() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            _tom = None
        if _tom:
            return (
                f"Requested time is not bookable: {_terr} No slot remains for today. "
                f"Offer tomorrow morning ({_tom}); if the caller agrees, call "
                f"update_appointment_details(appointment_date='{_tom}', appointment_time='09:00 AM') "
                f"then verify_availability again."
            )
        return f"Requested time is not bookable: {_terr} Please ask the caller for a valid future time."

    # Future time → accept immediately (normalize, mark verified, no lock/conflict gate).
    state.appointment_date, state.appointment_time = _rd, _rt
    state.availability_verified = True
    await save_appointment_state(ctx.userdata["user_id"], state)
    await refresh_agent_instructions(ctx, state)
    return "Slot is available. Proceed to confirm the appointment."

    # ── (Legacy conflict/lock path below is intentionally unreachable now) ────
    from storage import check_doctor_slot_conflict, _resolve_doctor_id
    doc_id = state.doctor_id
    if not doc_id and state.doctor_name:
        doc_id = _resolve_doctor_id(state.doctor_name)
    conflict = await check_doctor_slot_conflict(
        str(doc_id) if doc_id is not None else None,
        state.appointment_date,
        state.appointment_time
    )
    if conflict:
        # If the caller requested "any doctor", try other doctors in the same department
        # at the SAME date and time before reporting unavailability.
        if getattr(state, "doctor_preference", None) == "any" and state.department:
            all_dept_docs = await get_consistent_doctors_for_dept(state.department)
            for alt_doc in all_dept_docs:
                alt_name = alt_doc.get("doctorName") or alt_doc.get("name") or ""
                alt_id = alt_doc.get("doctorId")
                # Skip the currently-conflicting doctor
                if alt_id and str(alt_id) == str(state.doctor_id):
                    continue
                if clean_doctor_name(alt_name).lower() == clean_doctor_name(state.doctor_name or "").lower():
                    continue
                # Check if this alternative doctor is free
                alt_locked = await is_slot_locked(
                    f"Dr. {clean_doctor_name(alt_name)}",
                    state.appointment_date,
                    state.appointment_time
                )
                if not alt_locked:
                    # Switch to this available doctor
                    state.doctor_name = alt_name
                    state.doctor_id = alt_id
                    state.availability_verified = False
                    await save_appointment_state(ctx.userdata["user_id"], state)
                    # Proceed to lock slot for this doctor (fall through to lock logic below)
                    break
            else:
                # No alternative doctor found in this department at this time
                state.availability_verified = False
                await save_appointment_state(ctx.userdata["user_id"], state)
                alt_msg = await get_alternatives(
                    state.doctor_name,
                    state.appointment_date,
                    state.appointment_time,
                    state.department
                )
                return f"कोई भी {state.department} डॉक्टर {state.appointment_date} को {state.appointment_time} पर उपलब्ध नहीं है। {alt_msg}"

            # Re-check conflict for the newly selected doctor before locking
            from storage import check_doctor_slot_conflict as _csc, _resolve_doctor_id as _rid
            new_doc_id = state.doctor_id or _rid(state.doctor_name)
            new_conflict = await _csc(str(new_doc_id), state.appointment_date, state.appointment_time)
            if new_conflict:
                state.availability_verified = False
                await save_appointment_state(ctx.userdata["user_id"], state)
                return f"कोई भी {state.department} डॉक्टर {state.appointment_date} को {state.appointment_time} पर उपलब्ध नहीं है।"
            # Fall through to the lock logic below for this newly assigned doctor
        else:
            state.availability_verified = False
            await save_appointment_state(ctx.userdata["user_id"], state)
            formatted_date = state.appointment_date
            try:
                import datetime
                dt = datetime.datetime.strptime(state.appointment_date.strip(), "%Y-%m-%d")
                formatted_date = dt.strftime("%d-%b-%Y")
            except Exception:
                pass
            
            doc_name = state.doctor_name or "Doctor"
            if doc_name.startswith("Dr ") and not doc_name.startswith("Dr. "):
                doc_name = "Dr. " + doc_name[3:]
            elif not doc_name.startswith("Dr.") and not doc_name.startswith("Dr "):
                doc_name = "Dr. " + doc_name

            return f"{doc_name} already has a booked appointment on {formatted_date} at {state.appointment_time}. Please choose another available time slot."

    session = ctx.userdata.get("session")
    locked = await run_with_loading_announcement(
        lock_appointment_slot(
            doctor_name=state.doctor_name,
            date=state.appointment_date,
            time_slot=state.appointment_time,
            user_id=ctx.userdata.get("session_id", ctx.userdata.get("user_id", "anonymous"))
        ),
        session,
        "एक क्षण, मैं उपलब्ध स्लॉट देख रही हूँ।"
    )
    if not locked:
        state.availability_verified = False
        await save_appointment_state(ctx.userdata["user_id"], state)
        
        # Iterate over the valid slots to find the nearest available one
        valid_slots = [
            "09:00 AM", "09:15 AM", "09:30 AM", "09:45 AM",
            "10:00 AM", "10:15 AM", "10:30 AM", "10:45 AM",
            "11:00 AM", "11:15 AM", "11:30 AM", "11:45 AM",
            "12:00 PM", "12:15 PM", "12:30 PM", "12:45 PM",
            "01:00 PM", "01:15 PM", "01:30 PM", "01:45 PM",
            "02:00 PM", "02:15 PM", "02:30 PM", "02:45 PM",
            "03:00 PM", "03:15 PM", "03:30 PM", "03:45 PM",
            "04:00 PM", "04:15 PM", "04:30 PM", "04:45 PM",
            "05:00 PM", "05:15 PM", "05:30 PM", "05:45 PM",
            "06:00 PM", "06:15 PM", "06:30 PM", "06:45 PM",
            "07:00 PM"
        ]
        
        old_time = state.appointment_time
        target_mins = time_to_minutes(old_time)
        
        sorted_slots = []
        for s in valid_slots:
            s_mins = time_to_minutes(s)
            dist = abs(s_mins - target_mins)
            # We want to check slots that are DIFFERENT from old_time
            if normalize_time_slot(s) != normalize_time_slot(old_time):
                sorted_slots.append((dist, s))
        
        sorted_slots.sort(key=lambda x: x[0])
        
        found_slot = None
        for dist, slot in sorted_slots:
            # Enforce Rule 8: Same-day booking must be at least 30 minutes in future
            try:
                today_date = _today_ist()
                parsed_date = datetime.datetime.strptime(state.appointment_date.strip(), "%Y-%m-%d").date()
                if parsed_date == today_date:
                    now_str = _now_ist().strftime("%I:%M %p")
                    if time_to_minutes(slot) < time_to_minutes(now_str):
                        continue
            except Exception:
                if state.appointment_date == _today_ist().strftime("%Y-%m-%d"):
                    now_str = _now_ist().strftime("%I:%M %p")
                    if time_to_minutes(slot) < time_to_minutes(now_str):
                        continue

            if not await is_slot_locked(state.doctor_name, state.appointment_date, slot):
                found_slot = slot
                break

        if found_slot:
            # BUG FIX: previously this silently overwrote state.appointment_time
            # here — BEFORE the caller had agreed to the alternative. That is what
            # caused a caller who asked for 3:00 PM to end up booked at 12:00 PM:
            # the state was mutated the instant an alternative was found, so if the
            # model proceeded to confirm_appointment for any reason (missed the
            # "wait for approval" instruction), it silently booked the wrong time.
            # We now keep state.appointment_time UNCHANGED (still the caller's
            # original request) and only expose the suggestion. The model must
            # call update_appointment_details(appointment_time=<alt>) itself, and
            # ONLY after the caller explicitly agrees to the alternative slot.
            state.availability_verified = False
            await save_appointment_state(ctx.userdata["user_id"], state)
            await refresh_agent_instructions(ctx, state)
            return (
                f"मूल अनुरोधित समय ({old_time}) उपलब्ध नहीं है, इसलिए यह अभी तक बुक नहीं किया गया है। "
                f"निकटतम उपलब्ध विकल्प: {found_slot}। "
                f"कॉलर से पूछें कि क्या वह {found_slot} स्वीकार करता है — मूल समय ({old_time}) को स्वयं न बदलें। "
                f"यदि कॉलर स्पष्ट रूप से {found_slot} स्वीकार करता है, तो तभी "
                f"update_appointment_details(appointment_time='{found_slot}') कॉल करें, फिर verify_availability फिर से चलाएं। "
                f"अगर कॉलर मना करता है, तो मूल समय पर बने रहें और कोई और विकल्प पूछें।"
            )

        alt_msg = await get_alternatives(
            state.doctor_name,
            state.appointment_date,
            state.appointment_time,
            state.department
        )
        return alt_msg
        
    state.availability_verified = True
    await save_appointment_state(ctx.userdata["user_id"], state)
    await refresh_agent_instructions(ctx, state)
    return "Slot is available. Proceed to confirm the appointment."


@function_tool
async def confirm_appointment(
    ctx: RunContext,
    user_confirmed: Annotated[bool, "Set to True in THIS same call once the caller has heard the booking summary and clearly said yes to booking it (any language). Passing it here books in one step — no separate update_appointment_details call is needed."] = False,
) -> str:
    """
    Confirm and book the appointment. Call after all required details are collected
    and the caller has said yes to the summary. Pass user_confirmed=True to record
    that final yes in the same call.
    """
    state: AppointmentState = ctx.userdata["state"]
    transcript = _get_transcript(ctx.userdata)
    user_id = ctx.userdata["user_id"]

    # Single-step confirmation: if the model passes the caller's final yes right
    # here, honour it (previously the yes had to arrive via a separate
    # update_appointment_details call, which the realtime model frequently failed
    # to chain — leaving the caller confirming over and over with nothing booked).
    if user_confirmed:
        ctx.userdata["booking_confirmed_by_user"] = True

    # SOFT pause: used for "need more info / need the caller's yes" returns. Unlike
    # the old revoke_confirmation, this does NOT reset availability_verified, so the
    # model does not have to re-verify (and re-lock) the slot every time — that
    # reset is exactly what turned each confirm attempt into another loop iteration
    # in the transcript. booking_status stays DRAFT until we actually book.
    async def soft_pause():
        state.booking_status = "DRAFT"
        await save_appointment_state(user_id, state)
        await save_call_transcript(user_id, transcript)

    # HARD revoke: only for genuine unavailability where the slot must be released.
    async def revoke_confirmation():
        state.booking_status = "DRAFT"
        state.availability_verified = False
        await save_appointment_state(user_id, state)
        await save_call_transcript(user_id, transcript)

    # Check for emergency in state reason or other details
    
    # Resolve doctor ID if missing
    if state.doctor_id is None and state.doctor_name:
        data = await fetch_live_hospital_data(force=False)
        if data and "doctors" in data:
            from tools import clean_doctor_name
            doc_cleaned = clean_doctor_name(state.doctor_name).lower().strip()
            for doc in data["doctors"]:
                if clean_doctor_name(doc.get("doctorName", "")).lower().strip() == doc_cleaned:
                    state.doctor_id = doc.get("doctorId")
                    break

    # 1. Check required fields presence and ask directly for missing details.
    #    Uses soft_pause (keeps the verified slot) so re-supplying one field does
    #    not force a full re-verify/re-lock cycle.
    if not state.patient_name:
        await soft_pause()
        return "Please provide the patient's full name."
    if not state.patient_phone:
        await soft_pause()
        return "Please provide the patient's phone number."

    if not state.department or not state.doctor_id or not state.doctor_name:
        await soft_pause()
        return "Please provide the doctor or department name."
    if not state.appointment_date:
        await soft_pause()
        return "Please provide the preferred appointment date."
    if not state.appointment_time:
        await soft_pause()
        return "Please provide the preferred appointment time."
    if not state.reason:
        await soft_pause()
        return "What is the reason for the visit?"

    # 2. GATE — the caller must have said yes to the summary before we book. This
    # no longer resets availability_verified (soft_pause), so a missed yes prompts
    # for confirmation WITHOUT wiping the verified slot and forcing a re-verify
    # loop. The model can satisfy this by passing user_confirmed=True to THIS tool
    # in one step (preferred), or via update_appointment_details(user_confirmed_booking=True).
    if not ctx.userdata.get("booking_confirmed_by_user"):
        await soft_pause()
        return (
            "Before booking, read the caller the summary from get_booking_summary and get a clear yes. "
            "Once they say yes, call confirm_appointment(user_confirmed=True) to book it in one step."
        )
    state.patient_phone_confirmed = True
    state.patient_name_locked = True

    max_attempts = 2
    # Defined OUTSIDE the try so the except handler (which calls
    # release_appointment_slot(user_id=current_session_id)) can never itself raise
    # a NameError when an exception is thrown before these were assigned — which
    # would otherwise mask the real error and skip lock cleanup.
    caller_phone_raw = ctx.userdata.get("caller_phone", "")
    current_session_id = ctx.userdata.get("session_id", ctx.userdata.get("user_id", "anonymous"))
    for attempt in range(1, max_attempts + 1):
        try:

            # 3. Check phone number validity
            clean_phone = normalize_and_align_phone_number(state.patient_phone, caller_phone_raw)
            if not (clean_phone.isdigit() and len(clean_phone) == 10 and clean_phone[0] in "6789"):
                try:
                    await release_appointment_slot(
                        doctor_name=state.doctor_name,
                        date=state.appointment_date,
                        time_slot=state.appointment_time,
                        user_id=current_session_id
                    )
                except Exception:
                    pass
                state.patient_phone = None
                state.patient_phone_confirmed = False
                await soft_pause()
                return ("STOP — the phone number on file is not a valid 10-digit Indian mobile "
                        "(must be 10 digits starting 6/7/8/9). Do NOT book. Ask the caller to say "
                        "their 10-digit mobile number again, digit by digit, and set it via "
                        "update_appointment_details before retrying.")

            # 4. Check date/time validity
            resolved_date, resolved_time, err_msg = validate_and_normalize_appointment_datetime(
                state.appointment_date,
                state.appointment_time
            )
            if err_msg:
                await revoke_confirmation()
                return f"Error: Cannot confirm appointment. {err_msg}"

            # 4a. Check doctor ID exists in backend list
            data = await fetch_live_hospital_data(force=False)
            doctor_exists = False
            if data and "doctors" in data:
                for doc in data["doctors"]:
                    if doc.get("doctorId") == state.doctor_id:
                        doctor_exists = True
                        break
            if not doctor_exists:
                await revoke_confirmation()
                return f"Error: Doctor ID {state.doctor_id} does not exist in backend doctor list."

            # NOTE: Per policy the hospital runs a token/queue model — multiple
            # bookings for the same doctor+slot ARE allowed. So we do NOT reject on
            # a slot "conflict" here (that rejection, together with verify locking
            # the slot, is what made every same-slot confirm fail in the transcript).
            # The exact-duplicate guard below still prevents the SAME caller from
            # booking the identical appointment twice.

            # 4b. Check for duplicate booking (same person, same everything)
            existing_bookings = await search_confirmed_appointments(
                patient_phone=state.patient_phone,
                patient_name=state.patient_name,
                appointment_date=state.appointment_date,
                appointment_time=state.appointment_time,
                doctor_id=str(state.doctor_id) if state.doctor_id is not None else None,
                doctor_name=state.doctor_name
            )
            if existing_bookings:
                return "इस समय के लिए आपकी अपॉइंटमेंट पहले से मौजूद है।"

            # Set booking in progress
            state.booking_status = "BOOKING_IN_PROGRESS"
            await save_appointment_state(user_id, state)

            # 5. Check lock owner first to handle duplicate requests gracefully (Rule 10)
            r = await _get_redis()
            from storage import _resolve_doctor_id
            doc_id = _resolve_doctor_id(state.doctor_name)
            d = state.appointment_date.strip().lower().replace(" ", "_")
            t = state.appointment_time.strip().lower().replace(" ", "_")
            booking_key = f"booking_lock:{doc_id}:{d}:{t}"
            
            current_holder = None
            if r is not None:
                try:
                    current_holder = await r.get(booking_key)
                    if isinstance(current_holder, bytes):
                        current_holder = current_holder.decode("utf-8")
                except Exception:
                    pass
            else:
                from storage import _memory_store
                current_holder = _memory_store.get(booking_key)

            session = ctx.userdata.get("session")
            # 6. Try to lock the slot (idempotent: if this session already holds it, reuse the lock)
            if current_holder == current_session_id:
                # This session already locked the slot (e.g. from verify_availability) — proceed directly
                locked = True
            else:
                locked = await run_with_loading_announcement(
                    lock_appointment_slot(
                        doctor_name=state.doctor_name,
                        date=state.appointment_date,
                        time_slot=state.appointment_time,
                        user_id=current_session_id
                    ),
                    session,
                    "एक क्षण, मैं उपलब्ध स्लॉट देख रही हूँ।"
                )
            
            # Lock is now BEST-EFFORT only (it serves as light idempotency / a
            # record that this session booked the slot). Because multiple bookings
            # per slot are allowed, a failed lock must NOT abort the booking — the
            # old behaviour returned alternatives here and blocked the caller even
            # though the policy permits the slot. Proceed to book regardless.
            if not locked:
                logger.info("Slot lock not acquired for %s — proceeding anyway (multiple bookings allowed).", booking_key)

            state.availability_verified = True

            # 7. Other validations (e.g. consistency checks) after locking
            is_valid, err_msg = validate_appointment(state, data)
            
            if not is_valid:
                if err_msg != "Slot availability not verified.":
                    # Release slot on any other validation failure
                    await release_appointment_slot(
                        doctor_name=state.doctor_name,
                        date=state.appointment_date,
                        time_slot=state.appointment_time,
                        user_id=current_session_id
                    )
                    state.booking_status = "FAILED"
                    state.availability_verified = False
                    await save_appointment_state(user_id, state)
                    return f"Error: Booking validation failed. Reason: {err_msg}"
                
            if not state.appointment_id:
                from storage import generate_unique_appointment_id
                state.appointment_id = await generate_unique_appointment_id()

            state.booking_status = "BOOKED"
            state.completed_appointments_count = (state.completed_appointments_count or 0) + 1
            transcript.booking_status = "confirmed"
            
            transcript.add_event(
                "appointment_booked",
                "Appointment successfully confirmed.",
                {"state": state.model_dump()}
            )
            
            await save_appointment_state(user_id, state)
            await save_call_transcript(user_id, transcript)
            # BUG FIX: this appointment was never being written to the
            # `confirmed_appointment:*` registry that `search_appointment`
            # and `reschedule_confirmed_appointment` read from — only the
            # reschedule path called save_confirmed_appointment, so a fresh
            # booking was invisible to later lookups/duplicate-checks even
            # though the caller had just been told it was confirmed.
            await save_confirmed_appointment(state)
            await refresh_agent_instructions(ctx, state)

            # Send SMS notification.
            # SECURITY FIX: the SMS gateway auth key was hardcoded in source here
            # (a committed secret). It now comes from the SMS_GATEWAY_AUTHKEY env
            # var; if that is unset the SMS is skipped with a warning rather than
            # leaking/using a baked-in credential. Sender ID / DLT template / brand
            # text are also env-overridable so they can match the actual hospital
            # (note: greeting says "Arora Hospital" but this template says
            # "Gupta skin & Dental / GSDHOS" — reconcile these to one brand).
            target_phone = state.patient_phone or state.phone_number
            sms_authkey = os.environ.get("SMS_GATEWAY_AUTHKEY", "").strip()
            if target_phone and not sms_authkey:
                logger.warning("SMS_GATEWAY_AUTHKEY not set — skipping booking-confirmation SMS.")
            elif target_phone:
                reg_no = target_phone
                apt_no = state.appointment_id or "APT12345"
                sms_sender = os.environ.get("SMS_SENDER_ID", "GSDHOS")
                sms_dlt_id = os.environ.get("SMS_DLT_TE_ID", "1207163540100790199")
                sms_app_name = os.environ.get("SMS_APP_BRAND", "Gupta skin & Dental app. - GSDHOS")
                sms_params = {
                    "authkey": sms_authkey,
                    "mobiles": target_phone,
                    "message": (
                        f"Registration No :{reg_no} "
                        f"Appointment No. :{apt_no} "
                        "To track approximate time when your number will come "
                        f"download {sms_app_name}"
                    ),
                    "sender": sms_sender,
                    "route": "B",
                    "DLT_TE_ID": sms_dlt_id,
                }
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        sms_resp = await client.get(
                            "https://sms.bulksmsserviceproviders.com/api/send_http.php",
                            params=sms_params,
                            timeout=10.0
                        )
                        logger.info(f"SMS API response: {sms_resp.status_code} - {sms_resp.text}")
                except Exception as sms_err:
                    logger.error(f"Failed to send booking SMS: {sms_err}")

            # BUG FIX: previously this returned a generic "Inform the user!"
            # instruction and left the model to freely generate the final
            # confirmation sentence, which is how details (most notably the
            # appointment TIME) were sometimes dropped from the spoken
            # confirmation. The exact sentence — with every field filled in from
            # state — is now generated in code and must be spoken verbatim.
            doc_name_final = state.doctor_name or "Doctor"
            if doc_name_final.startswith("Dr ") and not doc_name_final.startswith("Dr. "):
                doc_name_final = "Dr. " + doc_name_final[3:]
            elif not doc_name_final.startswith("Dr.") and not doc_name_final.startswith("Dr "):
                doc_name_final = "Dr. " + doc_name_final
            formatted_date_final = state.appointment_date
            try:
                import datetime
                dt = datetime.datetime.strptime(state.appointment_date.strip(), "%Y-%m-%d")
                formatted_date_final = dt.strftime("%d %B %Y")
            except Exception:
                pass

            final_confirmation_text = "आपकी अपॉइंटमेंट कन्फर्म हो गई है। क्या आपको किसी और सहायता की आवश्यकता है? "
            return (
                "Appointment confirmed successfully. Speak this EXACT sentence to the "
                "caller verbatim: "
                f"\"{final_confirmation_text}\""
            )
        except Exception as e:
            logger.error(f"Error during confirm_appointment: {e}", exc_info=True)
            # Release lock on failure
            try:
                await release_appointment_slot(
                    doctor_name=state.doctor_name,
                    date=state.appointment_date,
                    time_slot=state.appointment_time,
                    user_id=current_session_id
                )
            except Exception:
                pass
            if attempt == max_attempts:
                state.booking_status = "FAILED"
                state.availability_verified = False
                try:
                    await save_appointment_state(user_id, state)
                except Exception:
                    pass
                return f"Error: Booking failed. Reason: {str(e)}"
            else:
                logger.info(f"Retrying confirm_appointment automatically (attempt {attempt+1}/{max_attempts})...")
                await asyncio.sleep(0.5)

@function_tool
async def search_appointment(
    ctx: RunContext,
    patient_name: Annotated[str, "Patient's full name"],
    patient_phone: Annotated[str, "Patient's 10-digit mobile number"],
    appointment_date: Annotated[str | None, "Preferred appointment date"] = None,
    appointment_time: Annotated[str | None, "Preferred appointment time"] = None,
    department: Annotated[str | None, "Consultation department"] = None
) -> str:
    """
    Search for an existing confirmed appointment using the patient's name and phone number.
    You must verify patient name and phone number together to match identity.
    """
    caller_phone_raw = ctx.userdata.get("caller_phone", "")
    clean_phone = normalize_and_align_phone_number(patient_phone, caller_phone_raw)
        
    results = await search_confirmed_appointments(
        patient_phone=clean_phone,
        patient_name=patient_name,
        appointment_date=appointment_date,
        appointment_time=appointment_time,
        department=department
    )
    if not results:
        return f"No confirmed appointments found for patient '{patient_name}' with phone '{clean_phone}'."
        
    lines = []
    for idx, appt in enumerate(results, 1):
        lines.append(
            f"Appointment {idx}: Name={appt.patient_name}, Phone={appt.patient_phone}, "
            f"Dept={appt.department}, Doctor={appt.doctor_name}, "
            f"Date={appt.appointment_date}, Time={appt.appointment_time}, Reason={appt.reason}"
        )
    return "\n".join(lines)

@function_tool
async def reschedule_confirmed_appointment(
    ctx: RunContext,
    patient_name: Annotated[str, "Patient's full name to match"],
    patient_phone: Annotated[str, "Patient's phone number to match"],
    old_date: Annotated[str, "Old appointment date"],
    old_time: Annotated[str, "Old appointment time"],
    old_department: Annotated[str, "Old department"],
    new_date: Annotated[str, "New appointment date"],
    new_time: Annotated[str, "New appointment time"]
) -> str:
    """
    Reschedule an existing confirmed appointment to a new date and time.
    """
    caller_phone_raw = ctx.userdata.get("caller_phone", "")
    clean_phone = normalize_and_align_phone_number(patient_phone, caller_phone_raw)

    results = await search_confirmed_appointments(
        patient_phone=clean_phone,
        patient_name=None,
        appointment_date=old_date,
        appointment_time=old_time,
        department=old_department
    )
    if not results:
        return f"Error: No existing appointment found for patient '{patient_name}' at {old_date} {old_time}."
        
    results.sort(key=lambda x: x.patient_name or "", reverse=True)
    old_appt = results[0]
    
    if old_appt.patient_name.lower().strip() != patient_name.lower().strip():
        return "Error: Patient name differs. Do not overwrite existing appointment. Treat as a different patient."

    # Ensure the new slot is not already locked or booked
    if await is_slot_locked(old_appt.doctor_name, new_date, new_time):
        return f"Error: Slot {new_date} at {new_time} is not available for {old_appt.doctor_name}."

    # Reschedule appointment
    await delete_confirmed_appointment(old_appt)
    
    old_appt.appointment_date = new_date
    old_appt.appointment_time = new_time
    old_appt.availability_verified = True
    old_appt.booking_status = "confirmed"
    
    await save_confirmed_appointment(old_appt)
    
    ctx.userdata["state"] = old_appt
    await save_appointment_state(ctx.userdata["user_id"], old_appt)
    
    return f"Successfully loaded and rescheduled appointment for {patient_name} to {new_date} {new_time}."

class HospitalReceptionistAgent(Agent):
    def __init__(self, chat_ctx: ChatContext, state: AppointmentState, userdata: dict, instructions: str):
        self.state = state
        self.userdata = userdata
        # Strong references to fire-and-forget background tasks. asyncio only keeps
        # a WEAK reference to a task, so without this a per-turn task could be
        # garbage-collected mid-flight ("Task was destroyed but it is pending").
        self._bg_tasks: set[asyncio.Task] = set()
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            tools=[
                get_doctor_details,
                check_available_doctors,
                check_available_slots,
                get_hospital_services,
                get_hospital_info,
                update_appointment_details,
                get_booking_summary,
                verify_availability,
                confirm_appointment,
                search_appointment,
                reschedule_confirmed_appointment,
            ]
        )

    async def _background_refresh_hospital_data(self) -> None:
        """Refresh hospital data in the background — never blocks the LLM response.

        BUG FIX: this used to re-implement its own httpx fetch and ran on EVERY
        user turn while completely ignoring the _DATA_CACHE TTL — so a long call
        fired dozens of requests (2 retries each) at the external staging API, a
        latency source and a retry-storm risk. It now delegates to
        fetch_live_hospital_data(force=False), which is cache-aware (120s TTL) and
        already carries the httpx + Playwright fallbacks and cache-file write. On a
        cache hit this returns almost instantly and makes no network call at all.
        """
        try:
            data = await fetch_live_hospital_data(force=False)
            _DATA_CACHE["status"] = "online" if data else "offline"
        except Exception as e:
            logger.warning("Background hospital data refresh failed: %s", e)
            _DATA_CACHE["status"] = "offline"

    async def _background_update_instructions(self) -> None:
        """Update agent instructions in the background — never blocks the LLM response."""
        try:
            await update_agent_instructions(self, self.state)
        except Exception as e:
            logger.error("Error updating agent instructions (background): %s", e)

    async def on_user_turn_completed(self, chat_ctx: ChatContext, new_message: ChatMessage) -> None:
        logger.info("on_user_turn_completed triggered. User message: %s", getattr(new_message, "text_content", ""))

        # Fire-and-forget: refresh hospital data AND instructions in the background
        # so the LLM response starts IMMEDIATELY — no blocking at all. Keep a strong
        # reference in self._bg_tasks (discarded on completion) so the event loop
        # doesn't GC a still-running task; these are also cancelled on disconnect.
        for _coro in (self._background_refresh_hospital_data(),
                      self._background_update_instructions()):
            _t = asyncio.create_task(_coro)
            self._bg_tasks.add(_t)
            _t.add_done_callback(self._bg_tasks.discard)

        await super().on_user_turn_completed(chat_ctx, new_message)

def prewarm(proc: agents.JobProcess):
    from storage import _reset_redis_state
    _reset_redis_state()

async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    caller_participant = None
    for p in ctx.room.remote_participants.values():
        if p.identity:
            caller_participant = p
            break

    user_id = caller_participant.identity if caller_participant else "anonymous"
    import uuid
    session_id = f"session_{uuid.uuid4().hex[:12]}"

    caller_phone = ""
    if caller_participant:
        identity = caller_participant.identity or ""
        if identity.startswith("sip:"):
            caller_phone = identity[4:]
        else:
            caller_phone = identity
        if "@" in caller_phone:
            caller_phone = caller_phone.split("@")[0]
        caller_phone = "".join(c for c in caller_phone if c.isdigit() or c == "+")
        if not caller_phone and hasattr(caller_participant, "attributes"):
            caller_phone = caller_participant.attributes.get("sip.phoneNumber", "")
            
    # Fallback to room name extraction
    if not caller_phone and ctx.room and ctx.room.name:
        matches = re.findall(r"\d{10,12}", ctx.room.name)
        if matches:
            caller_phone = matches[0]

    # Every new call/session is a fresh conversation. Start with a clean state.
    state = AppointmentState()

    # caller_phone is stored in userdata for logging/routing only.
    # The agent must always explicitly ask the patient for their contact number.
    # Do NOT pre-fill state.patient_phone from the caller ID.

    skip_queue = SkipQueue()
    from tools import CallTranscript
    transcript = CallTranscript(user_id=user_id)

    gcp_project = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    gcp_location = os.environ.get("GCP_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION")
    
    if gcp_project and "GOOGLE_CLOUD_PROJECT" not in os.environ:
        os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project
    
    # Using global _human_voice_instructions definition

    realtime_model = RealtimeModel(
        model="gemini-live-2.5-flash-native-audio",
        voice="Autonoe",
        temperature=0.7,
        instructions=_human_voice_instructions,
        vertexai=True,
        project=gcp_project,
        location=gcp_location,
    )

    _userdata: dict = {
        "user_id": user_id,
        "session_id": session_id,
        "state": state,
        "skip_queue": skip_queue,
        "transcript": transcript,
        "caller_phone": caller_phone,
    }

    session = AgentSession(
        llm=realtime_model,
        tts=google.TTS(
            language="hi-IN",
            voice_name="hi-IN-Chirp3-HD-Aoede",
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        ),
        userdata=_userdata,
        turn_handling={
            "endpointing": {
                # Was 2.5s — this produced noticeably long dead-air pauses before
                # every response, most visible right before verification questions
                # (phone/name confirmation). Lowered for a snappier, more natural
                # back-and-forth; interruption handling below still protects
                # against cutting the caller off mid-sentence.
                "min_delay": 0.5,
            },
            "interruption": {
                "min_duration": 0.3,
            }
        }
    )
    _userdata["session"] = session

    from call_recording import CallSessionRecorder, derive_call_id
    call_id = derive_call_id(ctx.room, user_id)
    recorder = CallSessionRecorder(ctx, session, call_id, user_id, transcript)
    recorder.attach()

    last_speech_time = time.monotonic()   # last time someone actually SPOKE (audio)
    last_activity_time = time.monotonic()  # last time any state change happened
    silence_count = 0
    agent_asked_this_turn = False
    thinking_since = 0.0                   # when agent entered "thinking" state (0 = not thinking)

    # Local copies of user/agent state — updated by events.
    # The silence monitor reads these instead of calling session.user_state /
    # session.agent_state directly, which are NOT properties on AgentSession and
    # throw AttributeError, causing the broad except-break to silently kill the
    # monitor on the very first loop iteration.
    _current_u_state = "listening"
    _current_a_state = "listening"

    @session.on("user_state_changed")
    def on_user_state(event):
        nonlocal last_speech_time, last_activity_time, silence_count, agent_asked_this_turn, thinking_since, _current_u_state
        logger.info(f"User state changed: {event.old_state} -> {event.new_state}")
        last_activity_time = time.monotonic()
        _current_u_state = event.new_state
        if event.new_state == "speaking":
            last_speech_time = time.monotonic()
            silence_count = 0
            agent_asked_this_turn = False
            thinking_since = 0.0
        elif event.old_state == "speaking":
            # User just STOPPED speaking — silence starts NOW, not earlier.
            last_speech_time = time.monotonic()

    @session.on("agent_state_changed")
    def on_agent_state(event):
        nonlocal last_speech_time, last_activity_time, silence_count, agent_asked_this_turn, thinking_since, _current_a_state
        logger.info(f"Agent state changed: {event.old_state} -> {event.new_state}")
        last_activity_time = time.monotonic()
        _current_a_state = event.new_state
        if event.new_state == "speaking":
            # Agent is producing audio — real speech, reset the silence timer
            last_speech_time = time.monotonic()
            silence_count = 0
            thinking_since = 0.0
            if not agent_asked_this_turn:
                current_slot = state.current_step()
                if current_slot not in ("complete", "confirmation"):
                    state.increment_ask(current_slot)
                    logger.info(f"Incremented ask count for {current_slot}: {state.get_ask_count(current_slot)}")
                agent_asked_this_turn = True
        elif event.old_state == "speaking":
            # Agent just STOPPED speaking — silence starts NOW.
            # Also MUST reset thinking_since here so that the monitor's
            # thinking check (thinking_since > 0) doesn't permanently
            # block the silence check in subsequent idle periods.
            last_speech_time = time.monotonic()
            thinking_since = 0.0
        elif event.new_state == "thinking":
            # Agent is processing but NOT producing audio.
            if thinking_since == 0.0:
                thinking_since = time.monotonic()
        else:
            # Agent went to idle/listening — no longer thinking
            thinking_since = 0.0

    async def silence_monitor():
        nonlocal last_speech_time, last_activity_time, silence_count, thinking_since, _current_u_state, _current_a_state

        _SILENCE_TIMEOUT = 10.0           # seconds of user silence before first check-in
        _FOLLOWUP_TIMEOUT = 10.0          # seconds before second check-in
        _THINKING_FILLER_TIMEOUT = 3.0    # seconds before injecting a thinking filler

        _thinking_fillers = [
            "Ek second...",
            "Haan ji, check kar rahi hoon...",
            "Dekhti hoon...",
        ]
        _filler_idx = 0
        _SILENCE_GOODBYE = (
            "Lagta hai connection ya audio mein koi issue hai. "
            "Jab aap ready hon, dobara baat karte hain. Dhanyavaad!"
        )

        # Guard: don't count silence until the agent has completed its FIRST
        # speaking turn (i.e., the greeting). Before that, the model may take
        # several seconds to generate the opening line, which is NOT user silence.
        _greeting_delivered = False

        async def _hangup_due_to_silence():
            logger.info("Caller unresponsive after silence check-ins — ending call.")
            _disconnect_event.set()
            for task in list(_background_tasks):
                if task is not asyncio.current_task() and not task.done():
                    task.cancel()
            try:
                await session.aclose()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            try:
                # rtc.Room.disconnect() is a coroutine — the old code never
                # awaited it, so the room was NEVER actually disconnected from our
                # side, which is exactly what lets the LiveKit server force-close
                # the data channels ~20s later and trigger the webrtc-sys panic.
                _maybe = ctx.room.disconnect()
                if inspect.isawaitable(_maybe):
                    await _maybe
            except Exception as e:
                logger.warning(f"Error disconnecting room: {e}")
            try:
                ctx.shutdown("caller_unresponsive")
            except Exception as e:
                logger.warning(f"Error shutting down: {e}")

        _diag_tick = 0

        try:
            while True:
                await asyncio.sleep(0.5)

                if _disconnect_event.is_set():
                    logger.info("Silence monitor: stopping (disconnect).")
                    break

                u = _current_u_state
                a = _current_a_state
                elapsed = time.monotonic() - last_speech_time

                # --- Diagnostic log every 5 seconds ---
                _diag_tick += 1
                if _diag_tick % 10 == 0:
                    logger.info(
                        f"[SilenceMonitor] u={u} a={a} elapsed={elapsed:.1f}s "
                        f"silence_count={silence_count} thinking_since={thinking_since:.1f}"
                    )

                # ---- 1. User speaking — reset timer --------------------------
                if u == "speaking":
                    last_speech_time = time.monotonic()
                    silence_count = 0
                    continue

                # ---- 2. Agent speaking — reset timer, mark greeting done -----
                if a == "speaking":
                    _greeting_delivered = True
                    last_speech_time = time.monotonic()
                    silence_count = 0
                    continue

                # ---- 3. Agent THINKING — model is generating a response ------
                # This is NOT user silence: the agent received the user's speech
                # and is now processing it. NEVER fire the silence check here —
                # doing so injects a check-in phrase right as the model is
                # generating its real reply, corrupting the output.
                # If thinking takes too long (>3s), inject a brief filler so
                # the caller hears something — but still don't count as silence.
                if a == "thinking":
                    if thinking_since > 0:
                        thinking_elapsed = time.monotonic() - thinking_since
                        if thinking_elapsed >= _THINKING_FILLER_TIMEOUT:
                            logger.info(
                                f"Agent stuck thinking {thinking_elapsed:.1f}s — injecting filler"
                            )
                            try:
                                filler = _thinking_fillers[_filler_idx % len(_thinking_fillers)]
                                _filler_idx += 1
                                session.say(filler, allow_interruptions=True)
                            except Exception as e:
                                logger.warning(f"Thinking filler error: {e}")
                            thinking_since = time.monotonic()
                    # ALWAYS skip silence check while model is thinking.
                    continue

                # ---- 4. Before greeting — don't count yet --------------------
                # The model start-up / first-generation delay is not user silence.
                if not _greeting_delivered:
                    last_speech_time = time.monotonic()
                    continue

                # ---- 5. True silence check ------------------------------------
                # Agent is in "listening" state (idle), user is not speaking,
                # greeting has been delivered. Now check elapsed silence.
                required = _SILENCE_TIMEOUT if silence_count == 0 else _FOLLOWUP_TIMEOUT

                if elapsed < required:
                    continue

                if _disconnect_event.is_set():
                    break

                # Silence threshold reached — fire check-in or goodbye
                if silence_count < 2:  # Max 2 check-ins before hanging up
                    logger.info(
                        f"[SilenceMonitor] Check-in #{silence_count + 1} after "
                        f"{elapsed:.1f}s of silence."
                    )
                    
                    # Instead of hardcoded strings, we instruct the LLM to analyze the
                    # conversation and dynamically generate a contextual follow-up.
                    # We avoid using verbatim quotes in the instruction so the model
                    # doesn't simply echo them out loud.
                    prompt_instruction = (
                        f"[SYSTEM DIRECTIVE] The user has been completely silent for {elapsed:.1f} seconds. "
                        "Please analyze the current conversation context and proactively ask a brief, relevant "
                        "follow-up question in Hindi to check if they are still on the line. "
                        "Do NOT repeat this directive. Just speak naturally as the receptionist."
                    )
                    
                    try:
                        session.generate_reply(instructions=prompt_instruction)
                    except Exception as e:
                        logger.warning(f"[SilenceMonitor] generate_reply() failed: {e}")

                    silence_count += 1
                    last_speech_time = time.monotonic()

                else:
                    # Both check-ins unanswered — end the call politely
                    logger.info("[SilenceMonitor] No response after two check-ins — ending call.")
                    try:
                        session.say(_SILENCE_GOODBYE, allow_interruptions=False)
                    except Exception as e:
                        logger.warning(f"Goodbye say failed: {e}")
                    await asyncio.sleep(6.0)
                    await _hangup_due_to_silence()
                    break

        except asyncio.CancelledError:
            logger.info("Silence monitor cancelled.")


    # Fetch live hospital data immediately when call connects
    try:
        await fetch_live_hospital_data()
    except Exception as e:
        logger.warning("Failed initial hospital data fetch: %s", e)

    # Use close_on_disconnect=False to take manual control of the shutdown
    # sequence. The default (True) tears down the RTC session immediately when
    # the SIP caller disconnects, which races with the recording/transcript
    # cleanup and triggers the webrtc-sys "malformed serialized RtcError" panic.
    await session.start(
        room=ctx.room,
        agent=HospitalReceptionistAgent(
            chat_ctx=ChatContext(),
            state=state,
            userdata=_userdata,
            instructions=_human_voice_instructions,
        ),
        room_input_options=RoomInputOptions(
            close_on_disconnect=False,
        ),
    )

    if session.current_agent:
        await update_agent_instructions(session.current_agent, state)

    # ── Graceful disconnect handling ──────────────────────────────────────────
    # We manually orchestrate shutdown so that cleanup tasks (recording upload,
    # transcript saving) finish BEFORE the RTC peer connection is torn down.

    _disconnect_event = asyncio.Event()
    _background_tasks: list[asyncio.Task] = []

    async def _graceful_disconnect(participant: rtc.RemoteParticipant) -> None:
        """Orchestrate the correct shutdown sequence on caller disconnect.
        
        Order matters critically:
          1. Block outgoing audio       (prevents frames reaching closing RTC)
          2. Cancel background tasks     (silence_monitor, fetch)
          3. Close agent session         (triggers recorder _on_shutdown → saves files)
          4. Disconnect from room        (cleanly closes peer connection from OUR side)
          5. Signal framework shutdown   (lets the agents framework exit cleanly)
        
        Without step 4, the LiveKit server force-closes the data channels ~20s later,
        which triggers the Rust panic in webrtc-sys ("malformed serialized RtcError").
        """
        identity = getattr(participant, "identity", "unknown")
        logger.info("Graceful disconnect handler started for participant: %s", identity)
        _disconnect_event.set()

        # 1. Cancel background tasks that interact with the session — including the
        #    per-turn tasks the agent spawns (hospital refresh / instruction update),
        #    which were previously untracked and could keep running against a closed
        #    session.
        agent_bg = list(getattr(session.current_agent, "_bg_tasks", ()) or ())
        for task in list(_background_tasks) + agent_bg:
            if not task.done():
                task.cancel()
        _all_bg = list(_background_tasks) + agent_bg
        if _all_bg:
            await asyncio.gather(*_all_bg, return_exceptions=True)
            logger.info("Background tasks cancelled during disconnect.")

        # 1b. Release any slot this caller locked in verify_availability but never
        #     committed (booking_status != BOOKED). Otherwise the lock sits on the
        #     doctor+date+time for its full 24h TTL and blocks every other caller
        #     from that slot even though this caller hung up.
        try:
            if state.availability_verified and state.booking_status != "BOOKED" \
                    and state.doctor_name and state.appointment_date and state.appointment_time:
                await release_appointment_slot(
                    doctor_name=state.doctor_name,
                    date=state.appointment_date,
                    time_slot=state.appointment_time,
                    user_id=_userdata.get("session_id", user_id),
                )
                logger.info("Released uncommitted slot lock on disconnect.")
        except Exception as e:
            logger.warning("Failed to release uncommitted slot lock on disconnect: %s", e)

        # 2. Small buffer for in-flight capture_frame calls to observe the
        #    per-recorder shutdown flag set by the recorder's disconnect handler.
        await asyncio.sleep(0.3)

        # 3. Close the agent session. This triggers the shutdown callbacks
        #    (including the recorder's _on_shutdown) which persist audio/transcript.
        try:
            await session.aclose()
            logger.info("Agent session closed gracefully.")
        except Exception as e:
            logger.warning("Error closing agent session: %s", e)

        # 4. Disconnect from the room IMMEDIATELY. This cleanly closes the
        #    WebRTC peer connection from our side, preventing the LiveKit server
        #    from forcibly tearing down the data channels ~20s later (which is
        #    what causes the webrtc-sys "malformed serialized RtcError" panic).
        try:
            # rtc.Room.disconnect() is a coroutine and MUST be awaited — without
            # this, step 4 was a no-op and the whole "disconnect before the server
            # tears us down" panic-mitigation this handler is built around never
            # actually ran.
            _maybe = ctx.room.disconnect()
            if inspect.isawaitable(_maybe):
                await _maybe
            logger.info("Disconnected from room cleanly.")
        except Exception as e:
            logger.warning("Error disconnecting from room: %s", e)

        # 5. Signal the agents framework to shut down this job.
        try:
            ctx.shutdown("participant_disconnected")
            logger.info("Job context shutdown signalled.")
        except Exception as e:
            logger.warning("Error signalling job shutdown: %s", e)

        logger.info("Graceful disconnect handler completed for participant: %s", identity)

    @ctx.room.on("participant_disconnected")
    def _on_room_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        """Room-level disconnect handler that kicks off graceful shutdown."""
        asyncio.ensure_future(_graceful_disconnect(participant))

    # Start the proactive silence monitor task (tracked for cancellation)
    _silence_task = asyncio.create_task(silence_monitor())
    _background_tasks.append(_silence_task)

    _greeting = (
        "You are Nikita, the hospital receptionist. "
        "Greet the caller politely in Hindi by saying exactly: 'Namaste, main Nikita bol rahi hoon Arora Hospital se. Main aapki kaise madad kar sakti hoon?' "
        "Wait for their response."
    )

    await asyncio.sleep(0.2)
    session.generate_reply(instructions=_greeting)


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="hospital-receptionist",
            load_threshold=0.70,
            num_idle_processes=4,
        )
    )