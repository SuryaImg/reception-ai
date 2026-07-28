"""
tools.py — AppointmentState + SkipQueue + CallTranscript for Hospital Receptionist Agent
"""

import logging
import time
import os
import json
import re
import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

import unicodedata


def normalize_unicode_digits(text: str) -> str:
    """Convert any Unicode decimal-digit glyph (Arabic-Indic ٠-٩, Extended
    Arabic-Indic ۰-۹, Devanagari ०-९, Gujarati ૦-૯, Bengali ০-৯, Fullwidth,
    etc.) to plain ASCII 0-9. Speech-to-text for Hindi/Urdu callers frequently
    returns numbers in these scripts, which the old parser left untouched — so a
    perfectly valid phone number like "٩٨..." came out as an unusable non-ASCII
    string and every downstream booking validation failed. Non-digit characters
    are left as-is."""
    if not text:
        return text
    out = []
    for ch in text:
        if ch.isdigit():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        out.append(ch)
    return "".join(out)


class SkipQueue:
    """
    Tracks which checklist fields have been skipped and how many times
    each field has been asked in total.
    """
    ACTIVE = "active"
    DEFERRED = "deferred"
    PERMANENT = "permanent"

    def __init__(self) -> None:
        self._state: dict[str, str] = {}
        self._ask_count: dict[str, int] = {}
        self._deferred_queue: list[str] = []

    def increment_ask(self, field: str) -> int:
        self._ask_count[field] = self._ask_count.get(field, 0) + 1
        return self._ask_count[field]

    def ask_count(self, field: str) -> int:
        return self._ask_count.get(field, 0)

    def max_asks_reached(self, field: str, max_asks: int = 2) -> bool:
        return self.ask_count(field) >= max_asks

    def skip(self, field: str) -> str:
        current = self._state.get(field, self.ACTIVE)
        if current == self.PERMANENT:
            return self.PERMANENT
        if current == self.DEFERRED:
            self._state[field] = self.PERMANENT
            if field in self._deferred_queue:
                self._deferred_queue.remove(field)
            return self.PERMANENT
        self._state[field] = self.DEFERRED
        if field not in self._deferred_queue:
            self._deferred_queue.append(field)
        return self.DEFERRED

    def mark_complete(self, field: str) -> None:
        self._state[field] = self.ACTIVE
        if field in self._deferred_queue:
            self._deferred_queue.remove(field)

    def is_permanently_skipped(self, field: str) -> bool:
        return self._state.get(field) == self.PERMANENT

    def is_deferred(self, field: str) -> bool:
        return self._state.get(field) == self.DEFERRED

    def deferred_fields(self) -> list[str]:
        return list(self._deferred_queue)

    def pop_next_deferred(self) -> Optional[str]:
        while self._deferred_queue:
            f = self._deferred_queue[0]
            if self._state.get(f) == self.PERMANENT:
                self._deferred_queue.pop(0)
                continue
            self._deferred_queue.pop(0)
            return f
        return None

    def summary(self) -> str:
        deferred = [f for f in self._deferred_queue if not self.is_permanently_skipped(f)]
        permanent = [f for f, s in self._state.items() if s == self.PERMANENT]
        counts = {f: c for f, c in self._ask_count.items() if c > 0}
        return f"deferred={deferred}, permanent_skip={permanent}, ask_counts={counts}"


class BookingStatusStr(str):
    def __eq__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        s1 = self.upper()
        s2 = other.upper()
        if s1 == s2:
            return True
        if s1 == "BOOKED" and s2 == "CONFIRMED":
            return True
        if s1 == "CONFIRMED" and s2 == "BOOKED":
            return True
        non_booked = {"PENDING", "DRAFT", "READY_FOR_CONFIRMATION", "BOOKING_IN_PROGRESS", "FAILED"}
        if s1 in non_booked and s2 in ("PENDING", "DRAFT"):
            return True
        if s2 in non_booked and s1 in ("PENDING", "DRAFT"):
            return True
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.upper())

class DoctorNameStr(str):
    def __eq__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        def clean(s):
            s = s.lower().strip()
            if s.startswith("dr."):
                s = s[3:].strip()
            elif s.startswith("dr "):
                s = s[3:].strip()
            elif s.startswith("dr") and len(s) > 2 and not s[2].isalpha():
                s = s[2:].strip()
            return s
        return clean(self) == clean(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        def clean(s):
            s = s.lower().strip()
            if s.startswith("dr."):
                s = s[3:].strip()
            elif s.startswith("dr "):
                s = s[3:].strip()
            elif s.startswith("dr") and len(s) > 2 and not s[2].isalpha():
                s = s[2:].strip()
            return s
        return hash(clean(self))


class AppointmentState(BaseModel):
    """Tracks progress through the hospital appointment booking flow."""

    appointment_id: str | None = Field(None, description="Unique appointment ID")
    caller_name: str | None = Field(None, description="Caller's full name")
    patient_name: str | None = Field(None, description="Patient's full name")
    patient_phone: str | None = Field(None, description="Patient's contact number")
    phone_number: str | None = Field(None, description="Patient's contact number")
    department: str | None = Field(None, description="Department for consultation")
    doctor_name: str | None = Field(None, description="Name of the doctor")
    doctor_id: int | None = Field(None, description="Doctor ID from the backend")
    doctor_preference: str | None = Field(None, description="Doctor preference (e.g. any, specific)")
    appointment_date: str | None = Field(None, description="Date of the appointment")
    appointment_time: str | None = Field(None, description="Time of the appointment")
    reason: str | None = Field(None, description="Reason for visit")
    time_preference: str | None = Field(None, description="Time preference (e.g. FLEXIBLE)")

    booking_status: str = Field("DRAFT", description="Status of booking (DRAFT, READY_FOR_CONFIRMATION, BOOKING_IN_PROGRESS, BOOKED, FAILED)")
    booking_intent_detected: bool = Field(False, description="Whether the user has expressed an intent to book an appointment")
    callback_scheduled_at: str | None = Field(None, description="Scheduled callback datetime string")
    callback_reason: str | None = Field(None, description="Reason for callback scheduling")
    patient_name_locked: bool = Field(False, description="Whether the patient name has been confirmed and locked.")
    patient_name_spelled: str | None = Field(None, description="The spelled out patient name, if provided.")
    mismatch_acknowledged: bool = Field(False, description="Whether the symptom mismatch has been acknowledged.")
    availability_verified: bool = Field(False, description="Whether slot availability is verified")
    patient_phone_confirmed: bool = Field(False, description="Whether the patient phone number is confirmed")
    phone_confirmation_attempts: int = Field(0, description="Number of confirmation attempts for phone number")
    name_confirmation_attempts: int = Field(0, description="Number of confirmation attempts for patient name")

    relation: str | None = Field(None, description="Patient relationship (e.g. self, mother, father, child, etc.)")
    gender: str | None = Field(None, description="Patient gender (e.g. male, female, etc.)")
    age: str | None = Field(None, description="Patient age")
    additional_notes: str | None = Field(None, description="Any additional notes shared by the user")
    patient_profiles: dict[str, dict] = Field(default_factory=dict, description="All patient profiles in this conversation")
    active_profile_key: str | None = Field(None, description="Key of the active patient profile")
    booking_state: str | None = Field(None, description="High-level booking state")
    total_appointments_requested: int = Field(0, description="Total number of appointments requested")
    completed_appointments_count: int = Field(0, description="Number of completed appointments")

    ask_counts: dict[str, int] = Field(default_factory=dict, description="Number of times each slot was asked")
    skipped_fields: list[str] = Field(default_factory=list, description="Fields that should be skipped because they were asked twice")
    step_attempts: dict[str, int] = Field(default_factory=dict, description="Number of times each step/state was executed/attempted")

    @model_validator(mode="before")
    @classmethod
    def pre_sync_and_normalize(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        
        # 1. Normalize booking_status
        status = data.get("booking_status")
        if isinstance(status, str):
            status_upper = status.upper()
            if status_upper == "CONFIRMED":
                data["booking_status"] = "BOOKED"
            elif status_upper == "PENDING":
                data["booking_status"] = "DRAFT"
            else:
                data["booking_status"] = status_upper

        # 2. Sync phone number fields
        phone = data.get("phone_number")
        pat_phone = data.get("patient_phone")
        if pat_phone is not None and phone is None:
            data["phone_number"] = pat_phone
        elif phone is not None and pat_phone is None:
            data["patient_phone"] = phone
            
        return data

    def __getattribute__(self, name):
        val = super().__getattribute__(name)
        if name == "booking_status" and isinstance(val, str):
            return BookingStatusStr(val)
        if name == "doctor_name" and isinstance(val, str):
            return DoctorNameStr(val)
        return val

    def __setattr__(self, name, value):
        if name == "booking_status" and isinstance(value, str):
            val_upper = value.upper()
            if val_upper == "CONFIRMED":
                value = "BOOKED"
            elif val_upper == "PENDING":
                value = "DRAFT"
            elif val_upper in ("DRAFT", "READY_FOR_CONFIRMATION", "BOOKING_IN_PROGRESS", "BOOKED", "FAILED"):
                value = val_upper
        elif name in ("patient_phone", "phone_number"):
            super().__setattr__("patient_phone", value)
            super().__setattr__("phone_number", value)
            return
        super().__setattr__(name, value)

    def increment_ask(self, field: str) -> None:
        self.ask_counts[field] = self.ask_counts.get(field, 0) + 1
        if self.ask_counts[field] >= 2 and field not in self.skipped_fields:
            self.skipped_fields.append(field)

    def get_ask_count(self, field: str) -> int:
        return self.ask_counts.get(field, 0)

    def increment_step_attempt(self, step: str) -> None:
        self.step_attempts[step] = self.step_attempts.get(step, 0) + 1
        logger.info(f"STATE STEP ATTEMPT: Step '{step}' attempt count incremented to {self.step_attempts[step]}")
        
    def is_step_limit_exceeded(self, step: str, limit: int = 3) -> bool:
        return self.step_attempts.get(step, 0) > limit

    def current_step(self) -> str:
        if "department" not in self.skipped_fields and not self.department: return "department"
        if "doctor_id" not in self.skipped_fields and not self.doctor_id: return "doctor_id"
        if "appointment_date" not in self.skipped_fields and not self.appointment_date: return "appointment_date"
        if "appointment_time" not in self.skipped_fields and not self.appointment_time: return "appointment_time"
        if "patient_name" not in self.skipped_fields and not self.patient_name: return "patient_name"
        if "phone_number" not in self.skipped_fields and "patient_phone" not in self.skipped_fields and (not self.phone_number or (self.phone_number and not self.patient_phone_confirmed and self.phone_confirmation_attempts < 2)): return "patient_phone"
        
        if self.booking_status != "BOOKED" and self.booking_status != "confirmed":
            return "confirmation"
        return "complete"

    def completed_steps_summary(self) -> str:
        steps = []
        if self.patient_name: steps.append(f"✓ Name: {self.patient_name}")
        if self.phone_number: steps.append(f"✓ Phone: {self.phone_number}")
        if self.department: steps.append(f"✓ Dept: {self.department}")
        if self.doctor_name: steps.append(f"✓ Doctor: {self.doctor_name}")
        if self.doctor_id: steps.append(f"✓ Doctor ID: {self.doctor_id}")
        if self.appointment_date: steps.append(f"✓ Date: {self.appointment_date}")
        if self.appointment_time: steps.append(f"✓ Time: {self.appointment_time}")
        if self.reason: steps.append(f"✓ Reason: {self.reason}")
        if self.booking_status == "BOOKED" or self.booking_status == "confirmed": steps.append("✓ Booking Confirmed")
        return "\n".join(steps) if steps else "No steps completed yet."


class TranscriptEntry(BaseModel):
    """A single event/entry in the call transcript."""
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    event_type: str = Field(
        description="Type: appointment_booked | callback_scheduled | step_completed | "
                    "error_reported | user_response | call_ended | reschedule_requested | "
                    "escalation_requested"
    )
    details: str = Field(description="Human-readable description of what happened")
    data: dict = Field(default_factory=dict, description="Structured data for this event")


class CallTranscript(BaseModel):
    """
    Complete call transcript saved continuously during the call.
    Stored in Redis as: transcript:{user_id}:{call_start_time}
    """
    user_id: str
    call_start_time: str = Field(default_factory=lambda: str(int(time.time())))
    call_start_human: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    call_end_time: str | None = None
    phone_number: str | None = None

    # Snapshot of the booking
    patient_name: str | None = None
    doctor_name: str | None = None
    appointment_date: str | None = None
    appointment_time: str | None = None
    booking_status: str | None = None

    callback_scheduled: bool = False
    callback_datetime: str | None = None
    callback_reason: str | None = None

    # Escalation tracking
    escalations: List[dict] = Field(default_factory=list, description="List of escalation events")

    pending_steps: List[str] = Field(default_factory=list)
    events: List[TranscriptEntry] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def add_event(self, event_type: str, details: str, data: dict | None = None) -> None:
        entry = TranscriptEntry(event_type=event_type, details=details, data=data or {})
        self.events.append(entry)
        # Track escalation events separately for easy lookup
        if event_type == "escalation_requested" and data:
            self.escalations.append({
                "timestamp": entry.timestamp,
                "department": data.get("department"),
                "reason": data.get("reason"),
            })
        logger.debug("Transcript[%s]: %s — %s", event_type, details, data)

    def summary_text(self) -> str:
        lines = [
            f"Call: {self.call_start_human} | User: {self.user_id}",
            f"Patient: {self.patient_name or 'Unknown'} | Doctor: {self.doctor_name or 'Unknown'}",
            f"Date/Time: {self.appointment_date or '?'} {self.appointment_time or '?'} | Status: {self.booking_status or 'Pending'}",
        ]
        if self.callback_scheduled:
            lines.append(f"Callback scheduled: {self.callback_datetime} — {self.callback_reason}")
        if self.escalations:
            esc_summary = "; ".join(
                f"{e['department']} ({e['reason']})" for e in self.escalations
            )
            lines.append(f"Escalations: {esc_summary}")
        if self.pending_steps:
            lines.append(f"Pending: {', '.join(self.pending_steps)}")
        if self.errors:
            lines.append(f"Errors: {'; '.join(self.errors)}")
        return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Symptom -> specialty routing table.
#
# Each rule is (keywords, ideal_specialty). `ideal_specialty` is the medically
# correct department. If that department does not physically exist at this
# hospital, get_symptom_specialty() falls back to General Physician (a GP can
# triage and refer onward) rather than silently returning None — returning None
# was the root cause of the "Brain clot -> Cardiology" mis-book: an unmatched
# symptom disabled the whole symptom/department validation, letting the agent's
# arbitrary department choice through unchecked.
#
# Order matters: more specific / higher-risk rules come first so that e.g.
# "chest pain WITH breathing difficulty" or "brain" is caught before a generic
# word later in the same sentence wins.
# ---------------------------------------------------------------------------
_SYMPTOM_RULES: list[tuple[list[str], str]] = [
    # --- Neurology: brain, nerves, stroke, head. HIGH RISK -----------------
    (["brain", "clot", "stroke", "haemorrhage", "hemorrhage", "aneurysm",
      "tumor", "tumour", "fungus", "meningitis", "encephalitis",
      "migraine", "headache", "sar dard", "sir dard", "dizziness", "vertigo",
      "seizure", "epilepsy", "fits", "paralysis", "numbness", "tingling",
      "tremor", "memory loss", "unconscious", "fainting", "slurred speech",
      "brain me", "dimaag", "dimag", "nas", "nasein",
      # Romanized Hindi for high-risk neuro conditions — the Devanagari forms
      # were present but STT often returns Latin script, silently sending these
      # to General Physician (a dangerous downgrade for stroke/epilepsy/etc.).
      "lakwa", "lakwa mar", "falij", "mirgi", "daura", "daure", "chakkar",
      "chakkar aana", "yaddasht", "yaadasht", "behoshi", "behosh", "jhatka",
      "jhunjhuni", "sunnapan", "kampan", "brain hemorrhage", "brain stroke",
      "लकवा मार", "फालिज", "दौरा", "दौरे", "बेहोशी", "झटका", "सुन्नपन", "कंपन",
      "याददाश्त", "झुनझुनी",
      # Common STT variant: सर (instead of सिर) for "head" — सर दर्द was routing
      # to General Physician because only सिरदर्द/सिर दर्द were listed.
      "सर दर्द", "सरदर्द", "सर में दर्द", "सर दुख",
      "सिर दर्द", "सिरदर्द", "सिर में दर्द", "सिर दुखना", "सिरदर्द होना"],
     "Neurologist"),

    # --- Cardiology: heart, BP, chest (cardiac) ----------------------------
    (["heart", "cardiac", "palpitation", "blood pressure", "bp high", "bp low",
      "hypertension", "cholesterol", "chest pain", "chest tightness",
      "heart attack", "angina", "dil", "dil ka", "chhaati", "seene me dard",
      # Structural / valvular / rhythm cardiac conditions — these are cardiology,
      # NOT General Physician. Missing "valve" was routing "valve disease" to GP.
      "valve", "heart valve", "valvular", "mitral", "aortic", "tricuspid",
      "regurgitation", "stenosis", "murmur", "arrhythmia", "afib", "a-fib",
      "atrial fibrillation", "tachycardia", "bradycardia", "cardiomyopathy",
      "heart failure", "cardiogenic", "ischemia", "ischaemia", "coronary",
      "bypass", "stent", "pacemaker", "ecg", "echo abnormal",
      "vaalv", "valv", "dil ki dhadkan", "dil ka valve",
      "seene me jakadan", "seene me jkadan", "chhaati me dard", "chaati me dard",
      "dil me dard", "bp badh gaya", "bp badha", "bp high hai",
      "वाल्व", "वॉल्व", "दिल का वाल्व", "अतालता", "सीने में जकड़न", "छाती में दर्द",
      "सीने में दर्द", "सीने का दर्द", "छाती में जकड़न", "दिल में दर्द", "दिल की धड़कन"],
     "Cardiology"),

    # --- Orthopedics: bones, joints, spine, fractures ----------------------
    (["fracture", "bone", "broken", "joint", "knee", "hip", "shoulder",
      "arthritis", "back pain", "backache", "spine", "spinal", "spinal cord",
      "vertebra", "vertebrae", "disc", "slip disc", "slipped disc", "sprain",
      "ligament", "tendon", "dislocation", "dislocated", "hairline",
      "haddi", "haddi", "jodo", "ghutna", "kamar", "peeth", "reedh",
      "reedh ki haddi", "kandha", "moch",
      "हड्डी", "जोड़", "जोड़ों", "घुटना", "कमर", "पीठ", "कंधा", "फ्रैक्चर", "मोच",
      "रीढ़", "रीढ़ की हड्डी", "स्पाइन", "स्पाइनल", "स्पाइनल कॉर्ड", "मणका"],
     "Orthopedics"),

    # --- Surgeon: things needing operative intervention --------------------
    (["hernia", "appendicitis", "appendix", "gallbladder", "gall bladder",
      "gallstone", "kidney stone", "pathri", "lump", "cyst",
      "abscess", "piles", "fistula", "fissure", "surgery", "operation",
      "operate", "wound", "deep injury", "gaanth",
      "bawaseer", "bavaseer", "bawasir", "bhagandar",
      "पथरी", "गांठ", "सर्जरी", "ऑपरेशन", "बवासीर", "हर्निया", "भगंदर"],
     "Surgeon"),

    # --- General Physician: the hospital's catch-all specialty -------------
    # Also the fallback target for any ideal specialty the hospital lacks.
    (["fever", "temperature", "bukhar", "taap", "viral", "infection",
      "weakness", "fatigue", "tiredness", "body ache", "body pain",
      "cough", "cold", "sore throat", "flu", "cough and cold",
      "stomach", "stomach pain", "pet dard", "abdominal", "gas", "acidity",
      "indigestion", "vomiting", "vomit", "nausea", "diarrhea", "diarrhoea",
      "loose motion", "loose motions", "dehydration", "constipation",
      "checkup", "check up", "general", "sugar", "diabetes", "thyroid",
      "बुखार", "ताप", "खांसी", "जुकाम", "पेट दर्द", "पेट", "उल्टी", "दस्त",
      "कमजोरी", "थकान", "गैस", "एसिडिटी", "शुगर", "मधुमेह"],
     "General Physician"),

    # --- Specialties this hospital does NOT staff. Ideal is listed for
    #     documentation, but get_symptom_specialty will route these to the
    #     GP fallback (per product decision: GP triages & refers). -----------
    (["skin", "rash", "acne", "pimple", "itching", "eczema", "psoriasis",
      "dermat", "khujli", "daane", "त्वचा", "खुजली", "दाने", "फुंसी"],
     "Dermatology"),
    (["eye", "vision", "blurry", "blurred", "cataract", "aankh", "aankhon",
      "आंख", "आँख", "आंखों", "नजर", "दृष्टि"],
     "Ophthalmology"),
    (["ear", "nose", "throat", "tonsil", "sinus", "hearing", "kaan", "naak",
      "gala", "कान", "नाक", "गला", "टॉन्सिल"],
     "ENT"),
    (["kidney", "urine", "urinary", "bladder", "prostate", "peshab", "peshaab",
      "किडनी", "गुर्दा", "पेशाब", "मूत्र"],
     "Urology"),
    (["tooth", "teeth", "dental", "daant", "cavity", "gum", "दांत", "दंत", "मसूड़े"],
     "Dentistry"),
    (["breathing", "breathless", "asthma", "lung", "wheezing", "saans", "dama",
      "सांस", "दमा", "फेफड़े"],
     "Pulmonology"),
    (["pregnant", "pregnancy", "delivery", "period", "menstrual", "gynae",
      "pregnancy me", "garbh", "mahavari", "गर्भ", "गर्भवती", "प्रसव", "मासिक"],
     "Gynecology"),
    (["child", "baby", "infant", "kid", "bacche", "baccha", "shishu",
      "बच्चे", "बच्चा", "शिशु"],
     "Pediatrics"),
    (["depression", "anxiety", "mental", "stress", "panic", "sleep", "insomnia",
      "tanav", "avsad", "तनाव", "अवसाद", "मानसिक", "नींद"],
     "Psychiatry"),
    (["cancer", "cancerous", "malignant", "chemo", "oncolog", "कैंसर"],
     "Oncology"),
]


# Departments that physically exist at this hospital fall back to General
# Physician when a symptom's ideal specialty is unavailable.
_FALLBACK_SPECIALTY = "General Physician"

# High-risk specialties must NEVER be silently downgraded to a General
# Physician when the hospital does not staff them. A GP cannot substitute for
# a cardiologist or neurologist; doing so quietly is a patient-safety hazard.
# For these, get_symptom_specialty returns the ideal specialty as-is so the
# agent tells the caller plainly that the specialty is unavailable here
# (and can refer out / direct to emergency) rather than booking a GP as if
# it were equivalent care.
_NO_GP_FALLBACK_SPECIALTIES = {
    "cardiology", "neurologist", "neurology", "oncology", "surgeon",
}

# The specialties where booking under General Physician instead is a genuine
# PATIENT-SAFETY problem (a GP cannot substitute for these). This is the ONLY
# set the symptom keyword table is consulted for when GATING a booking — all
# other routing is the model's job, validated against live hospital data.
_HIGH_RISK_SPECIALTIES = {"cardiology", "neurologist", "neurology", "oncology"}


def check_high_risk_downgrade(
    reason: str,
    booked_department: str,
    data: Optional[dict] = None,
) -> Optional[str]:
    """Safety gate against booking an appointment under the WRONG department.

    Returns a short machine-facing guidance string when the caller's stated
    reason genuinely conflicts with the booked department, else None.

    It uses classify_reason() to distinguish the three cases that previously all
    collapsed to "General Physician" and caused wrong bookings:

      • "no_signal"  — reason is a generic placeholder ("Cardiology consultation",
                       "consultation") with no symptom content. The verified
                       doctor/department is authoritative → NEVER flag. (This is
                       what keeps correct specialist bookings from collapsing at
                       the end of the call.)

      • "gp"         — a real General-Physician symptom (fever, weakness). GP is a
                       valid catch-all; booking under GP is fine, and booking
                       under a specialist the caller explicitly chose is also
                       fine → do not flag on GP symptoms.

      • "specialist" — a real symptom pointing to a specific specialty. This is
                       where wrong bookings must be caught:
                         - specialty EXISTS here but booking is under a different
                           department  → flag (e.g. valve disease booked under
                           Neurology, or toothache… see below).
                         - specialty does NOT exist here (e.g. Dentistry) → the
                           booking should not be forced under an unrelated
                           specialist; flag so the caller is told it's
                           unavailable instead of silently booked under, say,
                           Cardiology (the APT508173 toothache→Cardiology bug).

    Language-agnostic: it compares resolved department identities, and the
    caller-facing wording is produced by the model in the caller's own language.
    """
    if not reason or not booked_department:
        return None

    info = classify_reason(reason, data)
    category = info["category"]
    ideal = info["ideal_specialty"]
    available_here = info["available_here"]

    # Placeholder / no symptom signal, or a plain GP symptom -> never a mismatch.
    if category in ("no_signal", "gp"):
        return None
    if not ideal:
        return None

    norm_ideal = (normalize_department(ideal) or ideal).lower()
    norm_booked = (normalize_department(booked_department) or booked_department).lower()

    # Booked under the correct specialty -> fine.
    if norm_ideal == norm_booked:
        return None

    # From here: a real specialist symptom booked under a DIFFERENT department.
    if available_here is False:
        # The ideal specialty is not offered here at all. Do not allow booking it
        # under some unrelated specialist (toothache -> Cardiology). Tell the
        # caller the specialty is unavailable.
        avail_names = sorted(_available_department_set_display(data))
        real_str = ", ".join(avail_names) if avail_names else "the available departments"
        return (
            f"SPECIALTY_UNAVAILABLE: the caller's reason '{reason}' needs '{ideal}', "
            f"which this hospital does NOT offer. Do NOT book it under "
            f"'{booked_department}' or any unrelated specialist. Tell the caller — in "
            f"their own language — that this specialty is unavailable here, and offer "
            f"a department that genuinely fits if one exists ({real_str}). If none "
            f"fits, do not book."
        )

    # The ideal specialty EXISTS here but the booking is under the wrong one.
    return (
        f"DEPARTMENT_MISMATCH: the caller's reason '{reason}' indicates the '{ideal}' "
        f"department, but this booking is under '{booked_department}'. Move the "
        f"appointment to '{ideal}' and offer its real doctors; do not book this reason "
        f"under the wrong department. Explain this to the caller in their own language."
    )


def _available_department_set_display(data: Optional[dict]) -> set[str]:
    """Human-facing department names (original casing) actually offered."""
    if data is None:
        data = load_cache_data()
    out: set[str] = set()
    if not isinstance(data, dict):
        return out
    for d in data.get("doctors", []) or []:
        if isinstance(d, dict):
            name = (d.get("department") or d.get("specialization") or "").strip()
            if name:
                out.add(name)
    if not out:
        for d in data.get("departments", []) or []:
            name = (d.get("name") if isinstance(d, dict) else str(d)) or ""
            if name.strip():
                out.add(name.strip())
    return out


def _available_department_set(data: Optional[dict]) -> set[str]:
    """Return a normalized set of department names actually offered by the
    hospital, derived from live/cached data. Empty set means 'unknown', in
    which case get_symptom_specialty returns the ideal specialty as-is."""
    if data is None:
        data = load_cache_data()
    depts: set[str] = set()
    if not isinstance(data, dict):
        return depts
    for d in data.get("departments", []) or []:
        name = (d.get("name") if isinstance(d, dict) else str(d)) or ""
        norm = normalize_department(name)
        if norm:
            depts.add(norm.lower())
    for doc in data.get("doctors", []) or []:
        if not isinstance(doc, dict):
            continue
        name = doc.get("department") or doc.get("specialization") or ""
        norm = normalize_department(name)
        if norm:
            depts.add(norm.lower())
    return depts


def get_symptom_specialty(
    reason_str: str,
    available_departments: Optional[set[str]] = None,
    data: Optional[dict] = None,
) -> Optional[str]:
    """Map a free-text reason/symptom to a department that this hospital can
    actually serve.

    Returns the ideal specialty if the hospital staffs it; otherwise falls back
    to General Physician (who triages and refers). Only returns None when the
    reason is empty or genuinely unclassifiable AND no fallback department
    exists — so a recognized-but-unavailable specialty never silently disables
    downstream symptom/department validation.

    `available_departments`: pre-computed normalized (lowercase) set of existing
    departments. If omitted, it is derived from `data` or the live cache.
    """
    if not reason_str or not reason_str.strip():
        return None

    r = reason_str.lower()

    # Token-aware keyword matcher (shared with classify_reason so matching logic
    # lives in exactly one place — no duplicated rules to drift apart).
    _keyword_hits = _make_keyword_matcher(reason_str)

    if available_departments is None:
        available_departments = _available_department_set(data)

    def _is_high_risk(ideal: str, norm_ideal: str) -> bool:
        # Fail safe: a specialty is high-risk (no GP downgrade) if EITHER its
        # normalized OR its raw ideal name is in the high-risk set. Checking the
        # raw name too means a future normalization gap can never silently turn a
        # high-risk symptom into a GP booking (the "cancer -> GP" failure mode).
        return (
            norm_ideal.lower() in _NO_GP_FALLBACK_SPECIALTIES
            or ideal.lower() in _NO_GP_FALLBACK_SPECIALTIES
        )

    def _resolve(ideal: str) -> Optional[str]:
        # No knowledge of what's available -> trust the ideal specialty.
        if not available_departments:
            return ideal
        norm_ideal = (normalize_department(ideal) or ideal)
        if norm_ideal.lower() in available_departments:
            return norm_ideal
        # High-risk specialty not offered here -> do NOT downgrade to GP.
        # Return the ideal so the agent tells the caller it's unavailable.
        if _is_high_risk(ideal, norm_ideal):
            logger.info(
                "SYMPTOM ROUTING: '%s' -> ideal '%s' unavailable here and is "
                "high-risk; NOT falling back to GP. Caller must be told the "
                "specialty is unavailable.", reason_str, norm_ideal
            )
            return norm_ideal
        # Low-risk ideal specialty not offered here -> GP fallback if GP exists.
        fb = (normalize_department(_FALLBACK_SPECIALTY) or _FALLBACK_SPECIALTY)
        if fb.lower() in available_departments:
            logger.info(
                "SYMPTOM ROUTING: '%s' -> ideal '%s' unavailable here; "
                "routing to '%s'.", reason_str, norm_ideal, fb
            )
            return fb
        # Not even GP exists -> return ideal so caller can inform the patient.
        return norm_ideal

    # Score every rule by how many of its keywords hit. The old logic returned
    # the FIRST rule with any match, which let a generic word shared with the
    # General Physician catch-all (e.g. "pain", "infection") steal a case that
    # actually had a stronger specialist signal. Scoring picks the specialty
    # with the most keyword hits; ties are broken by specialist priority so a
    # specific specialist always beats the GP catch-all on an equal score, and
    # higher-risk specialties are preferred when genuinely ambiguous.
    best_ideal, _score = _score_symptom_rules(_keyword_hits)

    if best_ideal is not None:
        return _resolve(best_ideal)

    # Unclassifiable symptom: send to General Physician if available, so the
    # booking is still validated against a real department instead of skipping
    # the check entirely.
    if available_departments:
        fb = (normalize_department(_FALLBACK_SPECIALTY) or _FALLBACK_SPECIALTY)
        if fb.lower() in available_departments:
            logger.info(
                "SYMPTOM ROUTING: '%s' unclassified; defaulting to '%s'.",
                reason_str, fb
            )
            return fb
    return None


# Priority for tie-breaking when multiple symptom rules match (lower = wins):
# high-risk specialists first, GP catch-all last.
_SYMPTOM_PRIORITY = {
    "Neurologist": 0,
    "Cardiology": 0,
    "Oncology": 0,
    "Surgeon": 1,
    "Orthopedics": 1,
    "Dermatology": 1, "Ophthalmology": 1, "ENT": 1, "Urology": 1,
    "Nephrology": 1, "Dentistry": 1, "Pulmonology": 1, "Gynecology": 1,
    "Pediatrics": 1, "Psychiatry": 1, "Gastroenterology": 1,
    "General Physician": 9,   # catch-all: only wins if nothing else scores
}


def _score_symptom_rules(keyword_hits_fn) -> tuple[Optional[str], int]:
    """Return (best_ideal_specialty, score) for the given per-keyword match fn.

    best_ideal is the specialty the symptom text points to BEFORE any
    availability/GP-fallback logic — i.e. the clinically-ideal department. score
    is how many keywords of the winning rule matched (0 = no symptom recognized
    at all, which signals a generic/placeholder reason like "X consultation")."""
    best_ideal = None
    best_score = 0
    best_priority = 99
    for keywords, ideal in _SYMPTOM_RULES:
        score = sum(1 for w in keywords if keyword_hits_fn(w))
        if score == 0:
            continue
        prio = _SYMPTOM_PRIORITY.get(ideal, 5)
        if score > best_score or (score == best_score and prio < best_priority):
            best_ideal = ideal
            best_score = score
            best_priority = prio
    return best_ideal, best_score


def _make_keyword_matcher(reason_str: str):
    """Build the token-aware keyword matcher for a reason string (same logic
    get_symptom_specialty uses), so other functions can classify a reason
    without duplicating the matching rules."""
    r = (reason_str or "").lower()
    _reason_tokens = [t for t in re.split(r"[^\w\u0900-\u097F]+", r) if t]
    _reason_token_set = set(_reason_tokens)

    def _keyword_hits(keyword: str) -> bool:
        kw = keyword.strip().lower()
        if not kw:
            return False
        if " " in kw:
            parts = [p for p in kw.split() if p]
            if all(p in _reason_token_set for p in parts):
                return True
            return kw in r
        # Compound symptom words (checked first, works for short keywords too):
        # keyword + a known symptom suffix as a single token — "ear"->"earache",
        # "tooth"->"toothache", "back"->"backache". Safe because the suffix set is
        # closed and symptom-specific.
        if len(kw) >= 3:
            for tok in _reason_token_set:
                if tok.startswith(kw) and tok[len(kw):] in ("ache", "aches", "pain", "pains"):
                    return True
        if len(kw) < 4:
            return kw in _reason_token_set
        for tok in _reason_token_set:
            if tok == kw:
                return True
            if len(kw) >= 3 and tok.startswith(kw):
                remainder = tok[len(kw):]
                if remainder in ("ache", "aches", "pain", "pains"):
                    return True
            if len(tok) >= 4:
                common = 0
                for a, b in zip(kw, tok):
                    if a != b:
                        break
                    common += 1
                if common >= 4 and common >= len(kw) - 2 and common >= len(tok) - 2:
                    # Guard: if the TOKEN is a strict prefix of the KEYWORD (the
                    # token is the shorter word, e.g. tok="hear" kw="heart"), only
                    # accept it when the token is long enough that a coincidental
                    # real word is unlikely. "hear"->"heart", "car"->"cardiac",
                    # "bone"->"boneless" must NOT match; "fractur"->"fracture" or
                    # "vomitin"->"vomiting" (token >=6) still do. This keeps
                    # inflection recall while dropping short-word false positives.
                    if len(kw) > len(tok) and tok == kw[:len(tok)]:
                        if len(tok) < 6:
                            continue
                    return True
        return False

    return _keyword_hits


def classify_reason(reason: str, data: Optional[dict] = None) -> dict:
    """Classify a free-text reason so callers can make correct routing/gating
    decisions. Distinguishes the three cases that all otherwise collapse to
    'General Physician' and caused wrong bookings:

    Returns a dict:
      {
        "category": one of
            "no_signal"        -> reason carries no symptom (generic placeholder
                                   like "cardiology consultation", or empty).
                                   Trust the verified doctor/department.
            "gp"               -> a real General-Physician symptom (fever, etc.).
            "specialist"       -> a real symptom pointing to a specific specialty.
        "ideal_specialty":  the clinically-ideal specialty (e.g. "Dentistry" for
                            toothache) BEFORE availability fallback, or None.
        "available_here":   True/False/None — whether ideal_specialty is a
                            department the hospital actually staffs (None if
                            unknown / not applicable).
      }
    """
    if not reason or not reason.strip():
        return {"category": "no_signal", "ideal_specialty": None, "available_here": None}

    avail = _available_department_set(data)
    matcher = _make_keyword_matcher(reason)
    ideal, score = _score_symptom_rules(matcher)

    # Also detect an explicit DEPARTMENT NAME in the reason itself. Reasons like
    # "Cardiology consultation" / "न्यूरोलॉजी consultation" carry no symptom
    # keyword, but they name a specialty directly — that IS a routing signal and
    # must not be treated as "no_signal" (which allowed "Cardiology consultation"
    # to be booked under Neurology). normalize_department maps the phrase to a
    # canonical specialty; we only trust it when it resolves to a real specialty
    # name rather than an incidental fuzzy hit.
    named_dept = None
    norm_named = normalize_department(reason)
    if norm_named:
        # Guard against normalize_department's fuzzy matching firing on unrelated
        # text: require the reason to actually contain a recognizable department
        # token, in either script.
        rl = reason.lower()
        dept_tokens = [
            "cardiolog", "cardiac", "कार्डियो", "हृदय",
            "neurolog", "neuro", "न्यूरो", "तंत्रिका",
            "orthop", "ortho", "ऑर्थो", "हड्डी रोग",
            "surg", "सर्जन", "शल्य",
            "physician", "general medicine", "सामान्य", "जनरल",
            "dermat", "skin specialist", "त्वचा रोग",
            "dent", "दंत", "ent", "ophthal", "eye specialist",
            "gynae", "gyneco", "pediatric", "paediatric", "बाल रोग",
            "oncolog", "कैंसर रोग", "urolog", "nephrolog", "psychiatr",
        ]
        if any(t in rl for t in dept_tokens):
            named_dept = norm_named

    # If BOTH a symptom and a department name are present, the symptom wins (it is
    # the more specific clinical signal). If only a department name is present,
    # use it.
    if (ideal is None or score == 0) and named_dept:
        norm_named_l = named_dept.lower()
        fb_l = (normalize_department(_FALLBACK_SPECIALTY) or _FALLBACK_SPECIALTY).lower()
        available_here = (norm_named_l in avail) if avail else None
        if norm_named_l == fb_l:
            return {"category": "gp", "ideal_specialty": named_dept, "available_here": available_here}
        return {"category": "specialist", "ideal_specialty": named_dept, "available_here": available_here}

    if ideal is None or score == 0:
        # No symptom keyword AND no department name -> generic/placeholder reason.
        return {"category": "no_signal", "ideal_specialty": None, "available_here": None}

    norm_ideal = (normalize_department(ideal) or ideal)
    fb = (normalize_department(_FALLBACK_SPECIALTY) or _FALLBACK_SPECIALTY)
    available_here = (norm_ideal.lower() in avail) if avail else None

    if norm_ideal.lower() == fb.lower():
        return {"category": "gp", "ideal_specialty": norm_ideal, "available_here": available_here}

    return {"category": "specialist", "ideal_specialty": norm_ideal, "available_here": available_here}


_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_DIR, "hospital_cache.json")


def load_cache_data() -> dict | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            payload = json.load(f)
        return payload.get("data")
    except Exception:
        return None




def is_qualification_consistent(qualification: str, department: str) -> bool:
    if not qualification or not department:
        return True
    q = qualification.lower()
    d = department.lower().strip()
    
    if "ortho" in d:
        if any(x in q for x in ["cardio", "neuro", "dvl", "derm"]):
            return False
    if "cardio" in d:
        if any(x in q for x in ["ortho", "neuro", "dvl", "derm"]):
            return False
    if "neuro" in d:
        if any(x in q for x in ["cardio", "ortho", "dvl", "derm"]):
            return False
    if "derm" in d:
        if any(x in q for x in ["cardio", "ortho", "neuro"]):
            return False
    return True


def normalize_department(dept_str: str) -> Optional[str]:
    if not dept_str:
        return None
    d = dept_str.lower().strip()
    
    mapping = {
        "neurology": "Neurologist",
        "neurologist": "Neurologist",
        "neuro": "Neurologist",
        "दिमाग": "Neurologist",
        "नसों": "Neurologist",
        
        "cardiology": "Cardiology",
        "cardiologist": "Cardiology",
        "cardio": "Cardiology",
        "दिल": "Cardiology",
        
        "orthopedics": "Orthopedics",
        "orthopedic": "Orthopedics",
        "orthopaedic": "Orthopedics",
        "ortho": "Orthopedics",
        "orthopedic doctor": "Orthopedics",
        "orthopaedic doctor": "Orthopedics",
        "haddi": "Orthopedics",
        "हड्डी": "Orthopedics",
        "जोड़ों": "Orthopedics",
        
        "surgeon": "Surgeon",
        "surgery": "Surgeon",
        "surg": "Surgeon",
        "ऑपरेशन": "Surgeon",
        "सर्जरी": "Surgeon",
        
        "dermatology": "Dermatology",
        "dermatologist": "Dermatology",
        "derm": "Dermatology",
        "त्वचा": "Dermatology",
        
        "gastroenterology": "Gastroenterology",
        "gastroenterologist": "Gastroenterology",
        "gastro": "Gastroenterology",
        
        "ent": "ENT",
        "ent specialist": "ENT",
        "नाक": "ENT",
        "कान": "ENT",
        "गला": "ENT",
        
        "dentistry": "Dentistry",
        "dentist": "Dentistry",
        "दांत": "Dentistry",
        "दंत": "Dentistry",
        "दांतों": "Dentistry",
        
        "general physician": "General Physician",
        "physician": "General Physician",
        "general medicine": "General Physician",
        
        "pulmonology": "Pulmonology",
        "pulmonologist": "Pulmonology",
        "lungs": "Pulmonology",
        "फेफड़े": "Pulmonology",
        
        "urology": "Urology",
        "urologist": "Urology",
        "urine": "Urology",
        "मूत्र": "Urology",
        
        "nephrology": "Nephrology",
        "nephrologist": "Nephrology",
        "kidney": "Nephrology",
        "किडनी": "Nephrology",

        # Specialties that appear in _SYMPTOM_RULES and therefore can be returned
        # by get_symptom_specialty. These MUST be normalizable, otherwise the
        # high-risk fallback guard (which compares normalize_department(ideal)
        # against _NO_GP_FALLBACK_SPECIALTIES) silently fails and the symptom
        # gets downgraded to GP. This was the "cancer -> General Physician" bug.
        "oncology": "Oncology",
        "oncologist": "Oncology",
        "cancer": "Oncology",
        "कैंसर": "Oncology",

        "ophthalmology": "Ophthalmology",
        "ophthalmologist": "Ophthalmology",
        "eye": "Ophthalmology",
        "आंख": "Ophthalmology",

        "gynecology": "Gynecology",
        "gynaecology": "Gynecology",
        "gynae": "Gynecology",
        "obstetrics": "Gynecology",

        "pediatrics": "Pediatrics",
        "paediatrics": "Pediatrics",
        "pediatric": "Pediatrics",

        "psychiatry": "Psychiatry",
        "psychiatrist": "Psychiatry",
        "mental health": "Psychiatry",

        # Devanagari department names — callers naming a department in Hindi
        # ("न्यूरोलॉजी", "कार्डियोलॉजी विभाग") must resolve just like the English
        # names. Department names are a small, closed set (unlike free-text
        # symptoms), so listing them here is safe and stable.
        "कार्डियोलॉजी": "Cardiology",
        "कार्डियोलाजी": "Cardiology",
        "हृदय रोग": "Cardiology",
        "हृदय": "Cardiology",
        "दिल रोग": "Cardiology",
        "न्यूरोलॉजी": "Neurologist",
        "न्यूरोलाजी": "Neurologist",
        "न्यूरोलॉजिस्ट": "Neurologist",
        "तंत्रिका रोग": "Neurologist",
        "ऑर्थोपेडिक्स": "Orthopedics",
        "ऑर्थोपेडिक": "Orthopedics",
        "आर्थोपेडिक्स": "Orthopedics",
        "हड्डी रोग": "Orthopedics",
        "अस्थि रोग": "Orthopedics",
        "सर्जन": "Surgeon",
        "सर्जरी": "Surgeon",
        "शल्य चिकित्सा": "Surgeon",
        "जनरल फिजिशियन": "General Physician",
        "सामान्य चिकित्सक": "General Physician",
        "सामान्य रोग": "General Physician",
        "फिजिशियन": "General Physician",
        "त्वचा रोग": "Dermatology",
        "दंत चिकित्सा": "Dentistry",
        "दंत रोग": "Dentistry",
        "बाल रोग": "Pediatrics",
        "कैंसर रोग": "Oncology",
        "मूत्र रोग": "Urology",
    }
    
    if d in mapping:
        val = mapping[d]
        logger.info(f"NORMALIZATION DECISION: '{dept_str}' mapped to '{val}' via direct synonym.")
        return val
        
    for key, val in mapping.items():
        if key in d or d in key:
            logger.info(f"NORMALIZATION DECISION: '{dept_str}' mapped to '{val}' via substring match with '{key}'.")
            return val
            
    import difflib
    matches = difflib.get_close_matches(d, list(mapping.keys()), n=1, cutoff=0.5)
    if matches:
        val = mapping[matches[0]]
        logger.info(f"NORMALIZATION DECISION: '{dept_str}' mapped to '{val}' via fuzzy match with '{matches[0]}'.")
        return val
        
    return None


def is_doctor_consistent_with_dept(doctor_obj: dict, department: str) -> bool:
    if not doctor_obj or not department:
        return True
    spec = doctor_obj.get("specialization") or doctor_obj.get("department")
    qual = doctor_obj.get("doctorQualification")
    
    if spec:
        norm_spec = normalize_department(spec)
        norm_dept = normalize_department(department)
        if norm_spec and norm_dept and norm_spec.lower() != norm_dept.lower():
            return False
            
    if qual:
        if not is_qualification_consistent(qual, department):
            return False
            
    return True


def validate_appointment(state: AppointmentState, data: Optional[dict] = None) -> tuple[bool, Optional[str]]:
    # Check required fields exist
    required_fields = [
        "patient_name", "patient_phone", "reason", "department", 
        "doctor_name", "appointment_date", "appointment_time"
    ]
    for field in required_fields:
        val = getattr(state, field, None)
        if not val:
            return False, f"Missing required field: {field}"

    if not state.patient_name_locked:
        return False, "Patient name is not confirmed and locked."

    # Validate phone number
    clean_phone = re.sub(r"[\s\-\(\)\+]", "", state.patient_phone)
    if len(clean_phone) == 12 and clean_phone.startswith("91"):
        clean_phone = clean_phone[2:]
    elif len(clean_phone) == 11 and clean_phone.startswith("0"):
        clean_phone = clean_phone[1:]
    
    if not (clean_phone.isdigit() and len(clean_phone) == 10 and clean_phone[0] in "6789"):
        return False, "Invalid phone number."

    # ------------------------------------------------------------------
    # Clinical safety backstop (the ONLY symptom-based gate).
    #
    # Routing is the model's job, validated against live data (department must
    # exist, doctor must belong to it — checked below). We do NOT re-derive a
    # department from the free-text reason and reject on any difference; that
    # blanket check collapsed correct specialist bookings whenever the reason was
    # a generic placeholder ("neurology consultation") that keyword-maps to GP.
    #
    # We enforce exactly one thing here: a HIGH-RISK symptom (cardiac / neuro /
    # oncology) must not be booked under a different department when the hospital
    # actually offers the right one. check_high_risk_downgrade encapsulates that.
    # ------------------------------------------------------------------
    high_risk_err = check_high_risk_downgrade(state.reason, state.department, data)
    if high_risk_err:
        return False, high_risk_err

    # Validate doctor matches department and qualification is consistent
    if data and "doctors" in data:
        # Find doctor's object
        doc_obj = None
        for d in data["doctors"]:
            d_name = d.get("name", "")
            if state.doctor_name.lower().strip() in d_name.lower().strip() or d_name.lower().strip() in state.doctor_name.lower().strip():
                doc_obj = d
                break
        if doc_obj:
            if not is_doctor_consistent_with_dept(doc_obj, state.department):
                return False, f"Doctor {state.doctor_name} is inconsistent with the {state.department} department (due to qualification or specialization mismatch)."
    
    # Check slot verification succeeded
    if not state.availability_verified:
        return False, "Slot availability not verified."

    return True, None


def normalize_hindi_phone_number(phone_str: str) -> str:
    if not phone_str:
        return ""

    # Convert any non-ASCII digit glyphs (Arabic-Indic, Devanagari, etc.) to
    # ASCII first, so numbers dictated in any script parse correctly.
    phone_str = normalize_unicode_digits(phone_str)

    WORD_VALUES = {
        "शून्य": 0, "सुन्ना": 0, "जीरो": 0, "जीरों": 0, "zero": 0,
        # Urdu-script number words (STT often returns these for Hindi/Urdu callers)
        "صفر": 0, "زیرو": 0, "ایک": 1, "دو": 2, "تین": 3, "چار": 4,
        "پانچ": 5, "پانج": 5, "چھ": 6, "چھے": 6, "سات": 7, "آٹھ": 8,
        "اٹھ": 8, "نو": 9, "نائن": 9,
        # Romanized/Hinglish Hindi number words — Hindi STT very often transliterates
        # spoken digits to Latin ("nau aath do") instead of Devanagari, which the
        # map previously missed entirely (parsed to an empty string).
        "shunya": 0, "sunna": 0, "sifar": 0,
        "ek": 1, "eak": 1,
        "do": 2, "doe": 2,
        "teen": 3, "tin": 3,
        "char": 4, "chaar": 4,
        "paanch": 5, "panch": 5, "paach": 5,
        "chhe": 6, "che": 6, "chha": 6, "chah": 6, "cheh": 6,
        "saat": 7, "sat": 7,
        "aath": 8, "ath": 8, "aat": 8,
        "nau": 9, "nao": 9,
        "एक": 1, "one": 1,
        "दो": 2, "two": 2,
        "तीन": 3, "three": 3,
        "चार": 4, "four": 4,
        "पांच": 5, "पाँच": 5, "five": 5,
        "छह": 6, "छः": 6, "छ": 6, "six": 6,
        "सात": 7, "seven": 7,
        "आठ": 8, "eight": 8,
        "नौ": 9, "नो": 9, "nine": 9,
        "दस": 10, "ten": 10,
        "ग्यारह": 11, "eleven": 11,
        "बारह": 12, "twelve": 12,
        "तेरह": 13, "thirteen": 13,
        "चौदह": 14, "fourteen": 14,
        "पंद्रह": 15, "fifteen": 15,
        "सोलह": 16, "sixteen": 16,
        "सत्रह": 17, "seventeen": 17,
        "अठारह": 18, "eighteen": 18,
        "उन्नीस": 19, "nineteen": 19,
        "बीस": 20, "twenty": 20,
        "इक्कीस": 21, "twentyone": 21, "twenty-one": 21,
        "बाईस": 22, "twentytwo": 22, "twenty-two": 22,
        "तेईस": 23, "तेइस": 23, "twentythree": 23, "twenty-three": 23,
        "चौबीस": 24, "twentyfour": 24, "twenty-four": 24,
        "पच्चीस": 25, "twentyfive": 25, "twenty-five": 25,
        "छब्बीस": 26, "twentysix": 26, "twenty-six": 26,
        "सत्ताईस": 27, "twentyseven": 27, "twenty-seven": 27,
        "अठ्ठाईस": 28, "twentyeight": 28, "twenty-eight": 28,
        "उनतीस": 29, "twentynine": 29, "twenty-nine": 29,
        "तीस": 30, "thirty": 30,
        "इकतीस": 31, "thirtyone": 31, "thirty-one": 31,
        "बत्तीस": 32, "thirtytwo": 32, "thirty-two": 32,
        "तेतीस": 33, "thirtythree": 33, "thirty-three": 33,
        "चौंतीस": 34, "thirtyfour": 34, "thirty-four": 34,
        "पैंतीस": 35, "पैतिस": 35, "thirtyfive": 35, "thirty-five": 35,
        "छत्तीस": 36, "thirtysix": 36, "thirty-six": 36,
        "सैंतीस": 37, "सेंतीस": 37, "thirtyseven": 37, "thirty-seven": 37,
        "अड़तीस": 38, "thirtyeight": 38, "thirty-eight": 38,
        "उनतालीस": 39, "thirtynine": 39, "thirty-nine": 39,
        "चालीस": 40, "forty": 40,
        "इकतालीस": 41, "fortyone": 41, "forty-one": 41,
        "बयालीस": 42, "fortytwo": 42, "forty-two": 42,
        "तैंतालीस": 43, "तैतिस": 43, "fortythree": 43, "forty-three": 43,
        "चवालीस": 44, "fortyfour": 44, "forty-four": 44,
        "पैंतालीस": 45, "पैतिस": 45, "fortyfive": 45, "forty-five": 45,
        "छियालीस": 46, "fortysix": 46, "forty-six": 46,
        "सैंतालीस": 47, "fortyseven": 47, "forty-seven": 47,
        "अड़तालीस": 48, "fortyeight": 48, "forty-eight": 48,
        "उनचास": 49, "fortynine": 49, "forty-nine": 49,
        "पचास": 50, "fifty": 50,
        "इक्यावन": 51, "fiftyone": 51, "fifty-one": 51,
        "बावन": 52, "fiftytwo": 52, "fifty-two": 52,
        "तिरेपन": 53, "fiftythree": 53, "fifty-three": 53,
        "चौवन": 54, "fiftyfour": 54, "fifty-four": 54,
        "पचपन": 55, "fiftyfive": 55, "fifty-five": 55,
        "छ्प्पन": 56, "छप्पन": 56, "fiftysix": 56, "fifty-six": 56,
        "सत्तावन": 57, "fiftyseven": 57, "fifty-seven": 57,
        "अठ्ठावन": 58, "fiftyeight": 58, "fifty-eight": 58,
        "उनसठ": 59, "fiftynine": 59, "fifty-nine": 59,
        "साठ": 60, "sixty": 60,
        "इकसठ": 61, "sixtyone": 61, "sixty-one": 61,
        "बासठ": 62, "sixtytwo": 62, "sixty-two": 62,
        "तिरेसठ": 63, "sixtythree": 63, "sixty-three": 63,
        "चौसठ": 64, "sixtyfour": 64, "sixty-four": 64,
        "पैंसठ": 65, "sixtyfive": 65, "sixty-five": 65,
        "छियासठ": 66, "sixtysix": 66, "sixty-six": 66,
        "सरसठ": 67, "sixtyseven": 67, "sixty-seven": 67,
        "अड़सठ": 68, "sixtyeight": 68, "sixty-eight": 68,
        "उनहत्तर": 69, "sixtynine": 69, "sixty-nine": 69,
        "सत्तर": 70, "seventy": 70,
        "इकहत्तर": 71, "seventyone": 71, "seventy-one": 71,
        "बहत्तर": 72, "seventytwo": 72, "seventy-two": 72,
        "तिहत्तर": 73, "seventythree": 73, "seventy-three": 73,
        "चौहत्तर": 74, "seventyfour": 74, "seventy-four": 74,
        "पचहत्तर": 75, "seventyfive": 75, "seventy-five": 75,
        "छिहत्तर": 76, "seventysix": 76, "seventy-six": 76,
        "सतहत्तर": 77, "seventyseven": 77, "seventy-seven": 77,
        "अठहत्तर": 78, "seventyeight": 78, "seventy-eight": 78,
        "उन्यासी": 79, "seventynine": 79, "seventy-nine": 79,
        "अस्सी": 80, "eighty": 80,
        "इक्यासी": 81, "eightyone": 81, "eighty-one": 81,
        "बयासी": 82, "eightytwo": 82, "eighty-two": 82,
        "तिरासी": 83, "eightythree": 83, "eighty-three": 83,
        "चौरासी": 84, "eightyfour": 84, "eighty-four": 84,
        "पचासी": 85, "eightyfive": 85, "eighty-five": 85,
        "छियासी": 86, "eightysix": 86, "eighty-six": 86,
        "सत्तासी": 87, "eightyseven": 87, "eighty-seven": 87,
        "अठासी": 88, "eightyeight": 88, "eighty-eight": 88,
        "नवासी": 89, "eightynine": 89, "eighty-nine": 89,
        "नब्बे": 90, "ninety": 90,
        "इक्यानवे": 91, "ninetyone": 91, "ninety-one": 91,
        "बानवे": 92, "बयानवे": 92, "बांवे": 92, "ninetytwo": 92, "ninety-two": 92,
        "तिरानवे": 93, "ninetythree": 93, "ninety-three": 93,
        "चौरानवे": 94, "ninetyfour": 94, "ninety-four": 94,
        "पचानवे": 95, "ninetyfive": 95, "ninety-five": 95,
        "छियानवे": 96, "ninetysix": 96, "ninety-six": 96,
        "सत्तानवे": 97, "ninetyseven": 97, "ninety-seven": 97,
        "अट्ठानवे": 98, "ninetyeight": 98, "ninety-eight": 98,
        "निन्यानवे": 99, "ninetynine": 99, "ninety-nine": 99
    }

    SCALE_VALUES = {
        "सौ": 100, "sau": 100, "hundred": 100, "सैकड़ा": 100,
        "हजार": 1000, "hazar": 1000, "thousand": 1000
    }

    tokens = re.split(r"[\s\-\(\)\+\,\.]+", phone_str.strip())
    result_digits = []
    multiplier = 1
    
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i].lower().strip()
        if not token:
            i += 1
            continue
            
        if token in ("डबल", "double", "ڈبل"):
            multiplier = 2
            i += 1
            continue
        if token in ("ट्रिपल", "triple", "ٹرپل"):
            multiplier = 3
            i += 1
            continue
            
        if token in SCALE_VALUES:
            scale = SCALE_VALUES[token]
            if result_digits:
                prev = result_digits[-1]
                if len(prev) == 1:
                    result_digits.pop()
                    scaled_val = str(int(prev) * scale)
                    result_digits.extend([scaled_val] * multiplier)
                else:
                    result_digits.extend([str(scale)] * multiplier)
            else:
                result_digits.extend([str(scale)] * multiplier)
            multiplier = 1
            i += 1
            continue

        if token in WORD_VALUES:
            val = WORD_VALUES[token]
            if i + 1 < n and tokens[i+1].lower().strip() in SCALE_VALUES:
                scale = SCALE_VALUES[tokens[i+1].lower().strip()]
                combined_val = val * scale
                i += 2
                if i < n and tokens[i].lower().strip() in WORD_VALUES:
                    next_val = WORD_VALUES[tokens[i].lower().strip()]
                    combined_val += next_val
                    i += 1
                result_digits.extend([str(combined_val)] * multiplier)
            else:
                result_digits.extend([str(val)] * multiplier)
                i += 1
            multiplier = 1
            continue
            
        cleaned = re.sub(r"\D", "", token)
        if cleaned:
            result_digits.extend([cleaned] * multiplier)
            multiplier = 1
        i += 1
        
    return "".join(result_digits)


def normalize_time_slot(time_str: str) -> str:
    if not time_str:
        return ""
    # Try parsing standard formats
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M %p"):
        try:
            dt = datetime.datetime.strptime(time_str.strip(), fmt)
            return dt.strftime("%I:%M %p").lstrip('0')
        except ValueError:
            continue
    # Regex fallback
    m = re.match(r"^0?(\d+):(\d+)\s*(AM|PM|am|pm)?", time_str.strip())
    if m:
        hr = int(m.group(1))
        min_str = m.group(2)
        am_pm = (m.group(3) or "AM").upper()
        return f"{hr}:{min_str} {am_pm}"
    return time_str.strip()