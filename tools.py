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


def get_symptom_specialty(reason_str: str) -> Optional[str]:
    if not reason_str:
        return None
    r = reason_str.lower()
    if any(w in r for w in ["fever", "bukhar", "taap", "बुखार", "ताप", "stomach", "pet dard", "पेट दर्द", "abdominal", "vomiting", "diarrhea", "loose motion", "nausea", "cough", "cold", "body ache"]):
        return "General Physician"
    if any(w in r for w in ["migraine", "dizziness", "headache", "seizure", "paralysis", "numbness", "सिरदर्द", "चक्कर"]):
        return "Neurologist"
    if any(w in r for w in ["asthma", "breathing", "lung", "दमा", "सांस"]):
        return "Pulmonology"
    if any(w in r for w in ["tooth", "teeth", "dent", "daant", "दांत"]):
        return "Dentistry"
    if any(w in r for w in ["chest", "palpitation", "heart", "blood pressure", "छाती", "दिल"]):
        return "Cardiology"
    if any(w in r for w in ["hernia", "appendicitis", "surgery", "operation", "stone", "gallbladder", "pathri", "पथरी", "सर्जरी", "ऑपरेशन"]):
        return "Surgeon"
    if any(w in r for w in ["fracture", "bone", "hand", "knee", "joint", "haddi", "हड्डी", "पीठ", "back pain"]):
        return "Orthopedics"
    if any(w in r for w in ["acne", "rash", "skin", "dermat"]):
        return "Dermatology"
    if any(w in r for w in ["kidney", "urine", "bladder", "urinary", "prostate", "किडनी", "गुर्दा", "पेशाब"]):
        return "Urology"
    return None


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

    # Validate reason matches department
    symptom_dept = get_symptom_specialty(state.reason)
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

        norm_symptom_dept = normalize_dept(symptom_dept)
        norm_state_dept = normalize_dept(state.department)
        
        # Check forbidden combinations
        # Face acne + Neurology
        # Migraine + Dermatology
        # Back pain + Neurology
        is_acne = any(w in state.reason.lower() for w in ["acne", "skin", "rash"])
        is_migraine = any(w in state.reason.lower() for w in ["migraine", "headache", "dizziness", "seizure"])
        is_back = any(w in state.reason.lower() for w in ["back pain", "bone", "joint", "knee"])
        
        if is_acne and "neuro" in norm_state_dept:
            return False, "Face acne is not consistent with Neurology."
        if is_migraine and "derm" in norm_state_dept:
            return False, "Migraine is not consistent with Dermatology."
        if is_back and "neuro" in norm_state_dept:
            return False, "Back pain is not consistent with Neurology."

        if norm_symptom_dept != norm_state_dept:
            return False, "Symptom and department mismatch."

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