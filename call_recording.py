"""
call_session_recorder.py
─────────────────────────────────────────────────────────────────────────────
LiveKit call recording and transcript management for the agent.

Self-contained — no dependency on the backend project (call_recording.py,
call_session_store.py, config.py).  All audio I/O and transcript logic is
implemented inline, using only the Python standard library plus pydub (optional,
for MP3 export — falls back to WAV if pydub / ffmpeg is unavailable).

Public surface (used by agent.py)
──────────────────────────────────
    call_id = derive_call_id(room, user_id)
    recorder = CallSessionRecorder(ctx, session, call_id, user_id, transcript)
    recorder.attach()   # call once, right after AgentSession is created
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import os
import re
import sys
import time
import wave
from array import array
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from livekit import agents, rtc

if TYPE_CHECKING:
    from livekit.agents import AgentSession, JobContext
    from tools import CallTranscript

logger = logging.getLogger(__name__)

# ── Audio constants ────────────────────────────────────────────────────────────
# LiveKit AudioStream defaults to 48 kHz.  We read the actual sample_rate from
# the first captured frame and use that for WAV headers and MP3 encoding.
# These are fallback values only — used if we never receive a frame.
_DEFAULT_SAMPLE_RATE = 48_000   # LiveKit AudioStream default
_SAMPLE_WIDTH        = 2        # 16-bit signed PCM
_CHANNELS            = 1        # mono

# ── Output directories ─────────────────────────────────────────────────────────
# Override via env vars to match whatever path the rest of your project uses.
_RECORDINGS_DIR  = os.environ.get("RECORDINGS_DIR",  "recordings")
_TRANSCRIPTS_DIR = os.environ.get("TRANSCRIPTS_DIR", "transcripts")

# ── Participant-identity tokens that identify the agent's own track ────────────
_AGENT_IDENTITY_TOKENS = ("agent", "nikita")


# ─────────────────────────────────────────────────────────────────────────────
# In-memory transcript store  (mirrors call_session_store behaviour)
# ─────────────────────────────────────────────────────────────────────────────

# keyed by call_id
_transcripts: Dict[str, List[Dict[str, Any]]] = {}
_call_start_ms: Dict[str, int] = {}


def _utc_ms() -> int:
    return int(time.time() * 1000)


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write_debug_log(msg: str) -> None:
    try:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_recording.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass


def _relative_time(epoch_ms: Optional[int], call_id: str) -> Optional[str]:
    if epoch_ms is None:
        return None
    start = _call_start_ms.get(call_id)
    if start is None:
        return None
    delta_us = max(0, int(epoch_ms) - start) * 1000
    h, r  = divmod(delta_us, 3_600_000_000)
    m, r  = divmod(r,        60_000_000)
    s, us = divmod(r,        1_000_000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int(us):06d}"


def _merge_text(prev: str, cur: str) -> str:
    """Merge streaming chunks: prefer the longer version that contains the shorter."""
    prev, cur = prev.strip(), cur.strip()
    if not prev:
        return cur
    if not cur:
        return prev
    if cur in prev:
        return prev
    if prev in cur:
        return cur
    return prev + " " + cur


def _store_register_start(call_id: str) -> None:
    _call_start_ms[call_id] = _utc_ms()


def _store_append(call_id: str, role: str, text: str) -> None:
    """Append a transcript entry. Each committed speech is a NEW entry (finished=True)."""
    if not text or not text.strip():
        return
    store = _transcripts.setdefault(call_id, [])
    now_iso, now_ms = _utc_iso(), _utc_ms()
    # Always create a new entry for committed speech
    store.append({
        "role":           role,
        "text":           text.strip(),
        "finished":       True,
        "timestamp":      now_iso,
        "tsIso":          now_iso,
        "tsEpochMs":      now_ms,
        "startTsIso":     now_iso,
        "startTsEpochMs": now_ms,
        "endTsIso":       now_iso,
        "endTsEpochMs":   now_ms,
    })
    
    # Progressively flush transcript to disk
    try:
        transcripts_dir = os.path.abspath(_TRANSCRIPTS_DIR)
        _store_flush_transcript(call_id, transcripts_dir)
    except Exception as e:
        logger.debug("Progressive transcript flush failed: %s", e)


def _store_flush_transcript(call_id: str, transcripts_dir: str) -> Optional[str]:
    """Coalesce same-role consecutive entries and write JSON. Returns path or None."""
    entries = _transcripts.get(call_id, [])
    if not entries:
        return None

    # Coalesce consecutive same-role entries
    out: List[Dict[str, Any]] = []
    for e in entries:
        if out and out[-1]["role"] == e["role"]:
            out[-1]["text"] = _merge_text(out[-1]["text"], e["text"])
            out[-1]["endTsEpochMs"] = e.get("endTsEpochMs")
            out[-1]["endTsIso"]     = e.get("endTsIso")
        else:
            out.append(dict(e))

    # Inject relative timestamps (startTime / endTime)
    for row in out:
        st = _relative_time(row.get("startTsEpochMs"), call_id)
        et = _relative_time(row.get("endTsEpochMs"),   call_id)
        if st:
            row["startTime"] = st
        if et:
            row["endTime"] = et

    txt_path = os.path.join(transcripts_dir, f"{call_id}.txt")
    json_path = os.path.join(transcripts_dir, f"{call_id}.json")
    try:
        os.makedirs(transcripts_dir, exist_ok=True)
        
        # Write human-readable text transcript
        with open(txt_path, "w", encoding="utf-8") as fh:
            for row in out:
                role = str(row.get("role", "unknown")).upper()
                text = str(row.get("text", "")).strip()
                st = str(row.get("startTime", ""))
                fh.write(f"[{st}] {role}:\n{text}\n\n")
                
        # Write structured JSON transcript
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
            
        return txt_path
    except Exception as exc:
        logger.error("Transcript write failed %s: %s", txt_path, exc)
        return None


def _store_clear(call_id: str) -> None:
    _transcripts.pop(call_id, None)
    _call_start_ms.pop(call_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_wav(pcm_chunks: List[bytes], path: str,
               rate: int, width: int, channels: int) -> bool:
    if not pcm_chunks:
        return False
    try:
        pcm = b"".join(pcm_chunks)
        if len(pcm) == 0:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        duration_s = len(pcm) / (rate * width * channels)
        logger.info("WAV written: %s (%.1fs, %dHz, %dch)", path, duration_s, rate, channels)
        return True
    except Exception as exc:
        logger.error("WAV write failed %s: %s", path, exc)
        return False


def _resample_pcm(samples: array, from_rate: int, to_rate: int) -> array:
    if from_rate == to_rate:
        return samples
    
    input_len = len(samples)
    if input_len == 0:
        return array("h")
        
    duration = input_len / from_rate
    output_len = int(duration * to_rate)
    
    output_samples = array("h", [0] * output_len)
    ratio = from_rate / to_rate
    
    for i in range(output_len):
        input_idx = i * ratio
        idx_low = int(math.floor(input_idx))
        idx_high = min(idx_low + 1, input_len - 1)
        weight = input_idx - idx_low
        
        if idx_low < input_len:
            s_low = samples[idx_low]
            s_high = samples[idx_high]
            interpolated = s_low + weight * (s_high - s_low)
            output_samples[i] = int(round(interpolated))
            
    return output_samples


def _mix_wavs_python(user_wav: str, agent_wav: str, out_wav: str) -> bool:
    """
    Mix two mono WAV files into a single WAV file.
    Does not depend on any external package (uses only standard library 'wave' and 'array').
    Resamples tracks dynamically if their sample rates mismatch.
    """
    try:
        user_exists = os.path.isfile(user_wav)
        agent_exists = os.path.isfile(agent_wav)
        if not user_exists and not agent_exists:
            return False
        
        # If only one track exists, just copy it to the output
        import shutil
        if not user_exists:
            shutil.copy2(agent_wav, out_wav)
            return True
        if not agent_exists:
            shutil.copy2(user_wav, out_wav)
            return True

        with wave.open(user_wav, "rb") as w_user, wave.open(agent_wav, "rb") as w_agent:
            u_params = w_user.getparams()
            a_params = w_agent.getparams()
            
            if u_params.sampwidth != a_params.sampwidth or u_params.nchannels != 1 or a_params.nchannels != 1:
                logger.warning("WAV parameters mismatch or not mono, skipping python mixing.")
                return False
            
            u_frames = w_user.readframes(w_user.getnframes())
            a_frames = w_agent.readframes(w_agent.getnframes())
            
            u_samples = array("h", u_frames)
            a_samples = array("h", a_frames)
            
            # Determine target rate (maximum of the two)
            target_rate = max(u_params.framerate, a_params.framerate)
            
            # Resample user audio if necessary
            if u_params.framerate != target_rate:
                logger.info("Resampling user audio from %dHz to %dHz", u_params.framerate, target_rate)
                u_samples = _resample_pcm(u_samples, u_params.framerate, target_rate)
                
            # Resample agent audio if necessary
            if a_params.framerate != target_rate:
                logger.info("Resampling agent audio from %dHz to %dHz", a_params.framerate, target_rate)
                a_samples = _resample_pcm(a_samples, a_params.framerate, target_rate)
            
            max_len = max(len(u_samples), len(a_samples))
            mixed_samples = array("h", [0] * max_len)
            
            for i in range(max_len):
                s1 = u_samples[i] if i < len(u_samples) else 0
                s2 = a_samples[i] if i < len(a_samples) else 0
                mixed_val = s1 + s2
                if mixed_val > 32767:
                    mixed_val = 32767
                elif mixed_val < -32768:
                    mixed_val = -32768
                mixed_samples[i] = mixed_val
            
            os.makedirs(os.path.dirname(os.path.abspath(out_wav)) or ".", exist_ok=True)
            with wave.open(out_wav, "wb") as w_out:
                w_out.setnchannels(1)
                w_out.setsampwidth(u_params.sampwidth)
                w_out.setframerate(target_rate)
                w_out.writeframes(mixed_samples.tobytes())
            
            duration_s = max_len / target_rate
            logger.info("Pure-Python mixed WAV written: %s (%.1fs, %dHz, 1ch)", out_wav, duration_s, target_rate)
            return True
    except Exception as exc:
        logger.error("Pure-Python WAV mixing failed: %s", exc)
        return False


def _mix_to_mp3(user_wav: str, agent_wav: str, out_path: str,
                rate: int = _DEFAULT_SAMPLE_RATE) -> bool:
    """
    Mix user + agent WAV tracks into a single MP3 using pydub.
    Falls back to preserving unmixed WAV files if pydub is absent but both exist,
    or copying a single track to WAV if the other is missing.
    Returns True on success (where original files can be safely deleted).
    """
    try:
        from pydub import AudioSegment  # optional dependency
    except ImportError:
        logger.warning("pydub not installed — evaluating WAV fallback")
        wav_out = out_path.rsplit(".", 1)[0] + ".wav"
        mixed_ok = _mix_wavs_python(user_wav, agent_wav, wav_out)
        return mixed_ok

    try:
        user_seg  = AudioSegment.from_file(user_wav)  if os.path.isfile(user_wav)  else None
        agent_seg = AudioSegment.from_file(agent_wav) if os.path.isfile(agent_wav) else None

        if not user_seg and not agent_seg:
            return False

        fmt = "mp3" if out_path.lower().endswith(".mp3") else "wav"

        # Export parameters for higher quality MP3
        export_params = {}
        if fmt == "mp3":
            export_params = {"bitrate": "192k"}

        if not user_seg:
            agent_seg.set_frame_rate(rate).set_channels(1).export(
                out_path, format=fmt, **export_params
            )
            return True
        if not agent_seg:
            user_seg.set_frame_rate(rate).set_channels(1).export(
                out_path, format=fmt, **export_params
            )
            return True

        user_seg  = user_seg.set_frame_rate(rate).set_channels(1)
        agent_seg = agent_seg.set_frame_rate(rate).set_channels(1)

        # Pad shorter track with silence
        diff = len(user_seg) - len(agent_seg)
        if diff > 0:
            agent_seg += AudioSegment.silent(duration=diff)
        elif diff < 0:
            user_seg  += AudioSegment.silent(duration=-diff)

        mixed = user_seg.overlay(agent_seg)
        try:
            mixed.export(out_path, format=fmt, **export_params)
            return True
        except Exception as exc:
            logger.error("MP3 mix failed, falling back to WAV: %s", exc)
            wav_out = out_path.rsplit(".", 1)[0] + ".wav"
            try:
                mixed.export(wav_out, format="wav")
                return True
            except Exception as e2:
                logger.error("WAV fallback also failed: %s", e2)
                return False
    except Exception as exc:
        logger.error("Audio mixing with pydub failed: %s. Falling back to pure Python WAV mixing.", exc)
        wav_out = out_path.rsplit(".", 1)[0] + ".wav"
        return _mix_wavs_python(user_wav, agent_wav, wav_out)


def _log_audio_quality(path: str, call_id: str) -> None:
    """Log basic quality metrics for the mixed recording."""
    if not os.path.isfile(path):
        # Check if fallback .wav exists
        wav_path = path.rsplit(".", 1)[0] + ".wav"
        if os.path.isfile(wav_path):
            path = wav_path
        else:
            return

    try:
        if path.lower().endswith(".mp3"):
            from pydub import AudioSegment
            seg = AudioSegment.from_file(path)
            frames = seg.raw_data
            rate   = seg.frame_rate
        else:
            with wave.open(path, "rb") as wf:
                rate   = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

        samples = array("h")
        samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
        if not samples:
            return
        abs_s    = [abs(s) for s in samples]
        rms      = math.sqrt(sum(float(s) * float(s) for s in samples) / len(samples))
        silence  = sum(1 for s in abs_s if s <= 500) / len(samples)
        duration = len(samples) / float(rate)
        logger.info(
            "Recording quality [%s]: duration=%.1fs rms=%.1f silence=%.0f%%",
            call_id, duration, rms, silence * 100,
        )
    except Exception as exc:
        logger.debug("Quality metrics error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Public helper
# ─────────────────────────────────────────────────────────────────────────────

def derive_call_id(room: rtc.Room, user_id: str) -> str:
    """
    Return a stable, filesystem-safe call_id for this session.

    Uses the LiveKit room name when available; falls back to
    ``lk_<user_id>_<epoch>`` so filenames are always unique.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    room_name = getattr(room, "name", "") or ""
    if room_name:
        clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", room_name)[:60]
        return f"{clean_name}-{timestamp}"
    return f"lk-{user_id}-{timestamp}"


# ─────────────────────────────────────────────────────────────────────────────
# Agent Audio Interception
# ─────────────────────────────────────────────────────────────────────────────

_active_recorders = set()
_shutdown_in_progress = False  # Global flag: once True, no frames reach the RTC layer

class _FrameEvent:
    def __init__(self, frame):
        self.frame = frame

class InterceptedAudioStream:
    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return _FrameEvent(frame)

if hasattr(rtc, "AudioSource"):
    _original_capture_frame = rtc.AudioSource.capture_frame

    async def _patched_capture_frame(self, frame, *args, **kwargs):
        # Route to recorders that are still live (per-recorder `_shutdown` flag).
        # Previously a single GLOBAL `_shutdown_in_progress` flag was used: when
        # ANY call disconnected it was flipped True, which dropped audio frames for
        # every OTHER concurrent call in the same process. Using a per-recorder
        # flag means one call ending no longer corrupts another call's recording.
        #
        # When NO recorder is live, we drop the frame instead of forwarding it to
        # the original C++ capture_frame — pushing audio into a peer connection
        # that is being torn down causes a `webrtc-sys` RtcError panic (esp. on
        # Windows). This is the panic guard the global flag used to provide.
        active = [r for r in list(_active_recorders) if not getattr(r, "_shutdown", False)]
        if not active:
            return

        # Known limitation: with >1 live call in a single worker process we cannot
        # map this AudioSource back to its owning call, so agent audio may mix
        # across recordings. LiveKit dispatches one job per process by default, so
        # this normally never triggers — but warn (throttled) if it ever does.
        if len(active) > 1:
            if not hasattr(_patched_capture_frame, "_last_multi_warn"):
                _patched_capture_frame._last_multi_warn = 0.0
            if time.time() - _patched_capture_frame._last_multi_warn > 30.0:
                logger.warning(
                    "%d concurrent recorders in one process — agent audio may mix "
                    "across calls. Run LiveKit with one job per process.", len(active)
                )
                _patched_capture_frame._last_multi_warn = time.time()

        # Log periodically to avoid flooding the console
        if not hasattr(_patched_capture_frame, "_last_log_time"):
            _patched_capture_frame._last_log_time = 0.0
        now = time.time()
        if now - _patched_capture_frame._last_log_time > 2.0:
            logger.info("--- _patched_capture_frame called (intercepting agent outgoing audio) ---")
            _patched_capture_frame._last_log_time = now

        # We copy the frame bytes so LiveKit can safely free/reuse the original C++ memory
        frame_data_copy = bytes(frame.data)
        
        class MockFrame:
            def __init__(self, data, sample_rate, num_channels):
                self.data = data
                self.sample_rate = sample_rate
                self.num_channels = num_channels
        
        mock_frame = MockFrame(
            frame_data_copy,
            getattr(frame, "sample_rate", _DEFAULT_SAMPLE_RATE),
            getattr(frame, "num_channels", _CHANNELS),
        )
        
        for recorder in active:
            try:
                recorder._agent_stream_queue.put_nowait(mock_frame)
            except Exception as e:
                logger.warning(f"Intercept error: {e}")

        # Final guard: re-check that at least one recorder is still live in case a
        # shutdown started between the top check and here (the TTS pipeline is
        # async and this window matters). If everything just shut down, drop the
        # frame rather than forwarding into a closing peer connection.
        if not any(not getattr(r, "_shutdown", False) for r in list(_active_recorders)):
            return
        return await _original_capture_frame(self, frame, *args, **kwargs)

    rtc.AudioSource.capture_frame = _patched_capture_frame


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class CallSessionRecorder:
    """
    Manages audio capture, transcript mirroring, and artefact persistence for
    a single LiveKit agent call.

    Usage::

        recorder = CallSessionRecorder(ctx, session, call_id, user_id, transcript)
        recorder.attach()   # wire all hooks; call once after session is created

    Drop this file alongside agent.py — no other project files required.
    """

    def __init__(
        self,
        ctx: "JobContext",
        session: "AgentSession",
        call_id: str,
        user_id: str,
        lk_transcript: "CallTranscript",
    ) -> None:
        self._ctx           = ctx
        self._session       = session
        self._call_id       = call_id
        self._user_id       = user_id
        self._lk_transcript = lk_transcript

        self._user_pcm:  List[bytes] = []
        self._agent_pcm: List[bytes] = []
        self._stop_event             = asyncio.Event()
        # Per-recorder shutdown flag — checked by the global capture-frame patch so
        # that THIS call ending never drops audio for other concurrent calls.
        self._shutdown               = False
        self._capture_tasks: List[asyncio.Task] = []
        
        self._agent_stream_queue = asyncio.Queue()
        self._agent_stream = InterceptedAudioStream(self._agent_stream_queue)

        # Detected sample rates from actual audio frames (set on first frame)
        self._user_sample_rate:  int = _DEFAULT_SAMPLE_RATE
        self._agent_sample_rate: int = _DEFAULT_SAMPLE_RATE
        self._user_num_channels:  int = _CHANNELS
        self._agent_num_channels: int = _CHANNELS
        self._user_rate_detected:  bool = False
        self._agent_rate_detected: bool = False
        self._start_time: float = time.time()

    # ── Public ────────────────────────────────────────────────────────────────

    def attach(self) -> None:
        """
        Wire all event listeners and the shutdown callback.
        Call exactly once, immediately after AgentSession is created
        and before session.start().
        """
        _store_register_start(self._call_id)
        logger.info("CallSessionRecorder attached: call_id=%s user_id=%s",
                    self._call_id, self._user_id)
        _write_debug_log(f"attach() called: call_id={self._call_id} user_id={self._user_id}")
        
        global _shutdown_in_progress
        _shutdown_in_progress = False  # Reset for each new call in the same process

        _active_recorders.add(self)

        # Start agent audio capture immediately using the intercepted stream
        task = asyncio.ensure_future(
            self._capture_audio_track(self._agent_stream, self._agent_pcm, is_agent=True)
        )
        self._capture_tasks.append(task)

        self._ctx.room.on("track_subscribed", self._on_track_subscribed)
        self._ctx.room.on("track_subscription_failed", self._on_track_subscription_failed)
        self._ctx.room.on("local_track_published", self._on_local_track_published)
        self._ctx.room.on("participant_disconnected", self._on_participant_disconnected)
        self._subscribe_existing_tracks()

        # Real-time transcript mirroring
        self._session.on("user_input_transcribed",  self._on_user_input_transcribed)
        self._session.on("conversation_item_added", self._on_conversation_item_added)

        # Shutdown → persist artefacts
        self._ctx.add_shutdown_callback(self._on_shutdown)

    # ── Audio capture ─────────────────────────────────────────────────────────

    def _subscribe_existing_tracks(self) -> None:
        """Pick up tracks that were published before attach() was called."""
        _write_debug_log("--- _subscribe_existing_tracks started ---")
        for participant in self._ctx.room.remote_participants.values():
            _write_debug_log(f"Participant: {participant.identity} (remote)")
            for pub in participant.track_publications.values():
                _write_debug_log(f"  Track publication: sid={pub.sid} kind={pub.kind} subscribed={pub.subscribed} track={pub.track}")
                if pub.kind == rtc.TrackKind.KIND_AUDIO:
                    if not pub.subscribed:
                        _write_debug_log(f"  Forcing subscription on existing track {pub.sid} of participant {participant.identity}")
                        logger.info("Forcing subscription for existing track %s of participant %s", pub.sid, participant.identity)
                        pub.set_subscribed(True)
                    if pub.track:
                        self._on_track_subscribed(pub.track, pub, participant)
                    else:
                        logger.info("Track %s is subscribed/subscribing but pub.track is None yet.", pub.sid)

        local_participant = self._ctx.room.local_participant
        if local_participant:
            _write_debug_log(f"Participant: {local_participant.identity} (local)")
            for pub in local_participant.track_publications.values():
                _write_debug_log(f"  Local track publication: sid={pub.sid} kind={pub.kind} track={pub.track}")
                if pub.track and isinstance(pub.track, rtc.LocalAudioTrack):
                    self._on_local_track_published(pub, pub.track)
        _write_debug_log("--- _subscribe_existing_tracks completed ---")

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        """Capture a remote participant's audio track (usually the user/caller)."""
        identity   = (getattr(participant, "identity", "") or "").lower()
        track_sid  = getattr(track, "sid", "?")
        part_identity = getattr(participant, "identity", "?")
        _write_debug_log(f"_on_track_subscribed called for track={track_sid} participant={part_identity} kind={track.kind}")
        logger.info("--- _on_track_subscribed called for track=%s participant=%s (identity=%s) ---",
                    track_sid, part_identity, identity)
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        buf        = self._user_pcm
        role_label = "user"
        logger.info("Capturing %s audio from participant=%s",
                    role_label, part_identity)
        task = asyncio.ensure_future(
            self._capture_audio_track(track, buf, is_agent=False)
        )
        self._capture_tasks.append(task)

    def _on_track_subscription_failed(
        self,
        participant: rtc.RemoteParticipant,
        track_sid: str,
        error: str
    ) -> None:
        part_identity = getattr(participant, "identity", "?")
        _write_debug_log(f"_on_track_subscription_failed: participant={part_identity} track_sid={track_sid} error={error}")
        logger.warning("Track subscription failed for participant %s track %s: %s",
                       part_identity, track_sid, error)
    def _on_local_track_published(self, pub, track) -> None:
        pass

    def _on_participant_disconnected(self, participant) -> None:
        global _shutdown_in_progress
        identity = getattr(participant, "identity", "unknown")
        logger.info("Participant disconnected in call_recording (identity=%s), setting stop event.", identity)
        _write_debug_log(f"Participant disconnected: {identity}")

        # Mark THIS recorder as shutting down first so the capture-frame patch
        # stops routing frames for this call (per-recorder, not global).
        self._shutdown = True
        # Keep the legacy global flag too (harmless; used by teardown timing).
        _shutdown_in_progress = True

        if self._stop_event.is_set():
            # Already processed a disconnect — avoid duplicate work
            logger.debug("Stop event already set, skipping duplicate disconnect handler.")
            return

        self._stop_event.set()
        # Push a None to wake up the agent stream queue
        try:
            self._agent_stream_queue.put_nowait(None)
        except Exception:
            pass
    async def _capture_audio_track(
        self,
        track: rtc.Track,
        buf: List[bytes],
        is_agent: bool = False,
    ) -> None:
        """
        Read PCM frames from an AudioStream and append to buf.
        Detects the actual sample_rate from the first frame received.
        """
        track_sid = getattr(track, "sid", "agent_stream")
        _write_debug_log(f"--- _capture_audio_track starting: track={track_sid} is_agent={is_agent} ---")
        logger.info("--- Starting audio capture track %s (is_agent=%s) ---", track_sid, is_agent)
        
        # Create AudioStream with the LiveKit default sample rate (48kHz)
        # so we get the best quality from the track.
        if is_agent:
            audio_stream = track
        else:
            try:
                audio_stream = rtc.AudioStream.from_track(
                    track=track,
                    sample_rate=_DEFAULT_SAMPLE_RATE,
                    num_channels=_CHANNELS,
                )
            except TypeError:
                try:
                    audio_stream = rtc.AudioStream(
                        track,
                        sample_rate=_DEFAULT_SAMPLE_RATE,
                        num_channels=_CHANNELS,
                    )
                except TypeError:
                    # Older SDK versions may not accept sample_rate/num_channels
                    audio_stream = rtc.AudioStream(track)

        import time
        current_bytes = sum(len(c) for c in buf)
        has_logged_first_frame = False

        try:
            async for frame_event in audio_stream:
                if not has_logged_first_frame:
                    _write_debug_log(f"Received first audio frame for track={track_sid} (is_agent={is_agent})")
                    logger.info("--- Received first audio frame for track=%s (is_agent=%s) ---", track_sid, is_agent)
                    has_logged_first_frame = True
                if self._stop_event.is_set():
                    _write_debug_log(f"Capture loop stopping: stop event set (track={track_sid}, is_agent={is_agent})")
                    break
                frame = frame_event.frame

                # Detect actual sample rate from the first frame
                frame_rate = getattr(frame, "sample_rate", None)
                frame_channels = getattr(frame, "num_channels", None)
                if frame_rate and not (self._agent_rate_detected if is_agent else self._user_rate_detected):
                    if is_agent:
                        self._agent_sample_rate = frame_rate
                        self._agent_num_channels = frame_channels or _CHANNELS
                        self._agent_rate_detected = True
                        logger.info("Agent audio: detected %dHz, %dch",
                                    self._agent_sample_rate, self._agent_num_channels)
                    else:
                        self._user_sample_rate = frame_rate
                        self._user_num_channels = frame_channels or _CHANNELS
                        self._user_rate_detected = True
                        logger.info("User audio: detected %dHz, %dch",
                                    self._user_sample_rate, self._user_num_channels)

                active_rate = self._agent_sample_rate if is_agent else self._user_sample_rate
                active_channels = self._agent_num_channels if is_agent else self._user_num_channels
                
                # Sync padding for tracks that don't send continuous silence (like agent TTS)
                elapsed_s = time.time() - self._start_time
                expected_bytes = int(elapsed_s * active_rate * active_channels * _SAMPLE_WIDTH)
                alignment = active_channels * _SAMPLE_WIDTH
                expected_bytes = (expected_bytes // alignment) * alignment
                
                missing_bytes = expected_bytes - current_bytes
                # If we are missing more than 100ms of audio, insert silence to resync
                if missing_bytes > int(0.1 * active_rate) * alignment:
                    buf.append(b'\x00' * missing_bytes)
                    current_bytes += missing_bytes

                frame_bytes = bytes(frame.data)
                buf.append(frame_bytes)
                current_bytes += len(frame_bytes)

        except Exception as exc:
            _write_debug_log(f"Audio capture error for track {track_sid}: {exc}")
            logger.error("Audio capture failed for track %s: %s",
                         track_sid, exc, exc_info=True)
        finally:
            _write_debug_log(f"--- _capture_audio_track finished: track={track_sid} is_agent={is_agent} total_chunks={len(buf)} ---")
            # On Windows, awaiting audio_stream.aclose() during an active peer-
            # connection teardown is known to trigger `webrtc-sys` panics, so we
            # skip it there. On Linux/macOS (production) NOT closing the real
            # rtc.AudioStream leaks its underlying task/socket for the life of the
            # worker, so we close it. The agent stream is our own in-memory
            # InterceptedAudioStream (no aclose) — only close real remote streams.
            if not is_agent and sys.platform != "win32":
                closer = getattr(audio_stream, "aclose", None)
                if closer is not None:
                    try:
                        await closer()
                    except Exception as close_exc:
                        logger.debug("audio_stream.aclose() failed for %s: %s", track_sid, close_exc)

    # ── Transcript mirroring ──────────────────────────────────────────────────

    def _on_user_input_transcribed(self, ev) -> None:
        """Called when user speech is committed (finalized by STT/Realtime)."""
        # ev is UserInputTranscribedEvent
        if not getattr(ev, "is_final", False):
            return
        text = getattr(ev, "transcript", "")
        if text:
            _store_append(self._call_id, "user", str(text))
            logger.debug("Transcript user: %s", text[:80])

    def _on_conversation_item_added(self, ev) -> None:
        """Called when agent or user message is added."""
        # ev is ConversationItemAddedEvent
        item = getattr(ev, "item", None)
        if not item:
            return
        role = getattr(item, "role", "")
        # Only record assistant speech here (user is handled by transcribed event)
        if role == "assistant":
            content = getattr(item, "content", "")
            if isinstance(content, list):
                # Handle multimodal content or list of strings
                text = " ".join(str(c) for c in content if isinstance(c, str))
                if not text:
                    text = " ".join(getattr(c, "text", "") for c in content if hasattr(c, "text"))
            else:
                text = str(content)
            
            if text:
                _store_append(self._call_id, "assistant", text)
                logger.debug("Transcript assistant: %s", text[:80])
    def _flush_lk_transcript_events(self) -> None:
        """Sweep any CallTranscript.events not already captured via session events."""
        for ev in getattr(self._lk_transcript, "events", None) or []:
            role_raw = (getattr(ev, "role", None)
                        or (ev.get("role") if isinstance(ev, dict) else None))
            text_raw = (getattr(ev, "text", None)
                        or (ev.get("text") if isinstance(ev, dict) else None)
                        or (getattr(ev, "details", None))
                        or (ev.get("details") if isinstance(ev, dict) else None))
            if role_raw and text_raw:
                _store_append(self._call_id, str(role_raw), str(text_raw))

    def _rebuild_transcript_from_history(self) -> None:
        """Rebuild transcript using the session's ChatContext history to avoid truncation."""
        try:
            if not hasattr(self, "_session") or not self._session:
                logger.warning("No session available to rebuild transcript from history.")
                return
            
            history = getattr(self._session, "history", None)
            if not history:
                logger.warning("No history found in session.")
                return
                
            messages_attr = getattr(history, "messages", [])
            messages = messages_attr() if callable(messages_attr) else messages_attr
            if not messages:
                logger.warning("Session history messages list is empty.")
                return
                
            logger.info("Rebuilding transcript from session history: %d messages total", len(messages))
            
            new_entries = []
            for msg in messages:
                role = getattr(msg, "role", "")
                if role not in ("user", "assistant"):
                    continue
                
                # Check for text content
                text = getattr(msg, "text_content", "").strip()
                if not text:
                    # Fallback to parsing content list if text_content is not populated directly
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):
                        text = " ".join(getattr(c, "text", "") for c in content if hasattr(c, "text")).strip()
                        if not text:
                            text = " ".join(str(c) for c in content if isinstance(c, str)).strip()
                    else:
                        text = str(content).strip()
                
                if not text:
                    continue
                
                # Timestamp parsing
                created_at_s = getattr(msg, "created_at", None)
                if created_at_s is None or created_at_s <= 0:
                    created_at_s = time.time()
                
                msg_ms = int(created_at_s * 1000)
                try:
                    dt = datetime.datetime.fromtimestamp(created_at_s, datetime.timezone.utc)
                    now_iso = dt.isoformat()
                except Exception:
                    now_iso = _utc_iso()
                    msg_ms = _utc_ms()
                
                new_entries.append({
                    "role":           role,
                    "text":           text,
                    "finished":       True,
                    "timestamp":      now_iso,
                    "tsIso":          now_iso,
                    "tsEpochMs":      msg_ms,
                    "startTsIso":     now_iso,
                    "startTsEpochMs": msg_ms,
                    "endTsIso":       now_iso,
                    "endTsEpochMs":   msg_ms,
                })
            
            if new_entries:
                _transcripts[self._call_id] = new_entries
                logger.info("Successfully rebuilt transcript with %d entries from history.", len(new_entries))
            else:
                logger.warning("No user/assistant messages found in history, transcript not rebuilt.")
        except Exception as e:
            logger.error("Failed to rebuild transcript from history: %s", e, exc_info=True)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def _on_shutdown(self, *args, **kwargs) -> None:
        """
        Called by LiveKit when the session ends.
        Accepts *args because add_shutdown_callback may pass a reason string.
        """
        global _shutdown_in_progress
        logger.info("--- _on_shutdown started for call_id=%s ---", self._call_id)
        _write_debug_log(f"--- _on_shutdown started: call_id={self._call_id} ---")
        _write_debug_log(f"Active capture tasks to stop: {len(self._capture_tasks)}")

        # 1. Mark THIS recorder shut down first so the capture-frame patch stops
        #    routing frames for this call (per-recorder). Also set the legacy
        #    global flag for teardown-timing compatibility.
        self._shutdown = True
        _shutdown_in_progress = True

        self._stop_event.set()
        try:
            self._agent_stream_queue.put_nowait(None)
        except Exception:
            pass

        if self in _active_recorders:
            _active_recorders.remove(self)
        
        # Give in-flight capture_frame calls a moment to observe the shutdown flag
        await asyncio.sleep(0.1)

        if self._capture_tasks:
            await asyncio.gather(*self._capture_tasks, return_exceptions=True)
            _write_debug_log("Capture tasks stopped.")

        # 2. Flush remaining transcript events from the CallTranscript object
        self._flush_lk_transcript_events()

        # Set call end time and log final call ended event if not already present
        self._lk_transcript.call_end_time = str(int(time.time()))
        has_call_ended = any(getattr(e, "event_type", "") == "call_ended" for e in self._lk_transcript.events)
        if not has_call_ended:
            self._lk_transcript.add_event(
                event_type="call_ended",
                details="Call session ended/disconnected."
            )

        # Rebuild full transcript from session history to avoid truncation of agent speech
        self._rebuild_transcript_from_history()

        # 3. Persist artefacts (audio + transcript)
        try:
            await self._persist_artefacts()
        except Exception as exc:
            _write_debug_log(f"Failed to persist artefacts: {exc}")
            logger.error("Failed to persist artefacts for call_id=%s: %s",
                         self._call_id, exc)

        # 4. Save final CallTranscript object to storage (Redis/memory)
        try:
            from storage import save_call_transcript
            await save_call_transcript(self._user_id, self._lk_transcript)
            logger.info("CallTranscript saved to storage on shutdown for user_id=%s", self._user_id)
        except Exception as exc:
            logger.error("Failed to save CallTranscript on shutdown: %s", exc)

    async def _persist_artefacts(self) -> None:
        recordings_dir  = os.path.abspath(_RECORDINGS_DIR)
        transcripts_dir = os.path.abspath(_TRANSCRIPTS_DIR)
        os.makedirs(recordings_dir,  exist_ok=True)
        os.makedirs(transcripts_dir, exist_ok=True)

        await self._persist_audio(recordings_dir)
        self._persist_transcript(transcripts_dir)
        _store_clear(self._call_id)

        logger.info("Artefacts finalised: call_id=%s user_id=%s",
                    self._call_id, self._user_id)

    async def _persist_audio(
        self,
        recordings_dir: str,
    ) -> None:
        logger.info(
            "--- _persist_audio called! self._user_pcm length: %d chunks, self._agent_pcm length: %d chunks ---",
            len(self._user_pcm),
            len(self._agent_pcm),
        )
        _write_debug_log(f"_persist_audio started: user_pcm={len(self._user_pcm)} chunks, agent_pcm={len(self._agent_pcm)} chunks")
        user_wav  = os.path.join(recordings_dir, f"{self._call_id}_user.wav")
        agent_wav = os.path.join(recordings_dir, f"{self._call_id}_agent.wav")
        mixed_mp3 = os.path.join(recordings_dir, f"{self._call_id}.mp3")

        # Use the detected sample rates (read from actual audio frames)
        user_rate  = self._user_sample_rate
        agent_rate = self._agent_sample_rate
        # For mixing we use the higher of the two rates
        mix_rate   = max(user_rate, agent_rate)

        logger.info(
            "Persisting audio: user=%d chunks (%dHz), agent=%d chunks (%dHz), mix_rate=%dHz",
            len(self._user_pcm), user_rate,
            len(self._agent_pcm), agent_rate,
            mix_rate,
        )

        user_ok  = _write_wav(
            self._user_pcm, user_wav,
            user_rate, _SAMPLE_WIDTH, self._user_num_channels,
        )
        agent_ok = _write_wav(
            self._agent_pcm, agent_wav,
            agent_rate, _SAMPLE_WIDTH, self._agent_num_channels,
        )
        _write_debug_log(f"WAV write results: user_ok={user_ok}, agent_ok={agent_ok}")

        if not (user_ok or agent_ok):
            _write_debug_log("No audio captured for call_id")
            logger.warning("No audio captured for call_id=%s", self._call_id)
            return

        loop = asyncio.get_running_loop()
        mixed_ok: bool = await loop.run_in_executor(
            None,
            _mix_to_mp3,
            user_wav  if user_ok  else "",
            agent_wav if agent_ok else "",
            mixed_mp3,
            mix_rate,
        )
        _write_debug_log(f"Audio mixing result: mixed_ok={mixed_ok}")

        if mixed_ok:
            _log_audio_quality(mixed_mp3, self._call_id)

        # Remove temp WAVs only if mix succeeded, so we don't lose data on failure
        if mixed_ok:
            for tmp in (user_wav, agent_wav):
                try:
                    if os.path.isfile(tmp):
                        os.remove(tmp)
                except OSError:
                    pass

    def _persist_transcript(self, transcripts_dir: str) -> None:
        path = _store_flush_transcript(self._call_id, transcripts_dir)
        if path:
            logger.info("Transcript saved: %s", path)
            return

        # Fallback: serialise CallTranscript object directly
        fallback = os.path.join(transcripts_dir, f"{self._call_id}.json")
        try:
            payload = (self._lk_transcript.to_dict()
                       if hasattr(self._lk_transcript, "to_dict") else {})
            if not payload:
                payload = {
                    "user_id":      self._lk_transcript.user_id,
                    "phone_number": getattr(self._lk_transcript, "phone_number", ""),
                    "events": [
                        e.__dict__ if hasattr(e, "__dict__") else e
                        for e in (getattr(self._lk_transcript, "events", None) or [])
                    ],
                }
            with open(fallback, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            logger.info("Fallback transcript saved: %s", fallback)
        except Exception as exc:
            logger.error("Fallback transcript write failed: %s", exc)