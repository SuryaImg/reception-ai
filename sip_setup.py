"""
sip_setup.py — Run ONCE to wire Vobiz → LiveKit SIP → your agent.

Usage:
    python sip_setup.py

Required .env vars:
    LIVEKIT_URL         wss://voise-assistant-q87tz4gn.livekit.cloud
    LIVEKIT_API_KEY
    LIVEKIT_API_SECRET
    LIVEKIT_SIP_URI     1smn4ta9dd5.sip.livekit.cloud   ← from LiveKit dashboard Settings
    VOBIZ_PHONE_NUMBER  +918065481170

NOTE: No SIP auth credentials are configured on the LiveKit trunk.
Vobiz's inbound-trunk forwarding does not support per-destination Digest auth,
so we rely on the SIP URI being unguessable (standard practice for hosted SIP).
"""

import asyncio
import os
from dotenv import load_dotenv
from livekit import api

load_dotenv()

LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# LIVEKIT_SIP_URI: copy from LiveKit dashboard → Settings → SIP URI field.
# It is NOT derived from LIVEKIT_URL — they are different subdomains.
# e.g. if dashboard shows "sip:1smn4ta9dd5.sip.livekit.cloud"
#      set LIVEKIT_SIP_URI=1smn4ta9dd5.sip.livekit.cloud
LIVEKIT_SIP_URI    = os.environ["LIVEKIT_SIP_URI"]

VOBIZ_PHONE_NUMBER = os.environ["VOBIZ_PHONE_NUMBER"]

# Must match agent_name= in agent.py WorkerOptions exactly
AGENT_NAME = "hospital-receptionist"


def _inspect_fields(cls):
    try:
        return {f.name for f in cls.DESCRIPTOR.fields}
    except Exception:
        return set()


async def main():
    lk = api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

    # ------------------------------------------------------------------
    # Guard: abort if a trunk already exists for this number
    # ------------------------------------------------------------------
    try:
        existing = await lk.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
    except AttributeError:
        existing = await lk.sip.list_sip_inbound_trunk(api.ListSIPInboundTrunkRequest())

    already = [t for t in existing.items if VOBIZ_PHONE_NUMBER in (t.numbers or [])]
    if already:
        print(f"STOP: {len(already)} trunk(s) already exist for {VOBIZ_PHONE_NUMBER}:")
        for t in already:
            print(f"     {t.sip_trunk_id}  ({t.name})")
        print()
        print("   Delete them in LiveKit dashboard → Telephony → SIP trunks,")
        print("   AND delete any existing dispatch rules, THEN re-run this script.")
        print("   Running now would create duplicates and break routing.")
        await lk.aclose()
        return

    # ------------------------------------------------------------------
    # 1. Create the inbound SIP trunk on LiveKit (no auth — Vobiz does
    #    not support per-destination Digest credentials on forwarding)
    # ------------------------------------------------------------------
    print("Creating LiveKit inbound SIP trunk ...")
    trunk = await lk.sip.create_inbound_trunk(
        api.CreateSIPInboundTrunkRequest(
            trunk=api.SIPInboundTrunkInfo(
                name="vobiz-hospital-receptionist",
                allowed_addresses=[],   # accept from any IP
                auth_username="",       # no Digest auth (Vobiz doesn't support it)
                auth_password="",
                numbers=[VOBIZ_PHONE_NUMBER],
            )
        )
    )
    trunk_id = trunk.sip_trunk_id
    print(f"  Trunk created: {trunk_id}")

    # ------------------------------------------------------------------
    # 2. Create the dispatch rule
    # ------------------------------------------------------------------
    print("Creating dispatch rule ...")

    req_fields = _inspect_fields(api.CreateSIPDispatchRuleRequest)
    individual = api.SIPDispatchRuleIndividual(room_prefix="hospital-call-")
    oneof_rule = api.SIPDispatchRule(dispatch_rule_individual=individual)

    if "name" in req_fields:
        dispatch_req = api.CreateSIPDispatchRuleRequest(
            name="hospital-receptionist-rule",
            trunk_ids=[trunk_id],
            attributes={"livekit.agent_name": AGENT_NAME},
            rule=oneof_rule,
        )
    else:
        dispatch_req = api.CreateSIPDispatchRuleRequest(
            rule=api.SIPDispatchRuleInfo(
                name="hospital-receptionist-rule",
                trunk_ids=[trunk_id],
                attributes={"livekit.agent_name": AGENT_NAME},
                rule=oneof_rule,
            )
        )

    try:
        rule_resp = await lk.sip.create_dispatch_rule(dispatch_req)
    except AttributeError:
        rule_resp = await lk.sip.create_sip_dispatch_rule(dispatch_req)

    print(f"  Dispatch rule created: {rule_resp.sip_dispatch_rule_id}")

    # ------------------------------------------------------------------
    # 3. Print Vobiz configuration instructions
    # ------------------------------------------------------------------
    sip_host = LIVEKIT_SIP_URI.lstrip("sip:").rstrip("/")

    print()
    print("=" * 60)
    print("LIVEKIT SIDE SETUP COMPLETE")
    print("=" * 60)
    print(f"  LiveKit SIP URI : sip:{sip_host}")
    print(f"  Trunk ID        : {trunk_id}")
    print(f"  Phone number    : {VOBIZ_PHONE_NUMBER}")
    print(f"  Agent name      : {AGENT_NAME}")
    print()
    print("-" * 60)
    print("NOW configure Vobiz inbound trunk:")
    print("-" * 60)
    print("  Vobiz console -> SIP Trunk -> Inbound Trunks -> your trunk")
    print("  -> URI Configuration -> Primary URI:")
    print()
    print(f"      {sip_host}:5060")
    print()
    print("  (Exactly that string -- no 'sip:' prefix, no quotes)")
    print()
    print("Then start your agent:")
    print("      python agent.py start")
    print()
    print(f"Dial {VOBIZ_PHONE_NUMBER} -- you should hear Nikita answer.")

    await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())