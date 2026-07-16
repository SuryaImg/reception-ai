import asyncio
import os
from dotenv import load_dotenv
from livekit import rtc, api

load_dotenv()

async def main():
    print("Getting token for test room...")
    lk_api = api.LiveKitAPI(
        url=os.environ["LIVEKIT_URL"],
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    
    room_name = "hospital-call-test-panic123"
    token = api.AccessToken(
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"]
    ).with_grants(api.VideoGrants(room_join=True, room=room_name)).with_identity("sip_test_user").to_jwt()
    await lk_api.aclose()

    print("Connecting as sip_test_user...")
    room = rtc.Room()
    await room.connect(os.environ["LIVEKIT_URL"], token)
    
    print("Publishing a mock audio track...")
    source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
    await room.local_participant.publish_track(track)
    
    print("Waiting 10 seconds for agent to connect and start speaking...")
    await asyncio.sleep(10)
    
    print("Simulating SUDDEN disconnect (exiting without cleanup)...")
    # By forcing a disconnect, we see if the agent process panics.
    await room.disconnect()
    print("Done. Check the agent console for panic.")

if __name__ == "__main__":
    asyncio.run(main())