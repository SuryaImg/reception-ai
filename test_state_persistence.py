
from tools import AppointmentState
from agent import update_agent_instructions, _human_voice_instructions
import asyncio

class MockAgent:
    def __init__(self):
        self.instructions = ""
        self.description = ""

    async def update_instructions(self, new_instructions):
        self.instructions = new_instructions

async def test_update_agent_instructions_already_captured():
    state = AppointmentState()
    state.booking_intent_detected = True
    state.caller_name = "Shubham"
    state.patient_name = "Rohan"
    state.reason = "High fever"
    state.department = "General Physician"

    agent = MockAgent()
    await update_agent_instructions(agent, state)
    
    instructions = agent.instructions
    
    assert "- Caller Name: Shubham (ALREADY CAPTURED - DO NOT ASK AGAIN)" in instructions, "Missing caller name assertion"
    assert "- Patient Name: Rohan (locked=False) (ALREADY CAPTURED - DO NOT ASK AGAIN)" in instructions, "Missing patient name assertion"
    assert "- Appointment Reason: High fever (ALREADY CAPTURED - DO NOT ASK AGAIN)" in instructions, "Missing reason assertion"
    assert "- Department / Service: General Physician (INFERRED FROM SYMPTOMS - DO NOT ASK AGAIN)" in instructions, "Missing inferred department assertion"
    assert "Since the user is booking an appointment, you MUST collect these missing details:" in instructions, "Missing booking intent true text"

async def test_update_agent_instructions_enquiry_intent():
    state = AppointmentState()
    state.booking_intent_detected = False

    agent = MockAgent()
    await update_agent_instructions(agent, state)
    
    instructions = agent.instructions
    
    assert "The user has NOT yet initiated a booking. They may just be making an enquiry." in instructions, "Missing enquiry text"
    assert "Do NOT ask for their name or start the booking flow yet." in instructions, "Missing enquiry instruction"

if __name__ == "__main__":
    asyncio.run(test_update_agent_instructions_already_captured())
    asyncio.run(test_update_agent_instructions_enquiry_intent())
    print("All tests passed.")