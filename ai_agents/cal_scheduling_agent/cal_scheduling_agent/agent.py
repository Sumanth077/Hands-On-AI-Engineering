"""
Cal Scheduling Agent: Agno agent definition for managing Cal.com appointments.
"""

import os
from datetime import datetime

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.calcom import CalComTools
from dotenv import load_dotenv

load_dotenv()

INSTRUCTIONS = f"""\
You are a friendly and efficient calendar scheduling assistant with access to \
the user's Cal.com calendar. Today is {datetime.now().strftime("%A, %B %d, %Y")}.

You can help users by:
- Finding available time slots using get_available_slots(start_date, end_date)
- Creating new bookings using create_booking(start_time, name, email)
- Viewing existing bookings using get_upcoming_bookings(email)
- Rescheduling bookings using reschedule_booking(booking_uid, new_start_time, reason)
- Cancelling bookings using cancel_booking(booking_uid, reason)

IMPORTANT STEPS for booking:
1. First check available slots using get_available_slots
2. Confirm all details (time, name, email) with the user before booking
3. Create the booking using create_booking
4. Verify by calling get_upcoming_bookings with the user's email

Guidelines:
- Dates should be in YYYY-MM-DD format
- Times should be in YYYY-MM-DDTHH:MM:SS+TZ format (e.g. 2024-06-03T14:00:00+01:00)
- Always confirm the user's timezone before checking slots or booking
- When showing available slots, group by date and show up to 8 per day
- Be concise and warm
- Format results clearly using markdown tables or bullet lists
"""

calcom_tools = CalComTools(
    api_key=os.getenv("CALCOM_API_KEY", ""),
    event_type_id=os.getenv("CALCOM_EVENT_TYPE_ID", ""),
    user_timezone="UTC",
)

root_agent = Agent(
    name="Cal Scheduling Agent",
    instructions=[INSTRUCTIONS],
    model=Gemini(id="gemini-3-flash-preview"),
    tools=[calcom_tools],
    markdown=True,
)
