"""
Housekeeping Agent (Async)

Handles room preparation and readiness after booking confirmation.
"""

from src.models.state import HotelState


# ---------------------------------------------------------
# 🔹 Async Housekeeping Agent
# ---------------------------------------------------------

async def housekeeping_agent(state: HotelState) -> HotelState:
    """
    Async Housekeeping Agent.

    Responsibilities:
    - Check booking status
    - Prepare room if booking is confirmed
    - Update housekeeping state deterministically
    """

    print("🧹 Housekeeping Agent: Checking room status...")

    try:
        # -------------------------------------------------
        # Skip if booking failed
        # -------------------------------------------------
        if state.booking.status != "Confirmed":
            state.housekeeping.status = "Skipped"
            state.housekeeping.room_ready = False

            print("ℹ️ Housekeeping skipped (booking not confirmed)")
            state.workflow_step = 2
            return state

        # -------------------------------------------------
        # Prepare Room (Mock Logic)
        # -------------------------------------------------
        booking_details = state.booking.details
        room_number = booking_details.get("room_number")

        # Simulate successful cleaning
        state.housekeeping.status = "Cleaned"
        state.housekeeping.room_ready = True

        print(f"✅ Room {room_number} cleaned and ready")

    except Exception as exc:
        state.housekeeping.status = "Pending"
        state.housekeeping.room_ready = False
        state.errors.append(f"Housekeeping agent error: {exc}")

        print(f"❌ Housekeeping Agent error: {exc}")

    state.workflow_step = 2
    return state
