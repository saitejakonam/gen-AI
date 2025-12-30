"""
Housekeeping Agent

Responsibilities:
- Prepare rooms after confirmed bookings
- Update room status
- Create cleaning schedules
- Handle post-checkout cleanup
"""

from datetime import datetime, timedelta
from src.models.state import (
    HotelState,
    CleaningTask,
)


# ---------------------------------------------------------
# 🔹 Helper: Check if checkout has passed
# ---------------------------------------------------------

def is_checkout_passed(check_in: str, nights: int) -> bool:
    """
    Determines whether the checkout time has passed.
    """
    checkout_time = datetime.fromisoformat(check_in) + timedelta(days=nights)
    return datetime.utcnow() > checkout_time


# ---------------------------------------------------------
# 🔹 Housekeeping Agent (ASYNC)
# ---------------------------------------------------------

async def housekeeping_agent(state: HotelState) -> HotelState:
    print("🧹 Housekeeping Agent: Processing room status...")

    try:
        booking = state.booking

        # -------------------------------------------------
        # 🔹 Skip if booking not active
        # -------------------------------------------------
        if booking.status not in {"Confirmed", "Modified"}:
            state.housekeeping.room_status = "Skipped"
            print("⏭️ Housekeeping skipped (booking inactive)")
            return state

        details = booking.details
        room_number = details.get("room_number")
        check_in = details.get("check_in")
        nights = int(details.get("nights", 1))

        if not room_number or not check_in:
            state.errors.append("Housekeeping error: Missing booking details")
            state.housekeeping.room_status = "Skipped"
            return state

        # -------------------------------------------------
        # 🔹 POST-CHECKOUT CLEANUP
        # -------------------------------------------------
        if is_checkout_passed(check_in, nights):
            print(f"🧾 Checkout passed for room {room_number}. Scheduling cleanup.")

            state.housekeeping.room_status = "Dirty"

            cleaning_task = CleaningTask(
                room_number=room_number,
                scheduled_at=datetime.utcnow(),
                priority="High",
                assigned_staff="HK-STAFF-02",
            )
            state.housekeeping.schedule.append(cleaning_task)

            state.housekeeping.room_status = "Cleaning"
            state.housekeeping.room_status = "Cleaned"
            state.housekeeping.room_ready = True
            state.housekeeping.status = "Post-checkout cleaned"

            print(f"✅ Post-checkout cleaning completed for room {room_number}")
            return state

        # -------------------------------------------------
        # 🔹 PRE-CHECK-IN PREPARATION
        # -------------------------------------------------
        if state.housekeeping.room_status != "Cleaned":
            state.housekeeping.room_status = "Dirty"

            cleaning_task = CleaningTask(
                room_number=room_number,
                scheduled_at=datetime.utcnow() + timedelta(minutes=30),
                priority="Normal",
                assigned_staff="HK-STAFF-01",
            )
            state.housekeeping.schedule.append(cleaning_task)

            state.housekeeping.room_status = "Cleaning"
            state.housekeeping.room_status = "Cleaned"
            state.housekeeping.room_ready = True
            state.housekeeping.room_status = "Cleaned"

            print(f"✅ Room {room_number} cleaned and ready")

    except Exception as e:
        state.errors.append(f"Housekeeping error: {str(e)}")
        state.housekeeping.room_status = "Failed"
        print(f"🔥 Housekeeping Agent error: {e}")

    return state
