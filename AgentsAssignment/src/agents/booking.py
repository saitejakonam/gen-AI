"""
Booking Agent

Responsibilities:
- Create booking
- Update booking
- Cancel booking
- Enforce room availability
- Maintain booking history
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict

from src.models.state import HotelState, BookingHistoryEntry

DATA_FILE = Path("src/data/booked_rooms.json")


# ---------------------------------------------------------
# 🔹 Persistence Helpers
# ---------------------------------------------------------

def load_booked_rooms() -> Dict:
    if not DATA_FILE.exists():
        return {}
    return json.loads(DATA_FILE.read_text())


def save_booked_rooms(data: Dict):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------
# 🔹 Booking Agent (ASYNC)
# ---------------------------------------------------------

async def booking_agent(state: HotelState) -> HotelState:
    if state.request.intent != "booking":
        print("⏭️ Booking skipped (feedback-only request)")
        return state
    
    print("🏨 Booking Agent: Processing request...")

    try:
        request = state.request
        action = request.action  # create | update | cancel
        room_type = request.room_type
        nights = request.nights or 1
        customer = request.customer
        check_in = request.check_in
        booking_id = request.booking_id

        available_rooms = {
            "Standard": ["101", "102", "103"],
            "Deluxe": ["201", "202"],
            "Suite": ["301"],
        }

        booked_rooms = load_booked_rooms()

        # -------------------------------------------------
        # 🔹 CREATE BOOKING
        # -------------------------------------------------
        if action == "create":
            for room in available_rooms.get(room_type, []):
                if room not in booked_rooms:
                    booking_id = f"BK{uuid.uuid4().hex[:8].upper()}"

                    booked_rooms[room] = {
                        "booking_id": booking_id,
                        "check_in": check_in,
                        "nights": nights,
                    }
                    save_booked_rooms(booked_rooms)

                    state.booking.status = "Confirmed"
                    state.booking.details = {
                        "booking_id": booking_id,
                        "customer": customer,
                        "room_type": room_type,
                        "room_number": room,
                        "check_in": check_in,
                        "nights": nights,
                        "total_cost": 250 * nights if room_type != "Standard" else 150 * nights,
                        "created_at": datetime.utcnow().isoformat(),
                    }

                    state.booking.history.append(
                        BookingHistoryEntry(
                            action="created",
                            timestamp=datetime.utcnow(),
                            details=state.booking.details,
                        )
                    )

                    print(f"✅ Booking confirmed: {booking_id} | Room {room}")
                    return state

            # No room available
            state.booking.status = "Failed"
            state.errors.append(f"No {room_type} rooms available")
            print(f"❌ No {room_type} rooms available")
            return state

        # -------------------------------------------------
        # 🔹 UPDATE BOOKING
        # -------------------------------------------------
        if action == "update":
            booking_id = request.booking_id   # ✅ FIX

            for room, data in booked_rooms.items():
                if data["booking_id"] == booking_id:
                    data["nights"] = nights
                    save_booked_rooms(booked_rooms)

                    state.booking.status = "Modified"
                    state.booking.details = {
                        "booking_id": booking_id,
                        "nights": nights,
                        "room_number": room,
                        "check_in": data["check_in"],
                    }

                    state.booking.history.append(
                        BookingHistoryEntry(
                            action="updated",
                            timestamp=datetime.utcnow(),
                            details={"nights": str(nights)},
                        )
                    )

                    print(f"🔄 Booking updated: {booking_id}")
                    return state

            state.booking.status = "Failed"
            state.errors.append("Booking not found")
            return state

        # -------------------------------------------------
        # 🔹 CANCEL BOOKING
        # -------------------------------------------------
        if action == "cancel":
            booking_id = request.booking_id   # ✅ FIX

            for room in list(booked_rooms.keys()):
                if booked_rooms[room]["booking_id"] == booking_id:
                    del booked_rooms[room]
                    save_booked_rooms(booked_rooms)

                    state.booking.status = "Cancelled"

                    state.booking.history.append(
                        BookingHistoryEntry(
                            action="cancelled",
                            timestamp=datetime.utcnow(),
                            details={"booking_id": booking_id},
                        )
                    )

                    print(f"❌ Booking cancelled: {booking_id}")
                    return state

            state.booking.status = "Failed"
            state.errors.append("Booking not found")
            return state

    except Exception as e:
        state.errors.append(str(e))
        state.booking.status = "Failed"
        print(f"🔥 Booking Agent error: {e}")

    return state
