"""
Booking Agent (Async)

Handles room availability checks and booking creation.
Prevents double-booking using JSON-based persistence.
"""

import uuid
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from src.models.state import HotelState


# ---------------------------------------------------------
# 🔹 Mock Data
# ---------------------------------------------------------

AVAILABLE_ROOMS: Dict[str, List[str]] = {
    "Standard": ["101", "102", "103"],
    "Deluxe": ["201", "202"],
    "Suite": ["301"],
}

ROOM_PRICING = {
    "Standard": 150,
    "Deluxe": 250,
    "Suite": 400,
}

# ---------------------------------------------------------
# 🔹 Persistence Setup
# ---------------------------------------------------------

DATA_DIR = Path("src/data")
DATA_DIR.mkdir(exist_ok=True)

BOOKED_ROOMS_FILE = DATA_DIR / "booked_rooms.json"


def load_booked_rooms() -> set:
    if not BOOKED_ROOMS_FILE.exists():
        return set()

    try:
        with open(BOOKED_ROOMS_FILE, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_booked_rooms(booked_rooms: set) -> None:
    with open(BOOKED_ROOMS_FILE, "w") as f:
        json.dump(sorted(booked_rooms), f, indent=2)


# Load booked rooms at startup
BOOKED_ROOMS = load_booked_rooms()


# ---------------------------------------------------------
# 🔹 Async Booking Agent
# ---------------------------------------------------------

async def booking_agent(state: HotelState) -> HotelState:
    print("🏨 Booking Agent: Processing reservation request...")

    try:
        request = state.request

        customer = request.get("customer")
        room_type = request.get("room_type", "Standard")
        nights = int(request.get("nights", 1))
        check_in = request.get(
            "check_in",
            datetime.now().strftime("%Y-%m-%d")
        )

        # -------------------------------------------------
        # Availability Check (Exclude Persisted Bookings)
        # -------------------------------------------------
        available_rooms = [
            room for room in AVAILABLE_ROOMS.get(room_type, [])
            if room not in BOOKED_ROOMS
        ]

        if not available_rooms:
            state.booking.status = "Failed"
            state.booking.details = {
                "reason": f"All {room_type} rooms are already booked"
            }
            state.errors.append(f"No available {room_type} rooms")

            print(f"❌ Booking failed: All {room_type} rooms are already booked")
            state.workflow_step = 1
            return state

        # -------------------------------------------------
        # Create Booking
        # -------------------------------------------------
        booking_id = f"BK{uuid.uuid4().hex[:8].upper()}"
        room_number = available_rooms[0]

        total_cost = ROOM_PRICING.get(room_type, 150) * nights

        # Persist booking
        BOOKED_ROOMS.add(room_number)
        save_booked_rooms(BOOKED_ROOMS)

        state.booking.status = "Confirmed"
        state.booking.details = {
            "booking_id": booking_id,
            "customer": customer,
            "room_type": room_type,
            "room_number": room_number,
            "check_in": check_in,
            "nights": nights,
            "total_cost": total_cost,
            "created_at": datetime.now().isoformat(),
        }

        print(f"✅ Booking confirmed: {booking_id}")
        print(f"   Room: {room_type} #{room_number}")
        print(f"   💾 Room {room_number} persisted as booked")

    except Exception as exc:
        state.booking.status = "Failed"
        state.booking.details = {"error": str(exc)}
        state.errors.append(f"Booking agent error: {exc}")

        print(f"❌ Booking Agent error: {exc}")

    state.workflow_step = 1
    return state
