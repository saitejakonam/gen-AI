"""
Customer Service Agent

Responsibilities:
- Provide booking information
- Detect complaint vs compliment using EPAM DIAL
- Resolve complaints deterministically
- Acknowledge compliments professionally
- Generate AI-powered customer responses via EPAM DIAL (async)
"""

import uuid
from datetime import datetime

from src.models.state import (
    HotelState,
    Complaint,
)


# ---------------------------------------------------------
# 🔹 Customer Service Agent (ASYNC)
# ---------------------------------------------------------

async def customer_service_agent(state: HotelState) -> HotelState:
    print("🎧 Customer Service Agent: Processing customer interaction...")

    try:
        booking = state.booking
        housekeeping = state.housekeeping
        request = state.request
        dial = state.dial_client

        customer_name = request.customer
        customer_text = request.complaint or request.message

        # -------------------------------------------------
        # 🔹 Intent Detection (Complaint / Compliment / Other)
        # -------------------------------------------------
        if customer_text:
            intent = await dial.classify_intent_async(customer_text)
            print(f"🧠 Detected customer intent: {intent}")

            # ---------------------------------------------
            # 🔴 COMPLAINT
            # ---------------------------------------------
            if intent == "COMPLAINT":
                complaint_id = f"CMP-{uuid.uuid4().hex[:6].upper()}"

                complaint = Complaint(
                    complaint_id=complaint_id,
                    type=customer_text,
                    created_at=datetime.utcnow(),
                    status="Open",
                )

                state.customer_service.complaints.append(complaint)

                # Deterministic resolution
                resolution = (
                    "We sincerely apologize for the inconvenience. "
                    "Our team has initiated a resolution, and a partial refund "
                    "or service recovery will be provided."
                )

                complaint.status = "Resolved"
                state.customer_service.resolutions[complaint_id] = resolution

                ai_message = await dial.generate_response_async(
                    context="Hotel complaint resolution",
                    customer_query=(
                        f"Customer Name: {customer_name}\n"
                        f"Complaint: {customer_text}\n"
                        f"Resolution: {resolution}"
                    ),
                )

                state.customer_service.messages.append(ai_message)
                print(f"📝 Complaint resolved: {complaint_id}")
                return state

            # ---------------------------------------------
            # 🟢 COMPLIMENT
            # ---------------------------------------------
            if intent == "COMPLIMENT":
                ai_message = await dial.generate_response_async(
                    context="Hotel customer appreciation",
                    customer_query=(
                        f"Customer Name: {customer_name}\n"
                        f"Message: {customer_text}\n"
                        "Respond with gratitude and professionalism."
                    ),
                )

                state.customer_service.messages.append(ai_message)
                print("💚 Compliment acknowledged")
                return state

        # -------------------------------------------------
        # 🔹 Booking-Based Messaging (Default Flow)
        # -------------------------------------------------

        if booking.status == "Confirmed":
            details = booking.details
            message = f"""
Hello {customer_name},

Your booking is confirmed! Here are the details of your stay:

- Booking ID: {details.get("booking_id")}
- Room Type: {details.get("room_type")}
- Room Number: {details.get("room_number")}
- Check-in Date: {details.get("check_in")}
- Nights: {details.get("nights")}
- Total Cost: ${details.get("total_cost")}

Room Status: {housekeeping.room_status}

We look forward to welcoming you to Stay Inn!
"""

        elif booking.status == "Modified":
            message = f"""
Hello {customer_name},

Your booking has been successfully updated.
Please review the revised reservation details.
"""

        elif booking.status == "Cancelled":
            message = f"""
Hello {customer_name},

Your booking has been cancelled as requested.
If you would like help with a new reservation, we are happy to assist.
"""

        else:  # Failed booking
            message = f"""
Hello {customer_name},

Unfortunately, we were unable to complete your booking.

Reason:
{state.errors[-1] if state.errors else "Unknown error"}

Please let us know if you'd like help with alternative options.
"""

        ai_response = await dial.generate_response_async(
            context="Hotel booking communication",
            customer_query=message,
        )

        state.customer_service.messages.append(ai_response)
        print("✅ Customer response generated")

    except Exception as e:
        state.errors.append(f"Customer service error: {str(e)}")
        print(f"🔥 Customer Service Agent error: {e}")

    return state
