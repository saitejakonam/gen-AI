"""
Customer Service Agent (Async + DIAL)

Generates customer-facing responses using EPAM DIAL.
"""

from src.models.state import HotelState


# ---------------------------------------------------------
# 🔹 Async Customer Service Agent
# ---------------------------------------------------------

async def customer_service_agent(state: HotelState) -> HotelState:
    """
    Async Customer Service Agent.

    Responsibilities:
    - Generate polite, professional customer responses
    - Use DIAL for language generation
    - Update only customer_service state
    """

    print("🎧 Customer Service Agent: Generating customer response...")

    try:
        # -------------------------------------------------
        # Build Context for DIAL
        # -------------------------------------------------
        booking_status = state.booking.status
        booking_details = state.booking.details
        housekeeping_status = state.housekeeping.status

        context = (
            f"Booking Status: {booking_status}\n"
            f"Booking Details: {booking_details}\n"
            f"Housekeeping Status: {housekeeping_status}"
        )

        customer_query = (
            state.request.get("customer_query")
            or "Please confirm my booking and room status."
        )

        # -------------------------------------------------
        # Call DIAL (ASYNC)
        # -------------------------------------------------
        response = await state.dial_client.generate_response_async(
            context=context,
            customer_query=customer_query
        )

        # -------------------------------------------------
        # Update Customer Service State
        # -------------------------------------------------
        state.customer_service.messages.append(response)

        if booking_status == "Confirmed":
            state.customer_service.resolution = "Booking confirmed and room prepared"
        else:
            state.customer_service.resolution = "Booking issue communicated to customer"

        print("✅ Customer response generated via DIAL")

    except Exception as exc:
        error_message = f"Customer service agent error: {exc}"
        state.errors.append(error_message)

        state.customer_service.messages.append(
            "We encountered an issue while processing your request. "
            "Please contact the front desk for assistance."
        )
        state.customer_service.resolution = "Error handled gracefully"

        print(f"❌ Customer Service Agent error: {exc}")

    state.workflow_step = 3
    return state
