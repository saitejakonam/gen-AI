"""
🏨 Multi-Agent Hotel Management System (Async, LangGraph)

Final orchestration layer connecting all agents using LangGraph.
Provides a CLI interface for interacting with the system.
"""

import argparse
import asyncio
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, START, END

from src.models.state import HotelState, RequestState
from src.agents.booking import booking_agent
from src.agents.housekeeping import housekeeping_agent
from src.agents.customer_service import customer_service_agent


# ---------------------------------------------------------
# 🔹 Routing Logic (SYNC by LangGraph design)
# ---------------------------------------------------------

def route_after_booking(state: HotelState) -> str:
    """
    Decide next step after booking.
    """
    if state.booking.status == "Confirmed":
        return "housekeeping"
    return "customer_service"


def route_after_housekeeping(state: HotelState) -> str:
    """
    Always proceed to customer service after housekeeping.
    """
    return "customer_service"


# ---------------------------------------------------------
# 🔹 Build LangGraph Workflow
# ---------------------------------------------------------

def build_graph():
    workflow = StateGraph(HotelState)

    # Register async agents
    workflow.add_node("booking", booking_agent)
    workflow.add_node("housekeeping", housekeeping_agent)
    workflow.add_node("customer_service", customer_service_agent)

    # Entry point
    workflow.add_edge(START, "booking")

    # Conditional routing after booking
    workflow.add_conditional_edges(
        "booking",
        route_after_booking,
        {
            "housekeeping": "housekeeping",
            "customer_service": "customer_service",
        },
    )

    # Housekeeping always leads to customer service
    workflow.add_edge("housekeeping", "customer_service")

    # End after customer service
    workflow.add_edge("customer_service", END)

    return workflow.compile()


# ---------------------------------------------------------
# 🔹 Async Main Entry Point
# ---------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        prog="Hotel Management System",
        description=(
            "🏨 Multi-Agent Hotel Management System (Async, LangGraph)\n\n"
            "This CLI application simulates a hotel workflow using async agents:\n\n"
            "  • Booking Agent        → create / update / cancel bookings\n"
            "  • Housekeeping Agent   → pre-checkin & post-checkout cleaning\n"
            "  • Customer Service     → AI-powered guest communication\n\n"
            "Key Design Principles:\n"
            "  • Deterministic routing (no AI decisions)\n"
            "  • Shared Pydantic state\n"
            "  • EPAM DIAL used only for customer messaging\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # -----------------------------
    # Booking-related arguments
    # -----------------------------

    parser.add_argument(
        "--action",
        choices=["create", "update", "cancel"],
        default="create",
        help=(
            "Booking action to perform:\n"
            "  create  → create a new booking (default)\n"
            "  update  → modify an existing booking\n"
            "  cancel  → cancel an existing booking"
        ),
    )

    parser.add_argument(
        "--booking-id",
        type=str,
        help="Booking ID (required for update or cancel actions)",
    )

    parser.add_argument(
        "--customer",
        type=str,
        default="Alice Johnson",
        help="Customer full name (default: Alice Johnson)",
    )

    parser.add_argument(
        "--room-type",
        choices=["Standard", "Deluxe", "Suite"],
        default="Deluxe",
        help="Room type to book (default: Deluxe)",
    )

    parser.add_argument(
        "--nights",
        type=int,
        default=2,
        help="Number of nights for the stay (default: 2)",
    )

    # -----------------------------
    # Customer Service arguments
    # -----------------------------

    parser.add_argument(
        "--complaint",
        type=str,
        help=(
            "Submit a customer complaint or compliment.\n"
            "Examples:\n"
            "  --complaint \"Room was not clean\"\n"
            "  --complaint \"Great service!\""
        ),
    )

    # -----------------------------
    # Debug / utility flags
    # -----------------------------

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (prints full workflow state)",
    )

    args = parser.parse_args()

    print("🏨 Hotel Management System - Async Multi-Agent Demo")
    print("=" * 65)

    # -----------------------------
    # Build typed request payload
    # -----------------------------

    request = RequestState(
        action=args.action,
        booking_id=args.booking_id,
        customer=args.customer,
        room_type=args.room_type,
        nights=args.nights,
        complaint=args.complaint,
        check_in=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        special_requests=["late_checkout", "extra_towels"],
    )

    if args.debug:
        print("📥 Input Request:")
        print(request.model_dump())
        print("-" * 55)

    # -----------------------------
    # Initialize shared state
    # -----------------------------

    initial_state = HotelState(request=request)

    # Build & run LangGraph
    graph = build_graph()
    final_state: HotelState = await graph.ainvoke(initial_state)

    # -----------------------------------------------------
    # 🔹 Output Results
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("📊 WORKFLOW RESULTS")
    print("=" * 65)

    print("\n🛎️ Booking State:")
    print(final_state["booking"].model_dump())

    print("\n🧹 Housekeeping State:")
    print(final_state["housekeeping"].model_dump())

    print("\n🎧 Customer Service Messages:")
    for msg in final_state["customer_service"].messages:
        print(f"- {msg}")

    if final_state["errors"]:
        print("\n⚠️ Errors:")
        for err in final_state["errors"]:
            print(f"- {err}")

    if args.debug:
        print("\n🔍 Full State Dump:")
        print(final_state.model_dump())


# ---------------------------------------------------------
# 🔹 Python Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
