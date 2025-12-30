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

def route_after_start(state: HotelState) -> str:
    """
    Route based on request intent.
    """
    if state.request.intent == "feedback":
        return "customer_service"
    return "booking"


def route_after_booking(state: HotelState) -> str:
    """
    Decide next step after booking.
    """
    if state.booking.status in {"Confirmed", "Modified"}:
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

    workflow.add_node("booking", booking_agent)
    workflow.add_node("housekeeping", housekeeping_agent)
    workflow.add_node("customer_service", customer_service_agent)

    # Single START node with conditional routing
    workflow.add_conditional_edges(
        START,
        route_after_start,
        {
            "booking": "booking",
            "customer_service": "customer_service",
        },
    )

    workflow.add_conditional_edges(
        "booking",
        route_after_booking,
        {
            "housekeeping": "housekeeping",
            "customer_service": "customer_service",
        },
    )

    workflow.add_edge("housekeeping", "customer_service")
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
            "Agents:\n"
            "  • Booking Agent        → create / update / cancel bookings\n"
            "  • Housekeeping Agent   → room preparation & cleanup\n"
            "  • Customer Service     → AI-powered guest communication\n\n"
            "Examples:\n"
            "  Booking:\n"
            "    python -m src.main --action create --room-type Deluxe --nights 2\n\n"
            "  Feedback only:\n"
            "    python -m src.main --complaint \"Great service!\""
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # -----------------------------
    # Booking arguments
    # -----------------------------

    parser.add_argument("--action", choices=["create", "update", "cancel"], default="create")
    parser.add_argument("--booking-id", type=str, help="Required for update or cancel")
    parser.add_argument("--customer", type=str, default="Alice Johnson")
    parser.add_argument("--room-type", choices=["Standard", "Deluxe", "Suite"], default="Deluxe")
    parser.add_argument("--nights", type=int, default=2)

    # -----------------------------
    # Customer service arguments
    # -----------------------------

    parser.add_argument("--complaint", type=str, help="Complaint or compliment")

    # -----------------------------
    # Debug
    # -----------------------------

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print("🏨 Hotel Management System - Async Multi-Agent Demo")
    print("=" * 65)

    # -----------------------------
    # Detect intent
    # -----------------------------

    intent = "feedback" if args.complaint else "booking"

    # Validation for update / cancel
    if intent == "booking" and args.action in {"update", "cancel"} and not args.booking_id:
        parser.error("--booking-id is required for update or cancel actions")

    # -----------------------------
    # Build typed request
    # -----------------------------

    request = RequestState(
        intent=intent,
        action=args.action if intent == "booking" else None,
        booking_id=args.booking_id if intent == "booking" else None,
        customer=args.customer,
        room_type=args.room_type if intent == "booking" else None,
        nights=args.nights if intent == "booking" else None,
        check_in=(
            (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            if intent == "booking"
            else None
        ),
        complaint=args.complaint,
        special_requests=["late_checkout", "extra_towels"] if intent == "booking" else [],
    )

    if args.debug:
        print("📥 Input Request:")
        print(request.model_dump())
        print("-" * 65)

    # -----------------------------
    # Initialize state & run graph
    # -----------------------------

    initial_state = HotelState(request=request)
    graph = build_graph()
    final_state: HotelState = await graph.ainvoke(initial_state)

    # -----------------------------------------------------
    # 🔹 Output Results
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("📊 WORKFLOW RESULTS")
    print("=" * 65)

    if intent == "booking":
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