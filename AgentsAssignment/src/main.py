"""
🏨 Multi-Agent Hotel Management System (Async, LangGraph)

Final orchestration layer connecting all agents using LangGraph.
"""

import argparse
import asyncio
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, START, END

from src.models.state import HotelState
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
    Decide next step after housekeeping.
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
            "🏨 Multi-Agent Hotel Management System (Async)\n\n"
            "This application simulates a hotel workflow using async agents:\n"
            "  • Booking Agent – handles room reservations\n"
            "  • Housekeeping Agent – prepares rooms after booking\n"
            "  • Customer Service Agent – AI-powered responses via EPAM DIAL\n\n"
            "The system demonstrates LangGraph-based orchestration,\n"
            "shared state management, and deterministic routing logic."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--customer",
        type=str,
        default="Alice Johnson",
        help="Customer full name (default: Alice Johnson)",
    )

    parser.add_argument(
        "--room-type",
        type=str,
        choices=["Standard", "Deluxe", "Suite"],
        default="Deluxe",
        help="Room type to book: Standard | Deluxe | Suite (default: Deluxe)",
    )

    parser.add_argument(
        "--nights",
        type=int,
        default=2,
        help="Number of nights for the stay (default: 2)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (prints full workflow state)",
    )

    args = parser.parse_args()

    print("🏨 Hotel Management System - Async Multi-Agent Demo")
    print("=" * 55)

    request_data = {
        "customer": args.customer,
        "room_type": args.room_type,
        "nights": args.nights,
        "check_in": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "special_requests": ["late_checkout", "extra_towels"],
    }

    if args.debug:
        print("📥 Input Request:")
        print(request_data)
        print("-" * 40)

    # Initialize state
    initial_state = HotelState(request=request_data)

    # Build & run graph
    graph = build_graph()
    final_state = await graph.ainvoke(initial_state)

    # -----------------------------------------------------
    # 🔹 Output Results
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("📊 WORKFLOW RESULTS")
    print("=" * 55)

    print("\n🛎️ Booking:")
    print(final_state["booking"])

    print("\n🧹 Housekeeping:")
    print(final_state["housekeeping"])

    print("\n🎧 Customer Service Messages:")
    for msg in final_state["customer_service"].messages:
        print(f"- {msg}")

    if final_state.get("errors"):
        print("\n⚠️ Errors:")
        for err in final_state["errors"]:
            print(f"- {err}")

    if args.debug:
        print("\n🔍 Full State Dump:")
        print(final_state)


# ---------------------------------------------------------
# 🔹 Python Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
