"""
Pydantic-based shared state for the Multi-Agent Hotel Management System.

This state object is passed and mutated across LangGraph async agents.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.utils.dial_client import DIALClient


# ---------------------------------------------------------
# 🔹 Sub-state models (agent-specific)
# ---------------------------------------------------------

class BookingState(BaseModel):
    status: Optional[str] = None          # Confirmed | Failed
    details: Optional[Dict] = None        # Booking metadata


class HousekeepingState(BaseModel):
    room_ready: bool = False
    status: Optional[str] = None          # Cleaned | Pending | Skipped


class CustomerServiceState(BaseModel):
    messages: List[str] = Field(default_factory=list)
    resolution: Optional[str] = None


# ---------------------------------------------------------
# 🔹 Main shared LangGraph state
# ---------------------------------------------------------

class HotelState(BaseModel):
    """
    Central shared state for all LangGraph agents.
    """

    # Original user request
    request: Dict

    # Agent states
    booking: BookingState = Field(default_factory=BookingState)
    housekeeping: HousekeepingState = Field(default_factory=HousekeepingState)
    customer_service: CustomerServiceState = Field(default_factory=CustomerServiceState)

    # Errors & workflow metadata
    errors: List[str] = Field(default_factory=list)
    workflow_step: int = 0

    # Runtime-only dependency (NOT serialized)
    dial_client: DIALClient = Field(
        default_factory=DIALClient,
        exclude=True
    )

    class Config:
        arbitrary_types_allowed = True


