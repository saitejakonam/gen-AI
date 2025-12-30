"""
Pydantic-based shared state for the Multi-Agent Hotel Management System.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from src.utils.dial_client import DIALClient


# ---------------------------------------------------------
# 🔹 Request State
# ---------------------------------------------------------

class RequestState(BaseModel):
    """
    Normalized user request state.
    """
    intent: str                                  # booking | feedback

    # Booking-related (ONLY for booking intent)
    action: Optional[str] = None                 # create | update | cancel
    customer: Optional[str] = None
    room_type: Optional[str] = None
    nights: Optional[int] = None
    check_in: Optional[str] = None
    booking_id: Optional[str] = None

    # Customer service
    complaint: Optional[str] = None
    message: Optional[str] = None

    # Misc
    special_requests: List[str] = Field(default_factory=list)

# ---------------------------------------------------------
# 🔹 Booking State Models
# ---------------------------------------------------------

class BookingHistoryEntry(BaseModel):
    action: str                            # created | updated | cancelled
    timestamp: datetime
    details: Dict[str, Any]


class BookingState(BaseModel):
    status: Optional[str] = None           # Confirmed | Failed | Modified | Cancelled
    details: Dict[str, Any] = Field(default_factory=dict)
    history: List[BookingHistoryEntry] = Field(default_factory=list)


# ---------------------------------------------------------
# 🔹 Housekeeping State Models
# ---------------------------------------------------------

class CleaningTask(BaseModel):
    room_number: str
    scheduled_at: datetime
    priority: str = "Normal"
    assigned_staff: Optional[str] = None


class HousekeepingState(BaseModel):
    room_status: Optional[str] = None      # Dirty | Cleaning | Cleaned | Maintenance
    room_ready: bool = False
    schedule: List[CleaningTask] = Field(default_factory=list)


# ---------------------------------------------------------
# 🔹 Customer Service State Models
# ---------------------------------------------------------

class Complaint(BaseModel):
    complaint_id: str
    category: str                    # complaint | compliment
    description: str
    status: str = "Open"
    created_at: datetime


class CustomerServiceState(BaseModel):
    messages: List[str] = Field(default_factory=list)
    complaints: List[Complaint] = Field(default_factory=list)
    resolutions: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------
# 🔹 Main Shared LangGraph State
# ---------------------------------------------------------

class HotelState(BaseModel):
    request: RequestState
    booking: BookingState = Field(default_factory=BookingState)
    housekeeping: HousekeepingState = Field(default_factory=HousekeepingState)
    customer_service: CustomerServiceState = Field(default_factory=CustomerServiceState)
    errors: List[str] = Field(default_factory=list)
    workflow_step: int = 0

    dial_client: DIALClient = Field(
        default_factory=DIALClient,
        exclude=True
    )

    class Config:
        arbitrary_types_allowed = True
