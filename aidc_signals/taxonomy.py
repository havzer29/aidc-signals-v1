"""Event taxonomy definitions with supply/demand metadata."""
from __future__ import annotations

"""Event taxonomy definitions with supply/demand metadata."""

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class EventType:
    """Definition of an extractable supply/demand event type."""

    name: str
    description: str
    role: str  # "S" for supply, "D" for demand
    impact: float  # direction of supply/demand change when direction==1
    weight: float  # relative magnitude of the event

    @property
    def is_supply(self) -> bool:
        return self.role.upper() == "S"

    @property
    def is_demand(self) -> bool:
        return self.role.upper() == "D"


TAXONOMY: Dict[str, EventType] = {
    "capacity_up": EventType(
        name="capacity_up",
        description="Expansion of compute or manufacturing capacity coming online",
        role="S",
        impact=1.0,
        weight=0.9,
    ),
    "capacity_down": EventType(
        name="capacity_down",
        description="Capacity reduction, shutdown or slower ramp of infrastructure",
        role="S",
        impact=-1.0,
        weight=1.1,
    ),
    "new_datacenter": EventType(
        name="new_datacenter",
        description="New datacenter or fab build/expansion announced",
        role="S",
        impact=1.0,
        weight=0.8,
    ),
    "outage": EventType(
        name="outage",
        description="Unplanned outage or downtime reducing available capacity",
        role="S",
        impact=-1.0,
        weight=1.2,
    ),
    "power_constraint": EventType(
        name="power_constraint",
        description="Power or energy constraints limiting operations",
        role="S",
        impact=-1.0,
        weight=1.0,
    ),
    "lead_time_up": EventType(
        name="lead_time_up",
        description="Lead times increasing for products or services",
        role="S",
        impact=-1.0,
        weight=0.9,
    ),
    "lead_time_down": EventType(
        name="lead_time_down",
        description="Lead times easing, availability improving",
        role="S",
        impact=1.0,
        weight=0.7,
    ),
    "export_control": EventType(
        name="export_control",
        description="Regulation or export control affecting supply availability",
        role="S",
        impact=-1.0,
        weight=1.0,
    ),
    "pricing_up": EventType(
        name="pricing_up",
        description="Price increases driven by constrained supply",
        role="S",
        impact=-1.0,
        weight=0.85,
    ),
    "pricing_down": EventType(
        name="pricing_down",
        description="Price decreases signalling abundant supply",
        role="S",
        impact=1.0,
        weight=0.7,
    ),
    "capex_up": EventType(
        name="capex_up",
        description="Capital expenditure acceleration expanding future capacity",
        role="S",
        impact=1.0,
        weight=0.75,
    ),
    "capex_down": EventType(
        name="capex_down",
        description="Capex cuts, delays or cancellations limiting capacity",
        role="S",
        impact=-1.0,
        weight=0.8,
    ),
    "demand_up": EventType(
        name="demand_up",
        description="Broad-based demand growth for AI/datacenter offerings",
        role="D",
        impact=1.0,
        weight=1.0,
    ),
    "demand_down": EventType(
        name="demand_down",
        description="Demand softening for compute or cloud workloads",
        role="D",
        impact=-1.0,
        weight=1.0,
    ),
    "order_win": EventType(
        name="order_win",
        description="Large order, design win or partnership adding demand",
        role="D",
        impact=1.0,
        weight=0.95,
    ),
    "order_loss": EventType(
        name="order_loss",
        description="Order cancellation or loss reducing demand",
        role="D",
        impact=-1.0,
        weight=0.95,
    ),
    "partnership": EventType(
        name="partnership",
        description="Strategic partnership likely to lift usage",
        role="D",
        impact=1.0,
        weight=0.7,
    ),
    "customer_shift": EventType(
        name="customer_shift",
        description="Customer migration to alternatives impacting demand",
        role="D",
        impact=-1.0,
        weight=0.7,
    ),
    "macro_signal": EventType(
        name="macro_signal",
        description="Macro datapoint influencing demand or supply balance",
        role="D",
        impact=1.0,
        weight=0.6,
    ),
}


def taxonomy_names() -> List[str]:
    return list(TAXONOMY.keys())


def taxonomy_prompt() -> str:
    """Render a human readable taxonomy table for prompting the LLM."""

    rows: List[str] = []
    for event in TAXONOMY.values():
        rows.append(
            f"- {event.name}: role={event.role}, impact_dir={event.impact:+.0f}, "
            f"weight={event.weight:.2f} — {event.description}"
        )
    return "\n".join(rows)


def is_valid_event(name: str) -> bool:
    return name in TAXONOMY


def iter_supply_events() -> Iterable[EventType]:
    return (event for event in TAXONOMY.values() if event.is_supply)


def iter_demand_events() -> Iterable[EventType]:
    return (event for event in TAXONOMY.values() if event.is_demand)


__all__ = [
    "EventType",
    "TAXONOMY",
    "taxonomy_names",
    "taxonomy_prompt",
    "is_valid_event",
    "iter_supply_events",
    "iter_demand_events",
]
