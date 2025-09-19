"""Event taxonomy definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class EventType:
    name: str
    description: str
    weight: float


TAXONOMY: Dict[str, EventType] = {
    "capacity_increase": EventType(
        name="capacity_increase",
        description="Confirmed expansion of manufacturing or compute capacity",
        weight=-0.6,
    ),
    "capacity_reduction": EventType(
        name="capacity_reduction",
        description="Reduction or shutdown of capacity impacting supply",
        weight=0.7,
    ),
    "shortage": EventType(
        name="shortage",
        description="Component or service shortage increasing scarcity",
        weight=0.9,
    ),
    "surplus": EventType(
        name="surplus",
        description="Supply surplus or easing of constraints",
        weight=-0.5,
    ),
    "capex_increase": EventType(
        name="capex_increase",
        description="Capital expenditure increase to expand infrastructure",
        weight=-0.4,
    ),
    "capex_cut": EventType(
        name="capex_cut",
        description="Capex cuts or delays signalling weaker supply expansion",
        weight=0.4,
    ),
    "outage": EventType(
        name="outage",
        description="Datacenter or service outage reducing availability",
        weight=1.0,
    ),
    "power_constraint": EventType(
        name="power_constraint",
        description="Power limitations affecting datacenter scaling",
        weight=0.8,
    ),
    "export_control": EventType(
        name="export_control",
        description="Regulatory export controls limiting supply",
        weight=0.85,
    ),
    "partnership": EventType(
        name="partnership",
        description="Strategic partnerships that may boost demand or supply",
        weight=0.3,
    ),
    "demand_spike": EventType(
        name="demand_spike",
        description="Evidence of sharp increase in demand for compute/services",
        weight=0.6,
    ),
    "demand_drop": EventType(
        name="demand_drop",
        description="Demand weakening for compute/services",
        weight=-0.6,
    ),
}


def taxonomy_names() -> List[str]:
    return list(TAXONOMY.keys())


__all__ = ["EventType", "TAXONOMY", "taxonomy_names"]
