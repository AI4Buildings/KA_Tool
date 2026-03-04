"""Abstract base class for all AHU components."""

from abc import ABC, abstractmethod
from src.thermodynamics.air_state import AirState, ComponentResult


class Component(ABC):
    """Abstract base class for AHU components.

    Each component takes an inlet air state and computes the optimal
    outlet state along with the associated energy demand.
    """

    @abstractmethod
    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  **kwargs) -> ComponentResult:
        """Calculate outlet air state and energy demand.

        Args:
            air_in: Inlet air state
            setpoint_T: Temperature setpoint [°C]
            setpoint_x: Absolute humidity setpoint [kg/kg]
            timestep_h: Time step duration [h]
            **kwargs: Additional component-specific parameters

        Returns:
            ComponentResult with outlet state and energy demand
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Component name for identification."""
