"""Setpoint logic - resolves supply air setpoints from room conditions."""

from src.thermodynamics import moist_air


def resolve_supply_setpoints(
    room_T: float,
    room_phi: float,
    supply_T: float | None = None,
    supply_phi: float | None = None,
    p: float = 101325.0,
) -> tuple[float, float]:
    """Resolve supply air setpoints (T, x) from room conditions.

    If supply setpoints are not given, use room conditions as defaults.

    Args:
        room_T: Room temperature setpoint [°C]
        room_phi: Room relative humidity setpoint [% RH]
        supply_T: Supply air temperature [°C], optional
        supply_phi: Supply air relative humidity [% RH], optional
        p: Atmospheric pressure [Pa]

    Returns:
        (T_supply [°C], x_supply [kg/kg])
    """
    T = supply_T if supply_T is not None else room_T
    phi = supply_phi if supply_phi is not None else room_phi
    x = moist_air.absolute_humidity(phi / 100.0, T, p)
    return T, x
