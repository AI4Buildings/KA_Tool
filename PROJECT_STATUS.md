# KA_Tool – Projektstatus (Stand: 2026-03-04)

## Übersicht

HVAC-Planungstool zur energetischen Analyse von Lüftungsanlagen (AHU).
Konvertierung eines MATLAB-Modells (Masterarbeit Dragosits, FH Burgenland 2023)
nach Python mit MCP-Server-Anbindung.

**Repository:** https://github.com/AI4Buildings/KA_Tool

---

## Implementierte Module

### Thermodynamik (`src/thermodynamics/`)
| Datei | Beschreibung | Status |
|---|---|---|
| `moist_air.py` | Magnus-Formel, Enthalpie, Sättigungsfeuchte, Taupunkt | ✅ Fertig |
| `air_state.py` | AirState-Dataclass mit Properties (phi, h, T_dew, x_sat) | ✅ Fertig |

### Wärmeübertrager (`src/heat_exchangers/`)
| Datei | Beschreibung | Status |
|---|---|---|
| `ntu_dry.py` | NTU Gegenstrom + Kreuzstrom, Kaup-Teillast (n=0.4) | ✅ Fertig |
| `ntu_wet.py` | NTU nass mit c_s, Kaup-Teillast (n=0.2113) | ✅ Fertig |
| `rotary.py` | ÖNORM EN 16798-5-1, 3 Rotortypen, 3 Regelungsmodi | ✅ Fertig |

### Komponenten (`src/components/`)
| Datei | Beschreibung | Status |
|---|---|---|
| `fan.py` | Ventilator mit Wärmeeintrag | ✅ Fertig |
| `frost_protection.py` | Elektrischer Vorerhitzer + Bypass | ✅ Fertig |
| `heat_recovery.py` | Platte, Rotation, KVS (als Component-Wrapper) | ✅ Fertig |
| `heating_coil.py` | VHR/NHR mit coil_type='pre'/'post' | ✅ Fertig |
| `cooling_coil.py` | Mit/ohne Entfeuchtung (nasser WÜ) | ✅ Fertig |
| `humidifier.py` | Spray (adiabat, Sättigungslimit) + Dampf | ✅ Fertig |
| `adiabatic_cooling.py` | Adiabate Abluft-Kühlung | ❌ Noch nicht implementiert |
| `recirculation_bypass.py` | Umluft-Bypass | ❌ Noch nicht implementiert |

### System (`src/system/`)
| Datei | Beschreibung | Status |
|---|---|---|
| `ahu_system.py` | Komponentenkette + VHR-Vorheizlogik | ✅ Fertig |
| `simulation.py` | Stündliche Simulation über Wetterdaten | ✅ Fertig |
| `setpoint_logic.py` | Sollwertberechnung | ✅ Fertig |

### Wetter & MCP (`src/weather/`, `src/mcp_server/`)
| Datei | Beschreibung | Status |
|---|---|---|
| `open_meteo.py` | Geocoding + stündliche Wetterdaten | ✅ Fertig |
| `server.py` | 4 MCP-Tools | ✅ Fertig (Protokoll-Test ausstehend) |

### Tests (`tests/`)
| Datei | Anzahl Tests | Status |
|---|---|---|
| `test_moist_air.py` | 22 | ✅ Bestanden |
| `test_ntu_dry.py` | 13 | ✅ Bestanden |
| `test_components.py` | 20 | ⚠️ Nach Humidifier-Fix erneut prüfen |

---

## MCP-Server Tools

| Tool | Beschreibung | Internet nötig? |
|---|---|---|
| `calculate_single_timestep` | Einzelzeitschritt mit fixen Randbedingungen | Nein |
| `calculate_ahu_energy` | Jahressimulation mit Open-Meteo Wetterdaten | Ja |
| `compare_ahu_concepts` | Mehrere Konzepte unter gleichen Bedingungen vergleichen | Ja |
| `get_design_weather_data` | Auslegungswetterdaten für einen Standort | Ja |

---

## Kritische Bugfixes

### Sprühbefeuchter Nebelgebiet
- **Problem**: Sättigungsprüfung nutzte x_sat bei Einlasstemperatur statt bei Auslasstemperatur → physikalisch unmögliche Zustände (Nebel)
- **Fix**: `_find_saturation_dx()` mit brentq findet Schnittpunkt der Prozesslinie h_out = h_in + dx·h_water mit der Sättigungskurve
- **Datei**: `src/components/humidifier.py`

### VHR-Vorheizlogik für Sprühbefeuchter
- **Problem**: VHR heizte nur auf T_soll → Sprühbefeuchter konnte x_soll nicht erreichen
- **Fix**: Binärsuche für minimale T_preheat, sodass Sprühbefeuchter nach adiabater Befeuchtung x_soll erreicht
- **Datei**: `src/system/ahu_system.py`

### Rotary HRV Parameternamen
- **Problem**: Falsche Parameternamen (`x_SUP_hr_req_min` statt 4 separate Parameter) und Dict-Zugriff statt Dataclass-Attribut
- **Fix**: Korrekte 4-Parameter-Form + `result.T_SUP_hr_out` statt `result['T_SUP_hr_out']`
- **Datei**: `src/components/heat_recovery.py`

---

## Offene Aufgaben

1. **Tests erneut laufen lassen** nach Humidifier/AHU-Änderungen
2. **CoolingCoil intensiver testen** — Entfeuchtung bei Sommerbedingungen
3. **KVS-Modell verbessern** — aktuell vereinfacht, sollte NTU-Iteration nutzen
4. **MCP-Server End-to-End testen** via MCP-Protokoll (nicht nur Python-Aufrufe)
5. **Adiabate Kühlung** (`adiabatic_cooling.py`) implementieren
6. **Umluft-Bypass** (`recirculation_bypass.py`) implementieren
7. **Rotoroptimierung Energy-Modus** validieren
8. **Sollwertbereich** (`setpoint_T_range`/`setpoint_x_range`) in AHUSystem aktivieren
9. **README.md** mit Installationsanleitung und Nutzungsbeispielen

---

## Validierte Testrechnungen

### Testfall: Winterbetrieb mit Sorptionsrotor + Sprühbefeuchter
**Konfiguration**: Frostschutz → Sorptionsrotor → VHR → Sprühbefeuchter → NHR → Ventilator
- Außenluft: T=-5°C, φ=80%, V=10000 m³/h
- Abluft: T=22°C, φ=40%
- Sollwert: T=24°C, φ=60%

**Ergebnis (VHR mit design_capacity_kW=60)**:
- Zuluft: T=24.0°C, x=10.42 g/kg, φ=55.8%
- Q_heat=83.3 kWh, W_el=3.0 kWh, m_water=69.9 kg

**Ergebnis (VHR ohne Kapazitätslimit)**:
- Zuluft: T=24.0°C, x=11.21 g/kg, φ=60% ✅
- Q_heat=90.4 kWh (VHR: 72.3 kW + NHR: 18.1 kW)
