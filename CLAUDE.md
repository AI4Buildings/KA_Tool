# CLAUDE.md – Projektbeschreibung: HVAC-Analysetool (MATLAB → Python + MCP-Server)

## Projektübersicht

Dieses Projekt konvertiert ein bestehendes MATLAB-basiertes Analysemodell für Lüftungs- und
Klimaanlagen (beschrieben in der Masterarbeit von Sebastian Dragosits, FH Burgenland, 2023)
in ein modernes Python-Tool. Das Python-Tool wird anschließend als **MCP-Server** (Model
Context Protocol) bereitgestellt, sodass ein LLM (z.B. Claude) direkt damit interagieren kann,
um verschiedene Lüftungskonzepte in der Planungsphase energetisch zu vergleichen.

**Wichtig:** Der ursprüngliche MATLAB-Code enthielt einen Vergleich mit realen Messdaten
(Performance-Analyse gegen Messwerte). Dieser Teil wird **nicht** implementiert. Das neue
Tool dient **ausschließlich der Vorausberechnung / Konzeptplanung** – es berechnet den
Energiebedarf für Heizung, Befeuchtung und Kühlung für einen definierten Zeitraum und Ort
auf Basis realer Wetterdaten und vorgegebener Raumluftbedingungen.

---

## Teil 1: Beschreibung des ursprünglichen MATLAB-Modells

### 1.1 Thermodynamische Grundlagen (Modul: moist_air)

Alle Luftzustandsberechnungen basieren auf der Thermodynamik feuchter Luft.

#### Zustandsgrößen

```
relative Feuchte:       φ = p_d / p_s
absolute Feuchte:       x = 0.622 * (φ * p_s) / (p – φ * p_s)
Sättigungsdampfdruck:   p_s = 611.2 * exp(17.62 * t / (243.12 + t))   [Pa, t in °C]
```

#### Spezifische Enthalpie feuchter Luft (Bezugspunkt 0°C)

```
Stoffwerte:
  c_pl  = 1.004 kJ/(kg·K)   # trockene Luft
  c_pd  = 1.860 kJ/(kg·K)   # Wasserdampf
  c_w   = 4.180 kJ/(kg·K)   # Wasser
  c_eis = 2.040 kJ/(kg·K)   # Eis
  r_0   = 2500.9 kJ/kg       # Verdampfungsenthalpie bei 0°C
  σ_0   = 333.1  kJ/kg       # Schmelzenthalpie Eis bei 0°C

Fall 1 – ungesättigt/gesättigt (x ≤ x_s):
  h = c_pl * t + x * (r_0 + c_pd * t)

Fall 2 – übersättigt, t ≥ 0°C (x > x_s):
  h = c_pl * t + x_s * (r_0 + c_pd * t) + (x – x_s) * c_w * t

Fall 3 – übersättigt, t < 0°C (x > x_s):
  h = c_pl * t + x_s * (r_0 + c_pd * t) + (x – x_s) * (–σ_0 + c_eis * t)
```

#### Wärmestrom bei Zustandsänderungen

```
Q_dot = m_dot_dry * (h_out – h_in)     [kW]
```

#### Kondensatanfall bei Kühlung unter Taupunkt

```
Δx = x_1 – x_s(t_2)
```

#### Adiabate Mischung zweier Luftströme

```
m_dot_3 = m_dot_1 + m_dot_2
x_3     = (x_1 * m_dot_1 + x_2 * m_dot_2) / m_dot_3
h_3     = (h_1 * m_dot_1 + h_2 * m_dot_2) / m_dot_3
```

#### Befeuchtung

```
Δx = Δm_dot_H2O / m_dot_dry
Δh = Δx * h_H2O

  Dampf:  h_H2O = r_0 + c_pd * t_H2O
  Wasser: h_H2O = c_w  * t_H2O
```

---

### 1.2 NTU-Verfahren für trockene Wärmeübertrager (Modul: heat_exchanger_dry)

Wird verwendet für: Vorheizregister, Nachheizregister, Plattenwärmeübertrager (WRG),
Kreislaufverbundsystem (KVS).

```python
# Kapazitätsströme
C_L = m_dot_L * c_p_L      # luftseitig  [W/K]
C_W = m_dot_W * c_p_W      # wasserseitig [W/K]

# Kapazitätsstromverhältnisse
C_L_star = C_L / C_W
C_W_star = C_W / C_L

# NTU
NTU_L = UA / C_L
NTU_W = UA / C_W

# Effektivität – Gegenstrom (C* ≠ 1):
ε = (1 – exp(–NTU * (1 – C_star))) / (1 – C_star * exp(–NTU * (1 – C_star)))

# Effektivität – Gegenstrom (C* = 1):
ε = NTU / (1 + NTU)

# Effektivität – Kreuzstrom (ein Durchgang, für Plattenwärmeübertrager):
ε = (1 – exp(–C_star * (1 – exp(–NTU)))) / C_star

# Austrittstemperaturen
ε_L = (T_L_out – T_L_in) / (T_W_in – T_L_in)
ε_W = (T_W_in – T_W_out) / (T_W_in – T_L_in)

# Leistung
Q_dot = ε_L * C_L * (T_L_in – T_W_in)
```

#### Teillastkorrektur der Wärmedurchgangsfähigkeit (Kaup 2015)

```python
UA = UA_ref * ((V_dot_L / V_dot_L_ref) * (V_dot_W / V_dot_W_ref)) ** n

# Standard-Exponent für trockene Wärmeübertrager: n = 0.4
# Gültigkeitsbereich: V_dot_L/V_dot_L_ref in [0.4, 1.6]
#                     V_dot_W/V_dot_W_ref in [0.8, 1.4]
# Genauigkeit innerhalb Grenzen: ±3%
```

**Berechnungsablauf Teillast (trockener WÜ):**
1. ε_ref aus Auslegungszustand (Gleichung oben)
2. C*_ref aus Auslegungsvolumenströmen
3. NTU_ref invertiert aus ε und C*
4. UA_ref = NTU_ref * C_ref
5. Neues UA aus Kaup-Formel
6. Neues C* aus aktuellen Volumenströmen
7. Neues NTU = UA_new / C_new
8. Neue ε aus NTU und C*
9. Neue Austrittstemperatur und Leistung

---

### 1.3 NTU-Verfahren für nasse Wärmeübertrager (Modul: heat_exchanger_wet)

Wird verwendet für: Kühlregister mit aktiver Entfeuchtung.

```python
# Effektive spezifische Wärmekapazität (konstant gehalten im Teillast)
c_s = (h_W_sat_in – h_W_sat_out) / (T_W_in – T_W_out)

# Modifiziertes Kapazitätsstromverhältnis
m_star = (m_dot_L * c_s) / (m_dot_W * c_p_W)

# Stofftransport-NTU
NTU_star = UA_star / m_dot_L

# Effektivität (auf Basis Enthalpien)
ε_star = (h_L_in – h_L_out) / (h_L_in – h_W_sat_in)

# Leistung
Q_dot = ε_star * m_dot_L * (h_L_in – h_W_sat_in)
       = m_dot_W * c_p_W * (T_W_out – T_W_in)
```

#### Teillastkorrektur nasser WÜ (Kaup 2015, angepasst)

```python
UA_star = UA_star_ref * ((V_dot_L / V_dot_L_ref) * (V_dot_W / V_dot_W_ref)) ** n

# Optimierter Exponent (validiert gegen Herstellerdaten Kelvion/iPCoil):
n = 0.2113
# Mittlere Abweichung über gesamten Leistungsbereich: 5%
```

**Berechnungsablauf Teillast (nasser WÜ):**
1. ε*_ref aus Auslegungszustand
2. m*_ref aus c_s (konstant) und Auslegungsvolumenströmen
3. NTU*_ref invertiert aus ε* und m*
4. UA*_ref = NTU*_ref * m_dot_L_ref
5. Neues UA* aus Kaup-Formel (n=0.2113)
6. Neues m* = UA*_new / (m_dot_W_new * c_p_W) ... (c_s konstant)
7. Neues NTU* = UA*_new / m_dot_L_new
8. Neue ε* aus NTU* und m*
9. Neue Auslassenenthalpie → Temperatur, Feuchte, Leistung

---

### 1.4 Rotationswärmeübertrager (Modul: rotary_heat_exchanger)

Basiert auf ÖNORM EN 16798-5-1:2017 + A1:2020, mit Modifikationen nach IDA ICE (EQUA).

**Unterstützte Typen:** Sorptionsrotor (ROT_SORP), Enthalpierotor (ROT_HYG),
Kondensationsrotor (ROT_NH).

#### Temperaturübertragungsgrad

```
η_hr = η_hr_nom * f_q * f_v * f_n

f_q = ((q_V_ETA - q_V_SUP) / (q_V_SUP * f_ODA_min) + 1) ^ 0.4

f_v = C1 * (q_V_SUP / q_V_SUP_nom * v_nom * f_ODA_min - v_nom) + C2

f_n = C3 – C4 * (n_rot / n_rot_max + C5) ^ e1
```

#### Feuchteübertragungsgrad

```
η_xr = η_xr_nom * f_Δx_x * f_q_x * f_v_x * f_n_x

# Kondensationspotential-Korrekturfaktor:
# Sorptions-/Nicht-hygroskopischer Rotor:
f_Δx_x = max(0; C6 * (x_calc_cond – x_e_sat) + C7)

# Hygroskopischer Rotor:
f_Δx_x = max(0; 1 + C6*(x_calc_cond – x_e_sat – Δx_e_nom);
              C7 + C8*(x_calc_cond – x_e_sat – Δx_e_nom))

# Sättigungsfeuchte Außenluft:
x_e_sat = 0.622 * p_e_sat / (p_e_sat – p_atm)
p_e_sat = 611.2 * exp(17.62 * T_calc_evap / (243.12 + T_calc_evap))

# Heizfall (T_e ≤ T_ETA_dis_out):
T_calc_evap  = T_ODA_preh
x_calc_cond  = x_ETA_hr_in
q_calc_evap  = q_V_SUP
q_calc_cond  = q_V_ETA

# Kühlfall (T_e > T_ETA_dis_out):
T_calc_evap  = T_ETA_hr_in
x_calc_cond  = x_ODA_preh
q_calc_evap  = q_V_ETA
q_calc_cond  = q_V_SUP

f_q_x = 1 – C9 * (q_calc_cond – q_calc_evap) / (q_calc_evap * f_ODA_min)

f_v_x = C10 * (q_V_SUP / q_V_SUP_nom * v_nom * f_ODA_min – v_nom) + C11

f_n_x = max(0; C12 – C13 * (n_rot / n_rot_max * n_rot_max_norm + C14) ^ e2)
```

#### Modellkonstanten je Rotortyp

| Konstante       | ROT_NH  | ROT_HYG | ROT_SORP |
|-----------------|---------|---------|----------|
| η_hr_nom        | 0.69    | 0.67    | 0.69     |
| v_hr_nom        | 3.5 m/s | 3.5 m/s | 3.5 m/s  |
| C1              | -0.0643 | -0.0684 | -0.0665  |
| C2              | 1       | 1       | 1        |
| n_rot_max_norm  | 20 1/min| 20 1/min| 20 1/min |
| η_xr_nom        | 0.30    | 0.42    | 0.69     |
| C3              | 1.0182  | 1.0182  | 1.0182   |
| C4              | 0.0352  | 0.0352  | 0.0352   |
| C5              | 0.276   | 0.276   | 0.276    |
| e1              | -2.7    | -2.7    | -2.7     |
| C6              | 248     | 129     | 16.4     |
| C7              | -0.240  | 0.476   | 0.918    |
| C8              | –       | 23.8    | –        |
| Δx_e_nom        | 0.005 kg/kg | 0.005 kg/kg | 0.005 kg/kg |
| C9              | 0.1     | 0.1     | 0.1      |
| C10             | -0.200  | -0.152  | -0.098   |
| C11             | 1       | 1       | 1        |
| C12             | 1.0533  | 1.0533  | 1.0533   |
| C13             | 80000   | 80000   | 80000    |
| C14             | 15      | 15      | 15       |
| e2              | -4      | -4      | -4       |

---

### 1.5 Komponentenlogik (Modul: components)

Jede Komponente berechnet ausgehend vom **Eintrittszustand der Luft** den **optimalen
Austrittszustand** und den zugehörigen **Energiebedarf**. Die Komponenten werden in der
festgelegten Reihenfolge sequenziell abgearbeitet; der Austrittszustand einer Komponente
ist der Eintrittszustand der nächsten.

#### Ventilator
```
Wärmeeintrag = P_el * η_rec     (η_rec: Rückgewinnungsgrad, Standard: 1.0 für Motor im Kanal)
Zustandsänderung: Erwärmung bei konstantem x
```

#### Frostschutz – elektrischer Vorerhitzer
```
Wenn T_ein < T_grenz:  Erwärme auf T_grenz
Richtwert T_grenz Plattenwärmeübertrager: 0°C
Richtwert T_grenz Rotationswärmeübertrager: -5°C
Richtwert T_grenz KVS (ohne Frostschutzmittel): +5°C
```

#### Frostschutz – Bypass
```
Wenn T_ein < T_grenz:  Bypass aktiv → WRG wird überbrückt
Vorheizregister übernimmt dann die gesamte Aufheizung
```

#### Wärmerückgewinnung – Plattenwärmeübertrager
```
Teillastmodell: NTU-Kreuzstrom + Kaup (n=0.4)
Optimierung: Berechne Austrittszustand bei maximalem Wärmeübergang
Begrenzung auf Sollwert wenn nötig
```

#### Wärmerückgewinnung – Kreislaufverbundsystem (KVS)
```
Teillastmodell: NTU-Gegenstrom (für beide WÜ) + Kaup (n=0.4)
Optimierung: Iteration über Volumenstrom des Zwischenmediums (Schrittweite klein)
  → bestimme optimalen Volumenstrom für minimalen Gesamtenergiebedarf
  → berücksichtige Sollwert, nachfolgende Komponenten (Befeuchter, KR, Ventilator)
Vereinfachung: Wärmeverlust zwischen den beiden WÜ wird vernachlässigt
```

#### Wärmerückgewinnung – Rotationswärmeübertrager
```
Teillastmodell: ÖNORM EN 16798-5-1 (modifiziert, siehe 1.4)
Optimierungsmodi (auswählbar):
  1. Temperaturoptimierung: optimale Rotordrehzahl für Temperatursollwert
  2. Feuchteoptimierung:    optimale Rotordrehzahl für Feuchtesollwert
  3. Energieoptimierung:    minimaler Gesamtenergiebedarf der Anlage
     (kann Sollwertbereich voll ausnutzen oder ggf. überschreiten wenn energetisch vorteilhaft)
Iteration: n_rot von 0 bis n_rot_max in kleinen Schritten
```

#### Vorheizregister
```
Wenn x < x_soll UND Befeuchter vorhanden:
  Dampfbefeuchter: Erwärme so dass nach Befeuchter (T_soll, x_soll) erreicht wird
  Sprühbefeuchter: Erwärme so dass nach Befeuchter x_soll erreicht wird
Wenn x < x_soll UND kein Befeuchter:  Erwärme auf T_soll
Wenn x ≥ x_soll:  Erwärme auf T_soll (bzw. nächste Sollwertgrenze)
Korrektur um Ventilator-Wärmeeintrag wenn Ventilator nachgeschaltet
Teillastmodell: NTU-Gegenstrom + Kaup (n=0.4)
```

#### Kühlregister
```
Wenn x > x_soll UND Entfeuchtung aktiv:
  Kühle + entfeuchte bis x_soll (Entfeuchtungsgrenze: VL-Temperatur)
  Wenn Entfeuchtungsziel thermisch nicht erreichbar: Kühle nur auf T_soll
Wenn x ≤ x_soll:  Kühle auf T_soll
Wenn Sprühbefeuchter nachgeschaltet: Nutze adiabaten Kühleffekt
Wenn Dampfbefeuchter nachgeschaltet: Kühle so dass nach Befeuchter T_soll nicht überschritten
Korrektur um Ventilator-Wärmeeintrag wenn Ventilator nachgeschaltet
Teillastmodell: NTU-Gegenstrom nass + Kaup (n=0.2113)
```

#### Nachheizregister
```
Wenn T_ein < T_soll:  Erwärme auf T_soll
Korrektur um Ventilator-Wärmeeintrag wenn Ventilator nachgeschaltet
Teillastmodell: NTU-Gegenstrom + Kaup (n=0.4)
```

#### Sprühbefeuchter
```
Wenn x < x_soll:  Befeuchte bis x_soll (max. bis Sättigungslinie)
Prüfe ob adiabater Kühleffekt genutzt werden kann (nur bei Sollwertbereich)
Wirkungsgrad: η_bef = 0.9 (Standard nach ÖNORM, konfigurierbar)
Modell: konstanter Befeuchtungswirkungsgrad
```

#### Dampfbefeuchter
```
Wenn x < x_soll:  Befeuchte bis x_soll
Zustandsänderung: Δx = x_soll – x_ein
Enthalpie: h_dampf = r_0 + c_pd * t_dampf
```

#### Adiabate Kühlung (Abluft-Befeuchter)
```
Freigabe nur bei: Plattenwärmeübertrager, KVS, Kondensations- oder Enthalpierotor
  (nicht bei Sorptionsrotor, da Feuchterückgewinnung den Kühleffekt zunichte macht)
Optimierung: Bestimme optimale Befeuchtungsmenge so dass ZUL-Temperatursollwert
  nach WRG nicht unterschritten wird
Wenn x_AUL > x_soll: Befeuchte Abluft maximal (beste Kühlung)
```

#### Umluft-Bypass
```
Mischzustand aus Umluft und Außenluft (adiabate Mischung)
Keine eigene Optimierung – bildet Anlagenkonfiguration ab
```

---

## Teil 2: Aufgabenstellung – Python-Konvertierung

### 2.1 Ziel

Entwickle ein **reines Python-Berechnungstool** (kein Messdatenvergleich, keine Validierung)
für die **Konzeptplanung von Lüftungsanlagen**. Das Tool soll:

1. Automatisch **stündliche Wetterdaten** (Temperatur, relative Feuchte) für einen
   beliebigen Ort und Zeitraum von der **Open-Meteo API** herunterladen
2. Für jeden Zeitschritt die **thermodynamischen Zustandsänderungen** aller
   Anlagenkomponenten in der definierten Reihenfolge berechnen
3. Den **Energiebedarf** (Heizung, Kühlung, Befeuchtung) über den gesamten Zeitraum
   aufsummieren
4. Als **MCP-Server** bereitgestellt werden, damit ein LLM die Berechnungen direkt
   aufrufen kann

### 2.2 Was wird NICHT implementiert (im Vergleich zum MATLAB-Original)

- ❌ Vergleich mit gemessenen Messdaten
- ❌ Datenaufbereitung / Ausreißerbereinigung realer Messdaten
- ❌ Datenqualitätsbewertung
- ❌ Resampling von Messzeitreihen
- ❌ Temperaturniveau-Analyse (Parameterstudie für Umbaumaßnahmen)
- ❌ Erweiterte Analysen (Teillastvergleich gemessen vs. berechnet)
- ❌ MATLAB-spezifische Ergebnisausgabe (Command Window, Figures)

### 2.3 Projektstruktur

```
hvac_planning_tool/
│
├── CLAUDE.md                          # diese Datei
│
├── src/
│   ├── __init__.py
│   │
│   ├── thermodynamics/
│   │   ├── __init__.py
│   │   ├── moist_air.py               # Zustandsgrößen feuchter Luft
│   │   └── air_state.py               # Datenklasse AirState (T, x, h, φ, p)
│   │
│   ├── heat_exchangers/
│   │   ├── __init__.py
│   │   ├── ntu_dry.py                 # NTU-Verfahren trockener WÜ + Kaup (n=0.4)
│   │   ├── ntu_wet.py                 # NTU-Verfahren nasser WÜ + Kaup (n=0.2113)
│   │   └── rotary.py                  # Rotationswärmeübertrager (ÖNORM-Modell)
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstrakte Basisklasse Component
│   │   ├── fan.py                     # Ventilator
│   │   ├── frost_protection.py        # Frostschutz (Vorerhitzer oder Bypass)
│   │   ├── heat_recovery.py           # WRG (Platten, Rotation, KVS)
│   │   ├── heating_coil.py            # Heizregister (VHR, NHR)
│   │   ├── cooling_coil.py            # Kühlregister
│   │   ├── humidifier.py              # Befeuchter (Sprüh, Dampf)
│   │   ├── adiabatic_cooling.py       # Adiabate Kühlung
│   │   └── recirculation_bypass.py    # Umluft-Bypass
│   │
│   ├── system/
│   │   ├── __init__.py
│   │   ├── ahu_system.py              # AHU (Air Handling Unit) – Hauptsystem
│   │   ├── setpoint_logic.py          # Sollwertlogik und Rückrechnung
│   │   └── simulation.py              # Zeitschleife über Wetterdaten
│   │
│   ├── weather/
│   │   ├── __init__.py
│   │   └── open_meteo.py              # Open-Meteo API Client
│   │
│   └── mcp_server/
│       ├── __init__.py
│       └── server.py                  # MCP-Server Definition
│
├── tests/
│   ├── test_moist_air.py
│   ├── test_ntu_dry.py
│   ├── test_ntu_wet.py
│   ├── test_rotary.py
│   └── test_components.py
│
├── examples/
│   ├── example_plate_hrv.py           # Beispiel: Anlage mit Plattenwärmeübertrager
│   ├── example_rotary_hrv.py          # Beispiel: Anlage mit Sorptionsrotor
│   └── example_comparison.py          # Vergleich zweier Konzepte
│
├── pyproject.toml
└── README.md
```

### 2.4 Kernklassen und Interfaces

#### AirState (Datenklasse)

```python
@dataclass
class AirState:
    T: float        # Temperatur [°C]
    x: float        # absolute Feuchte [kg_H2O/kg_dry_air]
    p: float        # Luftdruck [Pa], default 101325
    # Abgeleitete Größen werden als Properties berechnet:
    # phi, h, T_dew, x_sat
```

#### Abstrakte Basisklasse Component

```python
class Component(ABC):
    @abstractmethod
    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  **kwargs) -> ComponentResult:
        """
        Berechnet optimalen Austrittszustand und Energiebedarf.
        Returns ComponentResult mit:
          - air_out: AirState
          - Q_heat: float  [kWh] Wärmebedarf (positiv = Heizung)
          - Q_cool: float  [kWh] Kältebedarf (positiv = Kühlung)
          - W_el:   float  [kWh] elektrischer Bedarf
          - m_water: float [kg]  Wasserverbrauch (Befeuchtung)
        """
```

#### AHUSystem

```python
class AHUSystem:
    def __init__(self,
                 supply_air_components: list[Component],
                 exhaust_air_components: list[Component],
                 design_airflow: float,          # [m³/h]
                 design_supply_T: float,          # [°C]
                 design_supply_x: float):         # [kg/kg]
    
    def calculate_timestep(self,
                           outdoor_air: AirState,
                           exhaust_air: AirState,
                           setpoint_T: float,
                           setpoint_x: float,
                           airflow: float) -> SystemResult:
        """Berechnet einen Zeitschritt sequenziell durch alle Komponenten."""
```

#### Open-Meteo Weather Client

```python
class OpenMeteoClient:
    def get_hourly_weather(self,
                           latitude: float,
                           longitude: float,
                           start_date: str,       # "YYYY-MM-DD"
                           end_date: str,
                           ) -> pd.DataFrame:
        """
        Ruft stündliche Wetterdaten ab:
          - temperature_2m [°C]
          - relative_humidity_2m [%]
        API: https://api.open-meteo.com/v1/forecast
             oder /v1/archive für historische Daten
        Returns DataFrame mit DatetimeIndex und Spalten T, phi
        """
```

### 2.5 MCP-Server Tools

Der MCP-Server stellt folgende Tools bereit:

#### `calculate_ahu_energy`
```
Beschreibung: Berechnet den Energiebedarf einer Lüftungsanlage für einen
              definierten Zeitraum und Ort.

Parameter:
  location:        str    – Ortsname oder "lat,lon" (z.B. "Wien" oder "48.2,16.3")
  start_date:      str    – Startdatum "YYYY-MM-DD"
  end_date:        str    – Enddatum   "YYYY-MM-DD"
  airflow_supply:  float  – Zuluftvolumenstrom [m³/h]
  airflow_exhaust: float  – Abluftvolumenstrom [m³/h]
  
  room_temp_setpoint:     float  – Raumtemperatur-Sollwert [°C]
  room_humidity_setpoint: float  – Raumluftfeuchte-Sollwert [% r.F.]
  
  supply_temp_setpoint:   float  – Zuluftsollwert Temperatur [°C] (optional,
                                    wenn nicht angegeben → Rückrechnung aus Raumsollwert)
  supply_humidity_setpoint: float – Zuluftsollwert Feuchte [% r.F.] (optional)
  
  system_config:   dict   – Anlagenkonfiguration (siehe unten)
  design_data:     dict   – Auslegungsdaten der Komponenten

Rückgabe:
  energy_summary: dict mit
    Q_heat_total_kWh:    float  – gesamter Heizbedarf
    Q_cool_total_kWh:    float  – gesamter Kältebedarf
    W_humidifier_kWh:    float  – Energiebedarf Befeuchtung
    m_water_total_kg:    float  – Wasserverbrauch Befeuchtung
    W_fan_total_kWh:     float  – Energiebedarf Ventilatoren
    operating_hours:     int    – Betriebsstunden gesamt
    monthly_breakdown:   list   – monatliche Aufschlüsselung
    peak_loads:          dict   – Spitzenlasten je Komponente
```

#### `compare_ahu_concepts`
```
Beschreibung: Vergleicht mehrere Lüftungskonzepte unter gleichen Randbedingungen.

Parameter:
  concepts: list[dict]  – Liste von Anlagenkonfigurationen (je eine system_config + design_data)
  concept_names: list[str] – Namen der Konzepte
  location: str
  start_date: str
  end_date: str
  room_conditions: dict  – Raumluftsollwerte (einmalig für alle Konzepte)

Rückgabe:
  comparison_table: list[dict]  – Energievergleich aller Konzepte
  recommendation:   str          – Kurzauswertung
```

#### `get_design_weather_data`
```
Beschreibung: Gibt typische Auslegungswetterdaten für einen Ort zurück
              (Sommerspitze, Winterspitze, Jahresmittel).

Parameter:
  location: str
  year: int (optional, default: aktuelles Jahr - 1 für komplettes Jahr)

Rückgabe:
  summer_design: dict  – T, phi für Auslegung Sommer
  winter_design: dict  – T, phi für Auslegung Winter
  annual_stats:  dict  – Mittelwerte, Stunden < 0°C etc.
```

### 2.6 Anlagenkonfiguration (system_config Format)

```python
system_config = {
    "supply_air": ["fan", "frost_protection", "hrv", "pre_heating_coil",
                   "cooling_coil", "humidifier", "post_heating_coil"],
    "exhaust_air": ["adiabatic_cooling", "hrv"],
    
    "components": {
        "fan": {
            "type": "fan",
            "position": "supply",
            "heat_recovery_factor": 1.0   # Motor im Kanal
        },
        "frost_protection": {
            "type": "frost_protection",
            "variant": "electric_preheater",   # oder "bypass"
            "limit_temperature": -5.0          # [°C]
        },
        "hrv": {
            "type": "hrv",
            "variant": "rotary",               # "plate", "rotary", "runaround"
            "rotor_type": "sorption",          # nur bei rotary: "sorption", "enthalpy", "condensation"
            "optimization_mode": "temperature", # "temperature", "humidity", "energy"
            "design_airflow_supply":  5000,    # [m³/h]
            "design_airflow_exhaust": 5000,    # [m³/h]
            "n_rot_max": 20,                   # [1/min]
            "f_ODA_min": 1.0
        },
        "pre_heating_coil": {
            "type": "heating_coil",
            "design_capacity_kW": 30,
            "design_supply_T_water": 60,       # [°C] Vorlauftemperatur
            "design_return_T_water": 45,       # [°C] Rücklauftemperatur
            "design_airflow": 5000,            # [m³/h]
            "design_air_T_in": -12,            # [°C]
            "design_air_T_out": 18             # [°C]
        },
        "cooling_coil": {
            "type": "cooling_coil",
            "dehumidification": True,
            "design_capacity_kW": 25,
            "design_supply_T_water": 6,
            "design_return_T_water": 12,
            "design_airflow": 5000,
            "design_air_T_in": 28,
            "design_air_phi_in": 60,           # [% r.F.]
            "design_air_T_out": 13,
            "design_air_phi_out": 95
        },
        "humidifier": {
            "type": "humidifier",
            "variant": "steam",                # "steam" oder "spray"
            "efficiency": 0.9                  # nur bei spray
        },
        "post_heating_coil": {
            "type": "heating_coil",
            "design_capacity_kW": 10,
            # ... analog pre_heating_coil
        },
        "adiabatic_cooling": {
            "type": "adiabatic_cooling",
            "efficiency": 0.9
        }
    }
}
```

### 2.7 Implementierungshinweise

#### Abhängigkeiten
```toml
[dependencies]
numpy = ">=1.24"
pandas = ">=2.0"
scipy = ">=1.10"          # für Optimierung (brentq etc.)
requests = ">=2.28"       # Open-Meteo API
mcp = ">=1.0"             # MCP SDK (pip install mcp)
pydantic = ">=2.0"        # Datenvalidierung
```

#### Numerische Besonderheiten
- **NTU-Inversion:** Zum Bestimmen von NTU aus bekanntem ε und C* → `scipy.optimize.brentq`
- **Rotoroptimierung:** Iteration von n_rot = 0..n_rot_max in Schritten von 0.1 1/min
- **KVS-Optimierung:** Iteration Volumenstrom Medium von 0..V_dot_max in kleinen Schritten
- **Sollwertbereich:** Beim Fehlen eines fixen Sollwerts → verwende ±Toleranz (z.B. ±1K, ±5% r.F.)
- **Einheitenkonsistenz:** Intern immer SI-Einheiten (Pa, K oder °C konsistent, kg/s, kW)
  Ausnahme: Volumenströme in m³/h (Eingabe), intern Umrechnung in m³/s

#### Open-Meteo API
```
Historische Daten:  https://archive-api.open-meteo.com/v1/archive
Aktuelle Daten:     https://api.open-meteo.com/v1/forecast
Parameter:          temperature_2m, relative_humidity_2m
Zeitauflösung:      hourly
```

#### Geocoding (Ortsname → Koordinaten)
```
API: https://geocoding-api.open-meteo.com/v1/search?name=Wien&count=1
```

### 2.8 Teststrategie

Für jedes Rechenmodul sind Unit-Tests zu erstellen, die folgende Kontrollwerte prüfen:

- `test_moist_air.py`: Bekannte Luftzustände aus h-x-Diagramm nachrechnen
- `test_ntu_dry.py`: Auslegungspunkt muss ε_ref reproduzieren; Teillast innerhalb ±3%
- `test_ntu_wet.py`: Mittlere Abweichung über Leistungsbereich ≤ 5% (analog Masterarbeit)
- `test_rotary.py`: Validierungspunkte aus Tabelle 4.1 der Masterarbeit nachrechnen
- `test_components.py`: Energiebilanz-Konsistenz (Q_ab = Q_zu + interne Quellen)

### 2.9 Konvertierungsreihenfolge (empfohlen für Claude Code)

1. `src/thermodynamics/moist_air.py` – Grundlage für alle weiteren Module
2. `src/thermodynamics/air_state.py` – Datenklasse mit Properties
3. `src/heat_exchangers/ntu_dry.py`  – NTU-Verfahren trocken + Kaup
4. `src/heat_exchangers/ntu_wet.py`  – NTU-Verfahren nass + Kaup (n=0.2113)
5. `src/heat_exchangers/rotary.py`   – Rotationswärmeübertrager
6. `src/components/*.py`             – Alle Komponenten (je eine Datei)
7. `src/system/ahu_system.py`        – Systemintegration, Zeitschleife
8. `src/weather/open_meteo.py`       – Wetterdaten-Client
9. `src/mcp_server/server.py`        – MCP-Server
10. `tests/`                         – Unit-Tests parallel zur Implementierung

---

## Teil 3: MCP-Server – Nutzungsbeispiel

Sobald der MCP-Server läuft, soll ein LLM folgende Anfragen verarbeiten können:

**Beispiel-Prompt an das LLM:**
> „Vergleiche eine Lüftungsanlage mit Sorptionsrotor gegen eine mit Plattenwärmeübertrager
> für Wien, Zeitraum Januar–Dezember 2024, Bürogebäude mit 5000 m³/h Zuluft,
> Raumtemperatur 21°C, Raumfeuchte 45% r.F. Welches Konzept verbraucht weniger Energie?"

**LLM ruft auf:**
1. `get_design_weather_data(location="Wien", year=2024)` → Überblick
2. `compare_ahu_concepts(concepts=[sorptionsrotor_config, platten_config], location="Wien", ...)`
3. Interpretiert Ergebnisse und gibt Empfehlung

---

---

## Teil 4: MATLAB-Quellcode – Dateistruktur und Leseanleitung für Claude Code

### 4.1 Dateistruktur im Arbeitsverzeichnis

Claude Code arbeitet in dem Ordner, in dem die Hauptdatei liegt. Die Struktur ist:

```
<arbeitsverzeichnis>/
│
├── VKA_Effizienzbeurteilung_V11_TT_AIagents4b.m   ← Hauptskript (Einstiegspunkt)
├── AHU_ENERGETIKUM_T...l_Vdotzul_Personen.mat      ← Messdaten (werden NICHT benötigt)
├── AHU_ENERGETIKUM_WEATHER.mat                      ← Wettermessdaten (werden NICHT benötigt)
│
└── Funktionen/                                      ← Alle Hilfsfunktionen
    ├── dT_Kanal.m              # Temperaturerhöhung durch Ventilator im Kanal
    ├── h.m                     # Spezifische Enthalpie feuchter Luft → moist_air.py
    ├── heat_rec_wheel_calc.m   # Rotationswärmeübertrager Hauptfunktion
    ├── heat_rec_wheel_calc_V2.m
    ├── heat_rec_wheel_calc_V3.m
    ├── heat_rec_wheel_calc_V4.m
    ├── heat_rec_wheel_calc_V5.m  ← wahrscheinlich aktuellste Version → rotary.py
    ├── m_dot_water_VDI.m       # Wassermassenstrom (VDI-Methode)
    ├── phi.m                   # Relative Feuchte → moist_air.py
    ├── ps.m                    # Sättigungsdampfdruck → moist_air.py
    ├── T_h_phi.m               # Temperatur aus Enthalpie und rel. Feuchte → moist_air.py
    ├── T.m                     # Temperatur aus Enthalpie und abs. Feuchte → moist_air.py
    ├── Ts.m                    # Taupunkttemperatur → moist_air.py
    ├── Ts2.m                   # Taupunkttemperatur (alternative Methode) → moist_air.py
    ├── x.m                     # Absolute Feuchte → moist_air.py
    └── xs.m                    # Sättigungsfeuchte → moist_air.py
```

### 4.2 Anleitung für Claude Code: Wie die MATLAB-Files zu lesen sind

**SCHRITT 1 – Zuerst alle Funktionen lesen, dann das Hauptskript:**

```bash
# Alle Hilfsfunktionen einlesen (Reihenfolge empfohlen):
cat Funktionen/ps.m          # Sättigungsdampfdruck – Basis für alles
cat Funktionen/xs.m          # Sättigungsfeuchte
cat Funktionen/x.m           # Absolute Feuchte
cat Funktionen/phi.m         # Relative Feuchte
cat Funktionen/h.m           # Enthalpie
cat Funktionen/T.m           # Temperatur aus h und x
cat Funktionen/T_h_phi.m     # Temperatur aus h und phi
cat Funktionen/Ts.m          # Taupunkt
cat Funktionen/Ts2.m         # Taupunkt (alternativ)
cat Funktionen/dT_Kanal.m    # Ventilator-Wärmeeintrag
cat Funktionen/m_dot_water_VDI.m  # Wassermassenstrom

# Rotationswärmeübertrager – alle Versionen vergleichen, aktuellste verwenden:
cat Funktionen/heat_rec_wheel_calc.m
cat Funktionen/heat_rec_wheel_calc_V2.m
cat Funktionen/heat_rec_wheel_calc_V3.m
cat Funktionen/heat_rec_wheel_calc_V4.m
cat Funktionen/heat_rec_wheel_calc_V5.m   # ← vermutlich final

# Hauptskript (groß, daher abschnittsweise lesen):
cat VKA_Effizienzbeurteilung_V11_TT_AIagents4b.m
```

**SCHRITT 2 – Mapping MATLAB-Funktionen → Python-Module:**

| MATLAB-Datei | Python-Zieldatei | Funktion |
|---|---|---|
| `ps.m` | `src/thermodynamics/moist_air.py` | `saturation_pressure(T)` |
| `xs.m` | `src/thermodynamics/moist_air.py` | `saturation_humidity(T, p)` |
| `x.m` | `src/thermodynamics/moist_air.py` | `absolute_humidity(phi, T, p)` |
| `phi.m` | `src/thermodynamics/moist_air.py` | `relative_humidity(x, T, p)` |
| `h.m` | `src/thermodynamics/moist_air.py` | `enthalpy(T, x)` |
| `T.m` | `src/thermodynamics/moist_air.py` | `temperature_from_h_x(h, x)` |
| `T_h_phi.m` | `src/thermodynamics/moist_air.py` | `temperature_from_h_phi(h, phi, p)` |
| `Ts.m` / `Ts2.m` | `src/thermodynamics/moist_air.py` | `dew_point(x, p)` |
| `dT_Kanal.m` | `src/components/fan.py` | `temperature_rise_fan(P_el, m_dot, eta_rec)` |
| `m_dot_water_VDI.m` | `src/components/humidifier.py` | Wassermassenstrom-Berechnung |
| `heat_rec_wheel_calc_V5.m` | `src/heat_exchangers/rotary.py` | `RotaryHeatExchanger.calculate()` |
| Hauptskript: NTU-Abschnitte | `src/heat_exchangers/ntu_dry.py` | NTU-Berechnung trocken |
| Hauptskript: NTU-nass Abschnitte | `src/heat_exchangers/ntu_wet.py` | NTU-Berechnung nass |
| Hauptskript: Komponentenlogik | `src/components/*.py` | je Komponente |
| Hauptskript: Hauptschleife | `src/system/simulation.py` | Zeitschleife |

**SCHRITT 3 – Beim Lesen des Hauptskripts auf folgende Sektionen achten:**

Das MATLAB-Hauptskript ist wahrscheinlich in folgende logische Abschnitte gegliedert.
Claude Code soll diese identifizieren und entsprechend zuordnen:

```
%% EINGABE / KONFIGURATION
→ entspricht system_config in Python

%% DATENAUFBEREITUNG / MESSDATEN LADEN
→ wird NICHT übernommen (nur für Messdatenvergleich)

%% HAUPTSCHLEIFE (for/while über Zeitschritte)
→ entspricht src/system/simulation.py

%% KOMPONENTENBERECHNUNGEN (je Komponente ein Abschnitt)
→ entspricht src/components/*.py

%% ERGEBNISAUSWERTUNG / PLOTS
→ wird NICHT übernommen (in Python: nur strukturierte Rückgabe als dict)
```

**SCHRITT 4 – Besonderheiten beim Lesen der heat_rec_wheel Versionen:**

Die fünf Versionen (V1–V5) sind Weiterentwicklungen. Claude Code soll:
1. Alle fünf einlesen
2. Die **aktuellste/vollständigste Version** identifizieren (wahrscheinlich V5)
3. Schauen ob V2–V4 zusätzliche Korrekturen enthalten die in V5 fehlen könnten
4. Die finale Python-Implementierung auf der besten Version basieren

### 4.3 Was aus dem MATLAB-Code NICHT übernommen wird

Claude Code soll folgende Codeblöcke im Hauptskript **explizit überspringen**:

```matlab
% NICHT ÜBERNEHMEN:
load('AHU_ENERGETIKUM_*.mat')          % Messdatenladen
% ... alle Abschnitte die mit Messdaten arbeiten
% ... Vergleich berechnet vs. gemessen
% ... Ausreißerbereinigung / Datenqualität
% ... MATLAB figure() / plot() Aufrufe
% ... disp() Ausgaben die Messdatenvergleiche zeigen
```


