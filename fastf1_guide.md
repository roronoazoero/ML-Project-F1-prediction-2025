# FastF1 Guide for Machine Learning Project

## Project Goal

Predict race-by-race finishing positions in Formula 1 using a
**Multi-layer Perceptron (MLP)**.

- **Why MLP**: Can capture complex non-linear relationships between
  track characteristics, driver form, and performance.

## Data Overview

- **Seasons**: 2019--2024
- **Races per season**: \~22
- **Drivers per race**: \~20
- **Total data points**: \~2,640

## Target & Loss

- **Target variable**: Finishing position (1--20, with DNFs as 21)
- **Loss function**: Mean Squared Error (MSE)

## Validation Strategy

- Train: 2019--2022
- Validation: 2023
- Test: 2024

---

## Features

1.  **Track Type** (street / permanent / semi-permanent)
    - ✅ Done
    - Manual mapping dictionary required
    - Examples:
      - Street: Monaco, Singapore, Baku, Miami, Las Vegas
      - Semi-permanent: Albert Park, Montreal
      - Permanent: Silverstone, Spa, Monza, etc.
2.  **Weather Conditions** (dry / wet / mixed)
    - ✅ Done
    - From `session.weather_data` in FastF1
3.  **Driver's Historical Performance at Track**
    - Use average finishing position at a specific circuit (past 3--5
      years)
    - `mean(finishing_positions_at_circuit)` per driver
4.  **Qualifying Position** (1--20)
    - From `session.results['Position']` in qualifying
5.  **Car Pace (Practice Sessions)**
    - Use FP3
    - Formula: `(driver_best_lap - fastest_lap) / fastest_lap * 100`
6.  **Driver Championship Position Entering Race**
    - Running total of driver points before each race
7.  **Team Constructor Points Entering Race**
    - Running total of constructor points before each race

---

## Feature Engineering

- Create "driver form" metrics:
  - Example: Average finishing position in the last 5 races

---

## FastF1 Data Collection Guide

### General Setup

```python
import fastf1
from fastf1 import plotting
import pandas as pd

fastf1.Cache.enable_cache('cache')  # Important for efficiency
```

### 1. Loading Sessions

```python
session = fastf1.get_session(2023, 'Monaco', 'R')  # Year, GP name, session ('R' for race, 'Q' for qualifying, 'FP1/2/3')
session.load()
```

### 2. Qualifying Position

```python
quali = fastf1.get_session(2023, 'Monaco', 'Q')
quali.load()
positions = quali.results[['Abbreviation', 'Position', 'Q1', 'Q2', 'Q3']]
```

### 3. Track Type

- Manual dictionary:

```python
track_types = {
    'Monaco': 'Street',
    'Singapore': 'Street',
    'Baku': 'Street',
    'Miami': 'Street',
    'Las Vegas': 'Street',
    'Albert Park': 'Semi-permanent',
    'Montreal': 'Semi-permanent',
    # default: Permanent
}
```

### 4. Weather Data

```python
weather = session.weather_data
# Columns: Time, AirTemp, Humidity, Pressure, Rainfall, TrackTemp, WindDirection, WindSpeed
wet_race = any(weather['Rainfall'] > 0)
```

### 5. Driver Historical Performance at Track

```python
def driver_avg_position(driver_code, circuit_name, years):
    positions = []
    for year in years:
        ses = fastf1.get_session(year, circuit_name, 'R')
        ses.load()
        res = ses.results
        pos = res.loc[res['Abbreviation'] == driver_code, 'Position'].values
        if len(pos) > 0:
            positions.append(int(pos[0]))
    return sum(positions) / len(positions) if positions else None
```

### 6. Car Pace from Practice

```python
fp3 = fastf1.get_session(2023, 'Monaco', 'FP3')
fp3.load()
laps = fp3.laps.pick_quicklaps()
best_laps = laps.pick_fastest()
fastest = best_laps['LapTime'].min()

car_pace = {}
for drv in best_laps['Driver'].unique():
    lap_time = best_laps[best_laps['Driver'] == drv]['LapTime'].values[0]
    car_pace[drv] = (lap_time - fastest) / fastest * 100
```

### 7. Driver & Constructor Standings

FastF1 does not directly provide standings, but you can calculate them
using cumulative points from session results.

```python
def get_driver_points(season, upto_round):
    points = {}
    for rnd in range(1, upto_round + 1):
        ses = fastf1.get_session(season, rnd, 'R')
        ses.load()
        results = ses.results[['Abbreviation', 'Points']]
        for _, row in results.iterrows():
            points[row['Abbreviation']] = points.get(row['Abbreviation'], 0) + row['Points']
    return points
```

---

## Useful FastF1 Functions

- **`fastf1.get_session(year, gp, session_type)`** → Load any session
- **`session.load()`** → Must call before accessing data
- **`session.laps`** → DataFrame of all laps (driver, time, tyre,
  sector times, etc.)
- **`session.results`** → Session results DataFrame
- **`session.weather_data`** → Weather information per session
- **`laps.pick_fastest()`** → Get each driver's fastest lap
- **`laps.pick_quicklaps()`** → Filter realistic laps (exclude in/out
  laps)
- **`plotting.setup_mpl()`** → Configure matplotlib for F1 plots

---

## Notes

- Consider encoding categorical features (track type, weather)
  properly
- Use normalization/scaling for continuous features (e.g., car pace)
- Handle DNFs consistently as position 21
