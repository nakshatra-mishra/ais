# AIS Data Collection and Synthetic AIS Generation Guide

## 1. Objective

This document defines a practical workflow for building an AIS dataset for:

- vessel trajectory forecasting,
- multi-vessel interaction modelling,
- closest-approach prediction,
- collision-risk research,
- real-time AIS model prototyping,
- comparison of real-only versus real-plus-synthetic training.

The recommended data strategy is:

```text
REAL AIS
   │
   ├── learn real vessel behaviour
   ├── learn real message timing and missingness
   ├── calibrate synthetic generator
   │
   └──────────────┐
                  │
                  ▼
          SYNTHETIC AIS GENERATOR
                  │
                  ├── head-on encounters
                  ├── crossing encounters
                  ├── overtaking
                  ├── close passes
                  ├── turns / manoeuvres
                  ├── missing packets
                  └── controlled future ground truth
                  │
                  ▼
        REAL + SYNTHETIC TRAINING DATA
                  │
                  ▼
            TRAJECTORY MODEL
                  │
                  ▼
          REAL-ONLY HELD-OUT TEST SET
```

The synthetic dataset should be treated as **augmentation**, not as a replacement for real AIS.

---

# 2. Recommended repository structure

Do not commit large raw AIS files to Git.

A repository can contain scripts, configuration files and documentation while raw datasets remain under `data/`.

```text
ais-project/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── simulator.yaml
│   └── regions.yaml
│
├── scripts/
│   ├── download_noaa.py
│   ├── download_denmark.py
│   ├── capture_fintraffic.py
│   ├── normalize_ais.py
│   ├── inspect_real_ais.py
│   ├── synthetic_ais_simulator.py
│   └── make_training_windows.py
│
├── notebooks/
│   ├── 01_real_ais_inspection.ipynb
│   ├── 02_synthetic_validation.ipynb
│   └── 03_training_dataset.ipynb
│
└── data/
    ├── raw/
    │   ├── noaa/
    │   ├── denmark/
    │   └── fintraffic/
    │
    ├── normalized/
    ├── synthetic/
    ├── windows/
    └── test_real_only/
```

Recommended `.gitignore`:

```gitignore
data/
*.zip
*.zst
*.parquet
*.csv
*.csv.gz
__pycache__/
.ipynb_checkpoints/
```

---

# 3. Python environment

A minimal environment:

```bash
python -m venv .venv
```

Activate it, then install:

```bash
pip install numpy pandas pyarrow requests scipy scikit-learn matplotlib tqdm
```

Optional packages:

```bash
pip install polars duckdb pyais
```

Why these are useful:

- `pandas` / `polars`: tabular AIS processing.
- `pyarrow`: Parquet files.
- `duckdb`: query very large CSV/Parquet without loading everything into RAM.
- `pyais`: decode raw NMEA AIS if a source provides raw AIS messages.
- `numpy`: simulation and vector maths.
- `scipy`: statistics and distributions.
- `matplotlib`: validation plots.

---

# 4. AIS fields relevant to trajectory prediction

A decoded AIS position dataset commonly contains some subset of:

```text
MMSI
timestamp
latitude
longitude
SOG
COG
heading
ROT
navigation status
vessel type
length
width
IMO
name
```

Important definitions:

- **MMSI**: Maritime Mobile Service Identity, typically used as a vessel identifier in AIS data.
- **SOG**: Speed Over Ground, in knots.
- **COG**: Course Over Ground, the direction of movement over the Earth's surface.
- **Heading**: direction the vessel's bow points.
- **ROT**: Rate Of Turn.
- **Navigation status**: underway, at anchor, moored, etc.

COG and heading are not interchangeable.

Official AIS field definitions:

- USCG Class A position reports:  
  https://navcen.uscg.gov/ais-class-a-reports
- USCG AIS message overview:  
  https://www.navcen.uscg.gov/ais-messages

Class A AIS position messages include fields such as navigational status, ROT, SOG, position, COG, true heading and timestamp.

Special "not available" values exist in raw AIS encoding. For example:

- raw SOG `1023` = not available,
- raw COG `3600` = not available,
- raw heading `511` = not available,
- raw ROT `-128` = no turn information available.

If using an already-decoded CSV, first inspect how that source represents missing values. It may use `NaN`, empty cells, `511`, `360`, `3600`, or another convention.

---

# 5. No-account AIS data sources

## 5.1 NOAA / MarineCadastre bulk historical AIS

### Recommended use

Primary historical real-AIS source for the first prototype.

Official dataset page:

https://coast.noaa.gov/digitalcoast/data/vesseltraffic.html

NOAA/MarineCadastre publishes historical AIS point data derived from the U.S. Coast Guard's Nationwide AIS system.

Historical coverage currently includes 2009 onward, with later years distributed as daily CSV-based files.

### Direct bulk archive

A convenient year for initial work is **2024**, because NOAA exposes individual daily ZIP files.

2024 index:

https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/

Example daily file pattern:

```text
https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_01_01.zip
https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_01_02.zip
...
```

No platform account is required for these bulk files.

### Download one day with `wget`

```bash
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_01_01.zip
```

Or:

```bash
curl -LO https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_01_01.zip
```

Unzip:

```bash
unzip AIS_2024_01_01.zip -d data/raw/noaa/2024-01-01/
```

### Python downloader

```python
from pathlib import Path
from datetime import date, timedelta
import requests

OUT = Path("data/raw/noaa")
OUT.mkdir(parents=True, exist_ok=True)

def noaa_2024_url(day):
    ds = day.strftime("%Y_%m_%d")
    return (
        "https://coast.noaa.gov/htdata/CMSP/"
        f"AISDataHandler/2024/AIS_{ds}.zip"
    )

def download_file(url, path):
    path = Path(path)
    if path.exists():
        print("exists:", path)
        return

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for block in r.iter_content(chunk_size=1024 * 1024):
                if block:
                    f.write(block)

start = date(2024, 1, 1)

for i in range(3):
    day = start + timedelta(days=i)
    url = noaa_2024_url(day)
    out = OUT / f"AIS_{day.strftime('%Y_%m_%d')}.zip"
    print("downloading", url)
    download_file(url, out)
```

### Important NOAA note

The interactive **AccessAIS** tool is useful for geographic subsetting, but custom orders request an email address. For a no-account/no-order workflow, use the direct bulk daily files instead.

NOAA's dataset is primarily terrestrial U.S. AIS rather than global satellite AIS. It is therefore excellent for real vessel movement and port/coastal traffic, but it should not be treated as a global ocean dataset.

---

## 5.2 Danish Maritime Authority historical AIS

Official page:

https://www.dma.dk/safety-at-sea/navigational-information/ais-data

Historical archive:

https://aisdata.ais.dk/

The Danish Maritime Authority states that historical AIS data are continuously updated, free to download, saved as CSV and packaged as ZIP files.

This is particularly useful because Danish waters contain:

- dense commercial traffic,
- straits,
- ferries,
- cargo vessels,
- tankers,
- crossing interactions,
- constrained navigation.

The archive commonly uses names similar to:

```text
aisdk-YYYY-MM-DD.zip
```

and archived files may be organised by year.

A commonly observed direct pattern is:

```text
https://aisdata.ais.dk/aisdk-2026-10-01.zip
https://aisdata.ais.dk/2024/aisdk-2024-04-15.zip
```

The archive structure can change, so use the official archive index if a hard-coded URL returns `404`.

### Defensive Denmark downloader

```python
from pathlib import Path
import requests

def danish_candidates(year, month, day):
    name = f"aisdk-{year:04d}-{month:02d}-{day:02d}.zip"
    return [
        f"https://aisdata.ais.dk/{name}",
        f"https://aisdata.ais.dk/{year:04d}/{name}",
        f"http://aisdata.ais.dk/{name}",
        f"http://aisdata.ais.dk/{year:04d}/{name}",
    ]

def download_first_working(urls, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    continue

                with output_path.open("wb") as f:
                    for block in r.iter_content(1024 * 1024):
                        if block:
                            f.write(block)

                print("downloaded:", url)
                return url
        except requests.RequestException:
            pass

    raise RuntimeError("No candidate URL succeeded")

urls = danish_candidates(2024, 4, 15)

download_first_working(
    urls,
    "data/raw/denmark/aisdk-2024-04-15.zip"
)
```

### Recommended first Danish sample

Start with only **one to three days**.

A single day can already be large. Inspect before scaling.

---

## 5.3 Fintraffic / Digitraffic current AIS API

Official marine traffic page:

https://www.digitraffic.fi/en/marine-traffic/

AIS documentation:

https://www.digitraffic.fi/en/marine-traffic/ais/

Current vessel locations endpoint:

```text
https://meri.digitraffic.fi/api/ais/v1/locations
```

Fintraffic describes Digitraffic as open data. The API does not require a platform login or a paid contract.

This is best treated as a **current AIS snapshot / collection source**, not as the main historical training archive.

A simple collector can repeatedly save the current response and gradually create a local history.

### Save raw JSON snapshots

```python
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import requests

OUT = Path("data/raw/fintraffic")
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://meri.digitraffic.fi/api/ais/v1/locations"

HEADERS = {
    "User-Agent": "AIS-Research-Collector/1.0"
}

while True:
    now = datetime.now(timezone.utc)

    r = requests.get(
        URL,
        headers=HEADERS,
        timeout=60
    )
    r.raise_for_status()

    payload = r.json()

    path = OUT / f"ais_{now.strftime('%Y%m%dT%H%M%SZ')}.json"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8"
    )

    print("saved", path)

    # Adapt interval to the service's published usage guidance.
    time.sleep(30)
```

Before running long-term polling, check the current Digitraffic usage and rate-limit guidance.

---

## 5.4 Norwegian Coastal Administration open real-time AIS

Information page:

https://www.kystverket.no/en/sea-transport-and-ports/ais/access-to-ais-data/

The Norwegian Coastal Administration states that open AIS access is free and does not require user registration.

Its information page publishes an open AIS socket:

```text
host: 153.44.253.27
port: 5631
```

This is useful as an optional **real-time raw AIS stream**, not as the primary historical no-interaction download.

### Raw TCP capture

Linux/macOS with `nc`:

```bash
nc 153.44.253.27 5631 > norway_ais.nmea
```

Stop with `Ctrl+C`.

A Python version:

```python
import socket
from pathlib import Path

HOST = "153.44.253.27"
PORT = 5631

out = Path("data/raw/norway/norway_ais.nmea")
out.parent.mkdir(parents=True, exist_ok=True)

with socket.create_connection((HOST, PORT), timeout=30) as sock:
    with out.open("ab") as f:
        while True:
            data = sock.recv(65536)
            if not data:
                break
            f.write(data)
            f.flush()
```

Decode NMEA AIS later using `pyais`.

```bash
pip install pyais
```

Historical custom exports from Norway currently ask for an email recipient, so they are not the primary historical workflow in this document.

---

# 6. Sources to avoid as the initial data dependency

For the first reproducible dataset, avoid depending on:

- commercial AIS providers,
- platforms that require API keys,
- sites that allow visual vessel tracking but prohibit bulk extraction,
- scraped map interfaces,
- datasets with unclear provenance,
- small Kaggle mirrors when the authoritative source is available directly.

Commercial AIS can be useful later for global satellite coverage, but it is unnecessary for proving the first modelling pipeline.

---

# 7. Recommended real-data collection plan

## Phase A — get something working

Download:

```text
NOAA:
1 day

Denmark:
1 day
```

Goal:

- inspect columns,
- understand message volume,
- make the cleaning pipeline work,
- convert to Parquet.

## Phase B — first useful dataset

Collect roughly:

```text
NOAA:
7 days from one period

Denmark:
3–7 days
```

Do not download months of data before the cleaning pipeline is stable.

## Phase C — geographically separate evaluation

Keep another location/date completely out of training.

For example:

```text
TRAIN:
NOAA region/time A
+ Denmark period A

VALIDATION:
different dates

REAL TEST:
different dates or different geographic area
```

The final real test set must never be generated from synthetic data.

---

# 8. Inspect a downloaded file

After extraction:

```python
import pandas as pd

path = "data/raw/noaa/example.csv"

df = pd.read_csv(path)

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.dtypes)
print(df.isna().mean().sort_values(ascending=False).head(20))
```

For a large file, do not immediately load everything.

```python
sample = pd.read_csv(
    path,
    nrows=100_000
)

print(sample.head())
```

Or inspect with DuckDB:

```python
import duckdb

con = duckdb.connect()

result = con.execute("""
    SELECT *
    FROM read_csv_auto('data/raw/noaa/example.csv')
    LIMIT 20
""").df()

print(result)
```

---

# 9. Normalize every source to one schema

Different providers use different column names.

Create a common schema such as:

```text
source
timestamp_utc
mmsi
latitude
longitude
sog_kn
cog_deg
heading_deg
rot_deg_min
nav_status
vessel_type
length_m
width_m
imo
name
```

Not every source will contain every column.

Missing columns remain null.

Example normalization skeleton:

```python
import pandas as pd

NORMAL_COLUMNS = [
    "source",
    "timestamp_utc",
    "mmsi",
    "latitude",
    "longitude",
    "sog_kn",
    "cog_deg",
    "heading_deg",
    "rot_deg_min",
    "nav_status",
    "vessel_type",
    "length_m",
    "width_m",
    "imo",
    "name",
]

def normalize_source(df, mapping, source_name):
    out = pd.DataFrame()

    for target in NORMAL_COLUMNS:
        source_col = mapping.get(target)

        if source_col and source_col in df.columns:
            out[target] = df[source_col]
        else:
            out[target] = pd.NA

    out["source"] = source_name

    out["timestamp_utc"] = pd.to_datetime(
        out["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    return out
```

Keep a source-specific mapping file rather than hard-coding assumptions throughout the pipeline.

---

# 10. Basic AIS cleaning

Do not begin trajectory training directly from raw rows.

A reasonable first cleaning pipeline is:

```text
raw messages
    ↓
parse timestamps
    ↓
remove invalid coordinates
    ↓
sort by MMSI + time
    ↓
remove exact duplicates
    ↓
mark impossible / unavailable fields
    ↓
split long gaps into separate track segments
    ↓
detect implausible jumps
    ↓
retain a clean track table
```

## Coordinate validation

```python
mask = (
    df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
)

df = df[mask].copy()
```

## Sort

```python
df = df.sort_values(
    ["mmsi", "timestamp_utc"]
)
```

## Drop exact duplicates

```python
df = df.drop_duplicates()
```

## Identify time gaps

```python
df["dt_s"] = (
    df.groupby("mmsi")["timestamp_utc"]
      .diff()
      .dt.total_seconds()
)
```

Do not choose the final track-break threshold arbitrarily. Inspect the actual gap distribution first.

## Do not silently convert missing heading to COG

This is wrong:

```python
df["heading_deg"] = df["heading_deg"].fillna(df["cog_deg"])
```

Heading and COG measure different things.

---

# 11. Convert CSV to Parquet

Parquet is much more convenient for large ML datasets.

```python
import pandas as pd

df = pd.read_csv("large_ais.csv")

df.to_parquet(
    "data/normalized/ais.parquet",
    index=False
)
```

For files too large for RAM, use DuckDB or Polars to stream/partition them.

Example DuckDB conversion:

```python
import duckdb

duckdb.sql("""
COPY (
    SELECT *
    FROM read_csv_auto('large_ais.csv')
)
TO 'data/normalized/ais.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);
""")
```

---

# 12. What to learn from real AIS before generating synthetic AIS

The synthetic generator should eventually be calibrated from real AIS.

Measure these separately by vessel type where possible:

```text
SOG distribution
COG-change distribution
heading-COG difference
ROT distribution
acceleration distribution
deceleration distribution
message interval distribution
short gap distribution
long gap distribution
heading missing rate
ROT missing rate
navigation-status distribution
turn duration distribution
track curvature
```

Example:

```python
df = df.sort_values(["mmsi", "timestamp_utc"])

df["dt_s"] = (
    df.groupby("mmsi")["timestamp_utc"]
      .diff()
      .dt.total_seconds()
)

print(
    df["dt_s"]
    .dropna()
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
)
```

Course differences must handle 0/360 wraparound:

```python
import numpy as np

def signed_angle_difference(a, b):
    return (a - b + 180) % 360 - 180

df["prev_cog"] = (
    df.groupby("mmsi")["cog_deg"]
      .shift(1)
)

df["delta_cog_deg"] = signed_angle_difference(
    df["cog_deg"],
    df["prev_cog"]
)
```

Speed change:

```python
df["prev_sog"] = (
    df.groupby("mmsi")["sog_kn"]
      .shift(1)
)

df["delta_sog_kn"] = (
    df["sog_kn"] - df["prev_sog"]
)

df["accel_kn_per_min"] = (
    df["delta_sog_kn"]
    / (df["dt_s"] / 60)
)
```

Filter tiny and very large `dt_s` before treating this as a physical acceleration estimate.

---

# 13. Synthetic AIS design principles

Synthetic data should not be random points.

A useful simulator contains two separate layers:

```text
HIDDEN PHYSICAL WORLD
    │
    │ exact vessel positions and motion
    │
    ▼
AIS OBSERVATION MODEL
    │
    │ reporting intervals
    │ noise
    │ missing packets
    │ missing fields
    │ duplicates
    ▼
OBSERVED SYNTHETIC AIS
```

The ML model trains on **observed synthetic AIS**.

The labels come from **hidden ground truth**.

This distinction is essential.

---

# 14. Coordinate system for simulation

Do not initially simulate vessel motion directly in latitude/longitude.

Use local metric coordinates:

```text
x = metres east
y = metres north
```

Example simulated area:

```text
20 km × 20 km
```

Then convert to lat/lon for AIS-style output.

For a small local region, an approximate conversion is:

```python
from math import cos, radians

def xy_to_latlon(x_m, y_m, lat0, lon0):
    lat = lat0 + y_m / 111_320.0

    lon = lon0 + x_m / (
        111_320.0 * cos(radians(lat0))
    )

    return lat, lon
```

For a large region or production work, use a proper projected coordinate reference system with `pyproj`.

---

# 15. Vessel motion model

For the first simulator, each vessel has:

```text
position x/y
speed
course
heading
rate of turn
acceleration
navigation status
vessel type
length
width
```

If course is clockwise from north:

```text
east velocity  = speed × sin(course)
north velocity = speed × cos(course)
```

Convert knots to metres/second:

```text
1 knot = 0.514444 m/s
```

At each simulation step:

```text
speed(t+1) =
speed(t) + acceleration × dt

course(t+1) =
course(t) + turn_rate × dt

x(t+1) =
x(t) + east_velocity × dt

y(t+1) =
y(t) + north_velocity × dt
```

This gives:

- straight movement,
- acceleration,
- deceleration,
- constant-rate turns.

It is not yet a full ship-dynamics simulator. It is an intentionally simple kinematic simulator for ML dataset generation.

---

# 16. Encounter types to generate

The simulator should explicitly generate multiple encounter families.

## 16.1 Normal unrelated traffic

Several vessels move without coming particularly close.

Purpose:

- prevent the model from assuming every nearby vessel is dangerous.

## 16.2 Parallel traffic

```text
A -------------------->

B -------------------->
```

Vary:

- lane separation,
- vessel speeds,
- longitudinal offset.

## 16.3 Head-on geometry

```text
A -------------------->

       <-------------------- B
```

Vary:

- lateral offset,
- speed A,
- speed B,
- initial separation,
- manoeuvre time.

## 16.4 Crossing

```text
              B
              |
              |
              ↓

A -------------------->
```

Vary relative arrival time.

Two paths can geometrically intersect while the ships pass that point at different times.

## 16.5 Overtaking

```text
slow B ---------->

fast A -------------------->
```

Vary:

- speed difference,
- lateral separation,
- initial distance.

## 16.6 Turning / avoidance-like manoeuvre

The initial motion suggests a close approach, then one ship changes course or speed.

Purpose:

- generate non-linear futures,
- test whether a learned predictor can outperform a constant-velocity model.

## 16.7 Dense background traffic

Combine an encounter pair with additional vessels that are nearby but irrelevant.

Purpose:

- train multi-vessel interaction models,
- reduce overreaction to every surrounding vessel.

---

# 17. AIS reporting intervals

For Class A AIS, reporting frequency depends on vessel speed and course change.

USCG/IMO guidance describes intervals approximately as:

```text
anchor/moored, <= 3 kn    : 3 minutes
anchor/moored, > 3 kn     : 10 seconds
0–14 kn                   : 10 seconds
0–14 kn, changing course  : about 3 1/3 seconds
14–23 kn                  : 6 seconds
14–23 kn, changing course : 2 seconds
> 23 kn                   : 2 seconds
```

Reference:

https://navcen.uscg.gov/ais-class-a-reports

Do not simply emit one perfect observation every 30 seconds.

The simulator can internally update the vessel every second while transmitting AIS at an interval based on its current state.

---

# 18. Synthetic observation imperfections

A useful synthetic feed should include controlled imperfections.

## Position noise

```python
obs_x = true_x + random_position_error
obs_y = true_y + random_position_error
```

The exact noise distribution should eventually be calibrated from real data rather than assumed.

## SOG noise

```python
observed_sog = true_sog + noise
```

## COG noise

```python
observed_cog = true_cog + noise
```

## Missing heading

Some rows should contain no heading.

## Missing ROT

Some rows should contain no ROT.

## Message loss

```text
message 1 received
message 2 received
message 3 dropped
message 4 received
```

## Timestamp irregularity

Do not create perfectly periodic timestamps.

## Duplicate messages

Occasional duplicates can be generated so that the preprocessing pipeline learns to handle them.

---

# 19. Ground-truth labels

Do not begin by assigning arbitrary:

```text
LOW
MEDIUM
HIGH
```

risk classes.

Store continuous objective labels first.

For every pair of vessels A/B over the future horizon:

```text
minimum future separation
time of minimum future separation
```

If true future positions are:

```text
A(t) = (xA(t), yA(t))
B(t) = (xB(t), yB(t))
```

then:

```text
distance(t) =
sqrt(
    (xA(t) - xB(t))²
    +
    (yA(t) - yB(t))²
)
```

and:

```text
minimum future separation =
min(distance(t))
```

Later, an operational/domain policy can map these quantities and other encounter information into risk levels.

This avoids embedding arbitrary safety thresholds into the synthetic-data generator.

---

# 20. Complete starter synthetic AIS simulator

Save the following as:

```text
scripts/synthetic_ais_simulator.py
```

```python
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import cos, radians, sin
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


KNOT_TO_MPS = 0.514444


def wrap_deg(angle: float) -> float:
    return angle % 360.0


def xy_to_latlon(
    x_m: float,
    y_m: float,
    lat0_deg: float,
    lon0_deg: float,
) -> Tuple[float, float]:

    lat = lat0_deg + y_m / 111_320.0

    lon_scale = (
        111_320.0
        * max(cos(radians(lat0_deg)), 1e-6)
    )

    lon = lon0_deg + x_m / lon_scale

    return lat, lon


@dataclass
class Maneuver:
    start_s: int
    end_s: int
    turn_rate_deg_min: float = 0.0
    accel_kn_min: float = 0.0


@dataclass
class Vessel:
    vessel_id: str
    mmsi: int
    vessel_type: str

    x_m: float
    y_m: float

    sog_kn: float
    cog_deg: float

    heading_deg: Optional[float] = None

    nav_status: str = "Under way using engine"

    length_m: float = 120.0
    width_m: float = 20.0

    maneuvers: List[Maneuver] = field(
        default_factory=list
    )


@dataclass
class SimulatorConfig:

    duration_s: int = 20 * 60

    # Hidden state update.
    truth_step_s: int = 1

    # Save hidden truth every N seconds.
    truth_save_every_s: int = 5

    # Approximate geographic anchor.
    lat0_deg: float = 55.0
    lon0_deg: float = 12.0

    # Development defaults.
    # Replace/calibrate these from real AIS.
    position_noise_m: float = 7.0
    sog_noise_kn: float = 0.08
    cog_noise_deg: float = 0.6
    heading_noise_deg: float = 0.8

    drop_probability: float = 0.04
    duplicate_probability: float = 0.003

    heading_missing_probability: float = 0.08
    rot_missing_probability: float = 0.15

    timestamp_jitter_fraction: float = 0.08

    seed: int = 42


def active_controls(
    vessel: Vessel,
    t_s: int,
) -> Tuple[float, float]:

    rot = 0.0
    accel = 0.0

    for m in vessel.maneuvers:

        if m.start_s <= t_s < m.end_s:
            rot += m.turn_rate_deg_min
            accel += m.accel_kn_min

    return rot, accel


def class_a_nominal_interval_s(
    sog_kn: float,
    nav_status: str,
    turning: bool,
) -> float:

    at_anchor_or_moored = (
        nav_status in {
            "At anchor",
            "Moored",
        }
    )

    if at_anchor_or_moored and sog_kn <= 3:
        return 180.0

    if at_anchor_or_moored and sog_kn > 3:
        return 10.0

    if sog_kn < 14:
        return 10.0 / 3.0 if turning else 10.0

    if sog_kn <= 23:
        return 2.0 if turning else 6.0

    return 2.0


def simulate_truth(
    vessels: List[Vessel],
    cfg: SimulatorConfig,
) -> pd.DataFrame:

    # Copy the mutable simulation state.
    state: Dict[str, dict] = {}

    for v in vessels:

        heading_offset = (
            0.0
            if v.heading_deg is None
            else v.heading_deg - v.cog_deg
        )

        state[v.vessel_id] = {
            "vessel": v,
            "x_m": float(v.x_m),
            "y_m": float(v.y_m),
            "sog_kn": float(v.sog_kn),
            "cog_deg": float(v.cog_deg),
            "heading_offset_deg": heading_offset,
        }

    rows = []

    for t_s in range(
        0,
        cfg.duration_s + 1,
        cfg.truth_step_s,
    ):

        for vessel_id, s in state.items():

            vessel = s["vessel"]

            rot_deg_min, accel_kn_min = (
                active_controls(
                    vessel,
                    t_s,
                )
            )

            if t_s % cfg.truth_save_every_s == 0:

                heading = wrap_deg(
                    s["cog_deg"]
                    + s["heading_offset_deg"]
                )

                lat, lon = xy_to_latlon(
                    s["x_m"],
                    s["y_m"],
                    cfg.lat0_deg,
                    cfg.lon0_deg,
                )

                rows.append({
                    "vessel_id": vessel_id,
                    "mmsi": vessel.mmsi,
                    "vessel_type": vessel.vessel_type,
                    "t_s": t_s,

                    "true_x_m": s["x_m"],
                    "true_y_m": s["y_m"],

                    "true_lat": lat,
                    "true_lon": lon,

                    "true_sog_kn": s["sog_kn"],
                    "true_cog_deg": s["cog_deg"],
                    "true_heading_deg": heading,

                    "true_rot_deg_min": rot_deg_min,

                    "nav_status": vessel.nav_status,

                    "length_m": vessel.length_m,
                    "width_m": vessel.width_m,
                })

            # ----------------------------------
            # Propagate hidden physical state
            # ----------------------------------

            dt_min = cfg.truth_step_s / 60.0

            s["sog_kn"] = max(
                0.0,
                s["sog_kn"]
                + accel_kn_min * dt_min,
            )

            s["cog_deg"] = wrap_deg(
                s["cog_deg"]
                + rot_deg_min * dt_min
            )

            speed_mps = (
                s["sog_kn"]
                * KNOT_TO_MPS
            )

            theta = radians(
                s["cog_deg"]
            )

            s["x_m"] += (
                speed_mps
                * sin(theta)
                * cfg.truth_step_s
            )

            s["y_m"] += (
                speed_mps
                * cos(theta)
                * cfg.truth_step_s
            )

    return pd.DataFrame(rows)


def generate_ais_observations(
    truth: pd.DataFrame,
    cfg: SimulatorConfig,
) -> pd.DataFrame:

    """
    Convert hidden ground truth into an imperfect
    decoded AIS-like feed.

    This simulates decoded AIS observations.
    It does not generate binary/NMEA radio packets.
    """

    rng = np.random.default_rng(
        cfg.seed
    )

    rows = []

    for vessel_id, g in truth.groupby(
        "vessel_id",
        sort=False,
    ):

        g = (
            g.sort_values("t_s")
             .reset_index(drop=True)
        )

        next_emit_s = 0.0

        for _, row in g.iterrows():

            t_s = float(
                row["t_s"]
            )

            if t_s + 1e-9 < next_emit_s:
                continue

            turning = (
                abs(
                    float(
                        row["true_rot_deg_min"]
                    )
                )
                > 0.05
            )

            interval = (
                class_a_nominal_interval_s(
                    float(row["true_sog_kn"]),
                    str(row["nav_status"]),
                    turning,
                )
            )

            jitter = rng.normal(
                0.0,
                (
                    cfg.timestamp_jitter_fraction
                    * interval
                ),
            )

            next_emit_s = (
                t_s
                + max(
                    1.0,
                    interval + jitter,
                )
            )

            # Simulated dropped AIS message.
            if (
                rng.random()
                < cfg.drop_probability
            ):
                continue

            obs_x = (
                float(row["true_x_m"])
                + rng.normal(
                    0.0,
                    cfg.position_noise_m,
                )
            )

            obs_y = (
                float(row["true_y_m"])
                + rng.normal(
                    0.0,
                    cfg.position_noise_m,
                )
            )

            lat, lon = xy_to_latlon(
                obs_x,
                obs_y,
                cfg.lat0_deg,
                cfg.lon0_deg,
            )

            sog = max(
                0.0,
                (
                    float(row["true_sog_kn"])
                    + rng.normal(
                        0.0,
                        cfg.sog_noise_kn,
                    )
                ),
            )

            cog = wrap_deg(
                float(row["true_cog_deg"])
                + rng.normal(
                    0.0,
                    cfg.cog_noise_deg,
                )
            )

            heading = None

            if (
                rng.random()
                >= cfg.heading_missing_probability
            ):

                heading = wrap_deg(
                    float(
                        row["true_heading_deg"]
                    )
                    + rng.normal(
                        0.0,
                        cfg.heading_noise_deg,
                    )
                )

            rot = None

            if (
                rng.random()
                >= cfg.rot_missing_probability
            ):
                rot = float(
                    row["true_rot_deg_min"]
                )

            observation = {
                "vessel_id": vessel_id,
                "mmsi": int(row["mmsi"]),
                "vessel_type": row["vessel_type"],

                "t_s": int(row["t_s"]),

                "x_m": obs_x,
                "y_m": obs_y,

                "latitude": lat,
                "longitude": lon,

                "sog_kn": sog,
                "cog_deg": cog,
                "heading_deg": heading,
                "rot_deg_min": rot,

                "nav_status": row["nav_status"],

                "length_m": float(row["length_m"]),
                "width_m": float(row["width_m"]),
            }

            rows.append(
                observation
            )

            # Occasional exact duplicate.
            if (
                rng.random()
                < cfg.duplicate_probability
            ):
                rows.append(
                    dict(observation)
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["t_s", "mmsi"]
        )
        .reset_index(drop=True)
    )


def pairwise_future_labels(
    truth: pd.DataFrame,
    forecast_start_s: int,
) -> pd.DataFrame:

    """
    Continuous pairwise future labels.

    No LOW/MEDIUM/HIGH operational thresholds
    are assumed.
    """

    future = truth[
        truth["t_s"] >= forecast_start_s
    ].copy()

    vessel_ids = sorted(
        future["vessel_id"]
        .unique()
    )

    indexed = {}

    for vessel_id in vessel_ids:

        indexed[vessel_id] = (
            future[
                future["vessel_id"]
                == vessel_id
            ][
                [
                    "t_s",
                    "true_x_m",
                    "true_y_m",
                ]
            ]
            .rename(columns={
                "true_x_m":
                    f"x_{vessel_id}",

                "true_y_m":
                    f"y_{vessel_id}",
            })
        )

    results = []

    for a, b in combinations(
        vessel_ids,
        2,
    ):

        merged = indexed[a].merge(
            indexed[b],
            on="t_s",
            how="inner",
        )

        if merged.empty:
            continue

        distance = np.hypot(
            (
                merged[f"x_{a}"]
                - merged[f"x_{b}"]
            ),
            (
                merged[f"y_{a}"]
                - merged[f"y_{b}"]
            ),
        )

        idx = int(
            np.argmin(
                distance.to_numpy()
            )
        )

        t_min = int(
            merged["t_s"].iloc[idx]
        )

        results.append({
            "vessel_a": a,
            "vessel_b": b,

            "min_future_separation_m":
                float(distance.iloc[idx]),

            "t_of_min_separation_s":
                t_min,

            "time_from_forecast_start_s":
                t_min - forecast_start_s,
        })

    return pd.DataFrame(
        results
    )


# ========================================================
# SCENE GENERATORS
# ========================================================


def make_head_on_scene(
    rng: np.random.Generator,
) -> List[Vessel]:

    lateral_offset = rng.uniform(
        -300.0,
        300.0,
    )

    return [

        Vessel(
            vessel_id="A",
            mmsi=990000001,
            vessel_type="Cargo",

            x_m=-4500.0,
            y_m=0.0,

            sog_kn=float(
                rng.uniform(9.0, 14.0)
            ),

            cog_deg=90.0,
            heading_deg=90.0,

            length_m=180.0,
            width_m=28.0,
        ),

        Vessel(
            vessel_id="B",
            mmsi=990000002,
            vessel_type="Tanker",

            x_m=4500.0,
            y_m=lateral_offset,

            sog_kn=float(
                rng.uniform(8.0, 13.0)
            ),

            cog_deg=270.0,
            heading_deg=270.0,

            length_m=210.0,
            width_m=32.0,
        ),
    ]


def make_crossing_scene(
    rng: np.random.Generator,
) -> List[Vessel]:

    x_offset = rng.uniform(
        -700.0,
        700.0,
    )

    y_offset = rng.uniform(
        -700.0,
        700.0,
    )

    return [

        Vessel(
            vessel_id="A",
            mmsi=990000011,
            vessel_type="Cargo",

            x_m=-4500.0,
            y_m=y_offset,

            sog_kn=float(
                rng.uniform(8.0, 15.0)
            ),

            cog_deg=90.0,
            heading_deg=90.0,

            length_m=160.0,
            width_m=25.0,
        ),

        Vessel(
            vessel_id="B",
            mmsi=990000012,
            vessel_type="Passenger",

            x_m=x_offset,
            y_m=4500.0,

            sog_kn=float(
                rng.uniform(8.0, 18.0)
            ),

            cog_deg=180.0,
            heading_deg=180.0,

            length_m=140.0,
            width_m=24.0,
        ),
    ]


def make_overtaking_scene(
    rng: np.random.Generator,
) -> List[Vessel]:

    return [

        Vessel(
            vessel_id="A",
            mmsi=990000021,
            vessel_type="Passenger",

            x_m=-4800.0,
            y_m=-100.0,

            sog_kn=float(
                rng.uniform(14.0, 18.0)
            ),

            cog_deg=90.0,
            heading_deg=90.0,

            length_m=130.0,
            width_m=22.0,
        ),

        Vessel(
            vessel_id="B",
            mmsi=990000022,
            vessel_type="Cargo",

            x_m=-2200.0,
            y_m=100.0,

            sog_kn=float(
                rng.uniform(7.0, 11.0)
            ),

            cog_deg=90.0,
            heading_deg=90.0,

            length_m=190.0,
            width_m=30.0,
        ),
    ]


def make_turning_crossing_scene(
    rng: np.random.Generator,
) -> List[Vessel]:

    vessels = make_crossing_scene(
        rng
    )

    # Vessel A starts a controlled turn
    # after six minutes.
    vessels[0].maneuvers.append(
        Maneuver(
            start_s=360,
            end_s=600,
            turn_rate_deg_min=5.0,
        )
    )

    return vessels


SCENE_BUILDERS = {
    "head_on":
        make_head_on_scene,

    "crossing":
        make_crossing_scene,

    "overtaking":
        make_overtaking_scene,

    "turning_crossing":
        make_turning_crossing_scene,
}


def simulate_scene(
    scene_type: str,
    cfg: SimulatorConfig,
    seed: int,
):

    rng = np.random.default_rng(
        seed
    )

    vessels = (
        SCENE_BUILDERS[scene_type](
            rng
        )
    )

    truth = simulate_truth(
        vessels,
        cfg,
    )

    local_cfg = SimulatorConfig(
        **{
            **cfg.__dict__,
            "seed": seed,
        }
    )

    ais = generate_ais_observations(
        truth,
        local_cfg,
    )

    labels = pairwise_future_labels(
        truth,
        forecast_start_s=10 * 60,
    )

    truth.insert(
        0,
        "scene_type",
        scene_type,
    )

    ais.insert(
        0,
        "scene_type",
        scene_type,
    )

    labels.insert(
        0,
        "scene_type",
        scene_type,
    )

    return truth, ais, labels


if __name__ == "__main__":

    cfg = SimulatorConfig()

    truth, ais, labels = (
        simulate_scene(
            "turning_crossing",
            cfg,
            seed=123,
        )
    )

    truth.to_csv(
        "synthetic_truth.csv",
        index=False,
    )

    ais.to_csv(
        "synthetic_ais.csv",
        index=False,
    )

    labels.to_csv(
        "synthetic_pairwise_labels.csv",
        index=False,
    )

    print(truth.head())
    print(ais.head())
    print(labels)
```

---

# 21. Run the simulator

```bash
python scripts/synthetic_ais_simulator.py
```

Expected files:

```text
synthetic_truth.csv
synthetic_ais.csv
synthetic_pairwise_labels.csv
```

Meaning:

### `synthetic_truth.csv`

Perfect hidden vessel state.

Use this for target trajectories.

### `synthetic_ais.csv`

Imperfect decoded AIS-like observations.

Use this as model input.

### `synthetic_pairwise_labels.csv`

Future pairwise minimum separation and time-to-minimum-separation.

Use this for encounter/risk experiments.

---

# 22. Generate many scenes

A simple bulk generator:

```python
from pathlib import Path
import pandas as pd

from synthetic_ais_simulator import (
    SimulatorConfig,
    simulate_scene,
)

OUT = Path("data/synthetic")
OUT.mkdir(parents=True, exist_ok=True)

cfg = SimulatorConfig()

scene_types = [
    "head_on",
    "crossing",
    "overtaking",
    "turning_crossing",
]

all_truth = []
all_ais = []
all_labels = []

scene_number = 0

for scene_type in scene_types:

    for repetition in range(250):

        scene_number += 1

        truth, ais, labels = simulate_scene(
            scene_type,
            cfg,
            seed=10_000 + scene_number,
        )

        scene_id = (
            f"SYN_{scene_number:06d}"
        )

        truth.insert(
            0,
            "scene_id",
            scene_id,
        )

        ais.insert(
            0,
            "scene_id",
            scene_id,
        )

        labels.insert(
            0,
            "scene_id",
            scene_id,
        )

        all_truth.append(truth)
        all_ais.append(ais)
        all_labels.append(labels)


truth_df = pd.concat(
    all_truth,
    ignore_index=True,
)

ais_df = pd.concat(
    all_ais,
    ignore_index=True,
)

labels_df = pd.concat(
    all_labels,
    ignore_index=True,
)


truth_df.to_parquet(
    OUT / "truth.parquet",
    index=False,
)

ais_df.to_parquet(
    OUT / "ais_observed.parquet",
    index=False,
)

labels_df.to_parquet(
    OUT / "pairwise_labels.parquet",
    index=False,
)

print(
    "truth rows:",
    len(truth_df)
)

print(
    "AIS observations:",
    len(ais_df)
)

print(
    "pair labels:",
    len(labels_df)
)
```

This example creates:

```text
4 scenario families
×
250 scenes
=
1,000 scenes
```

Do not immediately assume 250 per category is the correct training distribution. The distribution should be adjusted after real-data analysis and evaluation.

---

# 23. Add random background vessels

The two-vessel scenes are useful for debugging, but a final multi-vessel model should also see irrelevant surrounding traffic.

Example generator:

```python
def make_background_vessel(
    rng,
    index,
):

    course = float(
        rng.uniform(0, 360)
    )

    speed = float(
        rng.uniform(4, 16)
    )

    return Vessel(
        vessel_id=f"BG{index:02d}",

        mmsi=(
            991000000
            + index
        ),

        vessel_type="Other",

        x_m=float(
            rng.uniform(
                -8000,
                8000,
            )
        ),

        y_m=float(
            rng.uniform(
                -8000,
                8000,
            )
        ),

        sog_kn=speed,

        cog_deg=course,
        heading_deg=course,

        length_m=float(
            rng.uniform(
                60,
                220,
            )
        ),

        width_m=float(
            rng.uniform(
                10,
                35,
            )
        ),
    )
```

Then:

```python
vessels = make_crossing_scene(rng)

for i in range(8):
    vessels.append(
        make_background_vessel(
            rng,
            i,
        )
    )
```

The next step is to make background traffic route-aware instead of purely random.

---

# 24. Improve realism using real AIS

The defaults in the starter simulator are development values.

Do not claim they represent real traffic until calibrated.

Replace simulator assumptions with empirical distributions.

## Example: vessel speed by type

```python
import pandas as pd

df = pd.read_parquet(
    "data/normalized/ais.parquet"
)

speed_stats = (
    df.groupby("vessel_type")["sog_kn"]
      .describe(
          percentiles=[
              0.1,
              0.25,
              0.5,
              0.75,
              0.9,
          ]
      )
)

print(speed_stats)
```

Instead of:

```python
rng.uniform(8, 15)
```

eventually sample from the empirical distribution for the relevant vessel type.

## Empirical resampling

```python
cargo_speeds = (
    df.loc[
        df["vessel_type"] == "Cargo",
        "sog_kn"
    ]
    .dropna()
    .to_numpy()
)

synthetic_speed = float(
    rng.choice(
        cargo_speeds
    )
)
```

This is a simple but useful calibration technique.

---

# 25. Calibrate AIS message loss and missing fields

From real data:

```python
heading_missing_rate = (
    df["heading_deg"]
    .isna()
    .mean()
)

rot_missing_rate = (
    df["rot_deg_min"]
    .isna()
    .mean()
)

print(
    heading_missing_rate,
    rot_missing_rate,
)
```

Then:

```python
cfg.heading_missing_probability = (
    heading_missing_rate
)

cfg.rot_missing_probability = (
    rot_missing_rate
)
```

Message-loss probability cannot be estimated simply from missing rows because the expected reporting interval itself depends on vessel state.

For V1:

- measure observed gap distributions,
- compare them by speed/status,
- simulate gaps that produce similar distributions.

---

# 26. Real-trajectory augmentation

Fully simulated vessels are only one source of synthetic data.

A stronger augmentation technique is to take real trajectories and transform them.

If a track is stored in local XY coordinates:

```text
x(t)
y(t)
```

it can be:

- translated,
- rotated,
- slightly time-scaled,
- slightly speed-scaled,
- given additional observation noise,
- thinned to create missing observations.

## Rotate a real trajectory

```python
import numpy as np

def rotate_xy(
    x,
    y,
    degrees,
):

    theta = np.deg2rad(
        degrees
    )

    c = np.cos(theta)
    s = np.sin(theta)

    xr = c * x - s * y
    yr = s * x + c * y

    return xr, yr
```

## Translate

```python
xr = xr + offset_x
yr = yr + offset_y
```

This preserves the shape of a human-operated vessel trajectory while creating a new geographic placement.

Do not rotate raw latitude/longitude directly. Convert to a local projected XY system first.

---

# 27. Build synthetic encounters from real trajectory pieces

A useful advanced strategy:

```text
real trajectory A
+
real trajectory B
    │
    ▼
convert both to local XY
    │
    ▼
rotate / translate / time-shift
    │
    ▼
construct:
crossing / overtaking / close pass
```

This provides:

- realistic turn geometry,
- realistic acceleration,
- realistic speed fluctuations,

while still giving control over encounter geometry.

The resulting encounter should be checked for:

- impossible overlap,
- impossible acceleration,
- unrealistic speeds,
- vessel paths on land if using a real geographic region.

---

# 28. Add route and geography constraints later

Synthetic V1 can operate on an open plane.

Synthetic V2 should include:

```text
coastline
channel centreline
traffic-separation lanes
port entrance
restricted polygons
depth constraints
```

Then vessel waypoints are sampled along legal/plausible navigable routes.

Useful geospatial Python libraries:

```bash
pip install geopandas shapely pyproj
```

Possible validation:

```python
assert not vessel_track.intersects(land_polygon)
```

The exact navigation constraints should come from appropriate chart/geospatial data, not be guessed.

---

# 29. Create model-ready training windows

Suppose the model receives:

```text
past 10 minutes
```

and predicts:

```text
next 10 minutes
```

with a fixed 30-second grid.

Then each vessel needs:

```text
20 past timesteps
20 future timesteps
```

A basic feature set:

```text
x
y
SOG
sin(COG)
cos(COG)
sin(heading)
cos(heading)
ROT
delta_time
```

Represent angles as sine/cosine:

```python
import numpy as np

df["cog_sin"] = np.sin(
    np.deg2rad(df["cog_deg"])
)

df["cog_cos"] = np.cos(
    np.deg2rad(df["cog_deg"])
)
```

This prevents the model from treating:

```text
359 degrees
1 degree
```

as if they were far apart.

---

# 30. Resampling irregular AIS for a simple baseline

For a first GRU/TCN model, it is often easier to work on a regular time grid.

Example concept:

```python
def resample_one_track(
    track,
    interval="30s",
):

    track = (
        track
        .sort_values("timestamp_utc")
        .set_index("timestamp_utc")
    )

    numeric = [
        "x_m",
        "y_m",
        "sog_kn",
    ]

    out = (
        track[numeric]
        .resample(interval)
        .mean()
        .interpolate(
            limit=2
        )
    )

    return out
```

Do not interpolate across large gaps.

Keep a:

```text
valid_mask
```

or:

```text
time_since_last_observation
```

feature so the model can distinguish real observations from interpolated/old information.

A later model can directly handle irregular timestamps.

---

# 31. Train / validation / test split

Do not randomly split individual AIS rows.

Bad:

```text
80% rows → train
20% rows → test
```

Adjacent observations from the same vessel track would leak between splits.

Prefer:

```text
TRAIN
different dates / voyages / track segments

VALIDATION
different dates / voyages

TEST
different dates / voyages
```

For a stronger test, reserve a separate geographic area.

For synthetic data, split by:

```text
scene_id
```

not by individual observation.

All rows belonging to one synthetic scene must stay together.

---

# 32. Real-only final test

This is essential for evaluating synthetic data.

Create:

```text
REAL_TEST
```

before synthetic augmentation.

Never add synthetic scenes to this test set.

Train:

```text
MODEL A
real only
```

and:

```text
MODEL B
real + synthetic
```

Evaluate both on exactly:

```text
REAL_TEST
```

Then synthetic data is useful only if Model B performs better on real unseen traffic.

---

# 33. Recommended experiments

## Experiment 1 — motion baseline

Use:

```text
constant velocity
```

and optionally:

```text
Kalman filter
```

Metrics:

```text
ADE
FDE
```

The learned model should beat simple motion baselines.

## Experiment 2 — real-only GRU

Input:

```text
10 min AIS history
```

Predict:

```text
10 min trajectory
```

## Experiment 3 — real + synthetic GRU

Same architecture.

Only the training dataset changes.

Compare on real-only test.

## Experiment 4 — interaction-aware model

Add nearby vessels.

Possible later model:

```text
temporal encoder
+
graph neural network
```

## Experiment 5 — risk model

Use predicted future trajectories to estimate:

```text
minimum future separation
time to minimum separation
```

Then evaluate risk classification only after a risk policy and labels are defined.

---

# 34. Trajectory metrics

## ADE

Average Displacement Error:

```text
average distance between
predicted and true positions
over the full forecast horizon
```

## FDE

Final Displacement Error:

```text
distance between predicted and true
position at the final forecast time
```

Report by forecast horizon:

```text
+1 min
+5 min
+10 min
+20 min
```

Do not report only one vague "trajectory accuracy" percentage.

---

# 35. Collision-relevant evaluation

A trajectory model can have reasonable ADE while still predicting the wrong closest approach.

Also measure:

```text
error in minimum future separation
error in time to minimum separation
false close-approach alarms
missed close approaches
```

If a categorical collision-risk classifier is later created:

```text
precision
recall
F1
PR-AUC
false-alarm rate
miss rate
```

are more useful than raw accuracy when high-risk events are rare.

---

# 36. Validate synthetic data before training

Synthetic data should be compared to real data.

At minimum compare:

```text
SOG distribution
message intervals
course-change distribution
ROT distribution
acceleration distribution
missing-heading rate
missing-ROT rate
trajectory curvature
vessel-type proportions
```

Example:

```python
import matplotlib.pyplot as plt

real["sog_kn"].dropna().hist(
    bins=80,
    density=True,
    alpha=0.5,
    label="real",
)

synthetic["sog_kn"].dropna().hist(
    bins=80,
    density=True,
    alpha=0.5,
    label="synthetic",
)

plt.xlabel("SOG (knots)")
plt.ylabel("Density")
plt.legend()
plt.show()
```

Do not force synthetic and real distributions to be identical.

The goal is:

- realistic normal behaviour,
- plus deliberately oversampled rare/important encounters.

Keep metadata identifying synthetic rows.

---

# 37. Recommended dataset metadata

Every training scene should include metadata:

```text
scene_id
source_type
real_or_synthetic
source_provider
region
start_time
end_time
scenario_family
number_of_vessels
seed
simulator_version
```

Synthetic rows should include:

```text
simulator_version
random_seed
configuration_hash
```

This makes every scene reproducible.

---

# 38. Synthetic simulator configuration file

Instead of changing source code constantly, store parameters in YAML.

Example:

```yaml
version: 0.1

simulation:
  duration_s: 1200
  truth_step_s: 1
  truth_save_every_s: 5

observation:
  position_noise_m: 7.0
  sog_noise_kn: 0.08
  cog_noise_deg: 0.6
  heading_noise_deg: 0.8

  drop_probability: 0.04
  duplicate_probability: 0.003

  heading_missing_probability: 0.08
  rot_missing_probability: 0.15

scenes:
  head_on: 250
  crossing: 250
  overtaking: 250
  turning_crossing: 250
```

Treat these as development defaults until calibrated.

---

# 39. Version the generator

Synthetic data should always record which generator produced it.

Example:

```text
simulator_version = "0.1.0"
```

If the motion model changes:

```text
0.1.0 → 0.2.0
```

Generate a new dataset instead of silently overwriting the old one.

Recommended generated-data folder:

```text
data/synthetic/v0.1.0/
data/synthetic/v0.2.0/
```

---

# 40. Minimum end-to-end workflow

The first complete pass should be:

```text
1. Download 1 NOAA day
2. Download 1 Denmark day
3. Inspect schemas
4. Normalize columns
5. Convert to Parquet
6. Plot real speed / COG / gap distributions
7. Run synthetic simulator
8. Plot synthetic distributions
9. Adjust simulator
10. Generate 1,000 synthetic scenes
11. Build real-only baseline dataset
12. Reserve real-only test set
13. Train constant-velocity baseline
14. Train simple GRU on real data
15. Train same GRU on real + synthetic
16. Compare both on real-only test data
```

Only after this pipeline works should the dataset be expanded substantially.

---

# 41. Recommended progression

## Version 0

```text
1 real AIS day
100 synthetic scenes
2-vessel scenarios
```

Purpose:

- code correctness.

## Version 1

```text
7–14 real AIS days
1,000–5,000 synthetic scenes
realistic message timing
real-calibrated basic distributions
```

Purpose:

- first serious trajectory model.

## Version 2

```text
multi-region real AIS
background traffic
route constraints
real-trajectory augmentation
multi-vessel encounters
```

Purpose:

- interaction-aware model.

## Version 3

```text
authorised target-region historical AIS
environmental/geographic features
operational risk policy
```

Purpose:

- domain-specific evaluation.

---

# 42. Common mistakes

Avoid:

### Random synthetic positions

```text
lat = random
lon = random
speed = random
```

This is not realistic vessel movement.

### Training only on synthetic data

The model can learn the simulator instead of the real world.

### Testing synthetic augmentation on synthetic data

Synthetic data should prove itself on a real-only test set.

### Replacing heading with COG

They are different fields.

### Using raw accuracy for rare collision risk

A model can obtain high accuracy by predicting low risk almost everywhere.

### Random row-level splitting

This leaks adjacent trajectory points.

### Perfect synthetic AIS

Real feeds contain irregular timing, missing fields and missing messages.

### Inventing operational collision thresholds

Store continuous closest-approach labels first. Define operational risk thresholds separately with the appropriate domain authority.

### Downloading huge archives before the pipeline works

Start with one day.

---

# 43. Source references

## NOAA / MarineCadastre

Vessel traffic dataset:

https://coast.noaa.gov/digitalcoast/data/vesseltraffic.html

2024 bulk daily archive:

https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/

AIS FAQ:

https://coast.noaa.gov/data/marinecadastre/ais/faq.pdf

## Danish Maritime Authority

AIS information:

https://www.dma.dk/safety-at-sea/navigational-information/ais-data

Historical archive:

https://aisdata.ais.dk/

## Fintraffic / Digitraffic

Marine traffic:

https://www.digitraffic.fi/en/marine-traffic/

AIS documentation:

https://www.digitraffic.fi/en/marine-traffic/ais/

Current location endpoint:

https://meri.digitraffic.fi/api/ais/v1/locations

Swagger:

https://meri.digitraffic.fi/swagger/

## Norwegian Coastal Administration

AIS access:

https://www.kystverket.no/en/sea-transport-and-ports/ais/access-to-ais-data/

## US Coast Guard AIS technical reference

Class A position reports:

https://navcen.uscg.gov/ais-class-a-reports

AIS messages:

https://www.navcen.uscg.gov/ais-messages

AIS types/reporting behaviour:

https://navcen.uscg.gov/types-of-ais

---

# 44. First practical target

Before building a large ML model, produce these five artifacts:

```text
1. real_ais_normalized.parquet

2. real_ais_statistics.json

3. synthetic_truth.parquet

4. synthetic_ais_observed.parquet

5. synthetic_pairwise_labels.parquet
```

At that point the project has:

- real vessel movement,
- a reproducible synthetic generator,
- hidden future ground truth,
- encounter labels,
- a clean foundation for GRU/TCN/GNN trajectory experiments.

The next engineering milestone is then a **constant-velocity baseline and a simple GRU trajectory predictor**, evaluated on a real-only held-out dataset.
