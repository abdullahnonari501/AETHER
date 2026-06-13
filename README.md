# Project A.E.T.H.E.R.
### GPS-Denied Autonomous UAV — Vision-Guided Waypoint Navigation

> **2nd Place, Final Year Project Awards — GIK Institute, 2026** · Recognized at the GIKI Annual Industrial Open House 2026

A quadcopter that autonomously locates, identifies, and centers over a sequence of visual targets **with no GPS at any point** — using onboard optical-flow/LiDAR sensor fusion for state estimation and a custom-trained YOLOv8 detector for target identification, all running on a Jetson Nano companion computer.

**Demo mission:** visit 4 colored waypoint boxes at the corners of a 10 m square (WHITE → GREEN → RED → BLUE → WHITE). At each box: detect, confirm via multi-frame vote, center using vision, hover stable for 5 s, yaw toward the next target, transit by dead reckoning.

---

## Why GPS-denied?

GPS is trivially jammed or spoofed, which makes GPS-dependent autonomy useless exactly where autonomy matters most. AETHER's entire navigation stack — altitude, horizontal velocity, position hold — runs on local sensing: an MTF-02P optical-flow + LiDAR module fused by ArduPilot's EKF3, with vision closing the loop on target position.

---

## System architecture

**Sequential 20 Hz sense → decide → act loop** on the Jetson Nano, commanding ArduCopter GUIDED mode over MAVLink.

> At 20 Hz with a ~270 ms end-to-end latency budget, threading added concurrency risk and debugging complexity with no meaningful latency benefit at this scale. The design evolved during implementation; this repository — the code — is the source of truth.

```
Camera (640x480) --> YOLOv8-Nano (Docker, Ultralytics) --> Detections
                                                              |
MTF-02P (optical flow + LiDAR, UART) --> FC EKF3 --> Telemetry|
                                                              v
                                              12-state mission state machine
                                                              |
                                              Velocity / yaw commands (MAVLink)
                                                              v
                                          STM32H743 FC - ArduCopter GUIDED mode
```

### Mission state machine (12 states)

```
INIT -> WAIT_FOR_GUIDED -> (AUTO_TAKEOFF) -> TRANSIT -> SEARCH -> SEARCH_YAW
                                               ^                      |
                                               |                      v
                                          YAW_TO_NEXT <- HOVER_AT_BOX <- CENTER <- COMMIT

Terminal states: MISSION_COMPLETE, ABORT
```

### Key design decisions

| Decision | Value | Rationale |
|---|---|---|
| **COMMIT multi-frame vote** | 6 frames; winner needs avg conf >= 0.65 **and** >= 0.20 lead over runner-up | A single frame can misclassify at color boundaries under field lighting; a margin-gated vote suppresses it |
| **PD pixel-error centering** | KP = 0.0006, KD = 0.0001; image -> body-NED transform | Smooth convergence over the target without oscillation |
| **Velocity hard-clamp** | 0.25 m/s global, 0.20 m/s during centering | Derived from the 270 ms latency budget: 0.25 m/s x 0.27 s ~= 6.75 cm worst-case error before the next correction — a safety factor against the centering tolerance |
| **Transit by dead reckoning** | time x commanded velocity, no GPS | Known failure modes documented below |

### Safety architecture

- **Pilot always wins:** mode checked every loop; leaving GUIDED instantly releases controller authority
- **Heartbeat watchdog:** FC silent > 2 s -> halt
- **Low-voltage cutoff:** < 11.0 V -> forced land
- **Per-state timeouts:** 60 s max in any state (30 s for SEARCH_YAW) — the drone cannot get stuck

---

## Perception: two models, one navigation stack

**1. Campus landmark detector (concept validation).** Trained on real aerial footage of the GIKI campus across 6 building classes (Admin, Auditorium, Library, Logik, MGS, ORIC), ~2,430 labeled instances.
**Results: mAP@0.5 = 0.969 - F1 = 0.94 @ 0.745 confidence - recall 0.98.** Training curves, PR curves, and confusion matrices are in `Model/Model_1/`.

**2. Proxy-target detector (deployed in flight).** The field demo could not be flown at campus scale, so colored boxes served as proxy landmarks and a second model was trained for them: **350 images, 80 epochs, mAP 0.83**, deployed via the Ultralytics Docker container on the Jetson. Weights and detection samples are in `Model/Model_2/`.

The navigation stack is identical for both — only the detector weights change. Validating perception at real scale while flying a proxy-scale demo was a deliberate scaling decision, not a limitation discovered late.

---

## GPS-denied state estimation (EKF3) — measured performance

| Metric | Result |
|---|---|
| Yaw drift | < 2 deg over a 24 s rotation |
| Altitude std dev | 3.4 cm over 60 s |
| Horizontal drift | < 30 cm in a 60 s loiter |

---

## Hardware

| Component | Choice |
|---|---|
| Frame | F450 quadcopter |
| Flight controller | DakeFPV H743 (STM32H743), ArduCopter |
| Companion computer | Jetson Nano (JetPack 4.6), Ultralytics Docker |
| Flow/range sensor | MTF-02P (optical flow + LiDAR, single UART) |
| Camera | USB, 640x480 |

**The hardware journey is part of the engineering:**
- Started on a SpeedyBee F405 V4; its faulty barometer backfed 5 V onto the I2C bus, destroying the board -> diagnosed and migrated to the H743.
- Original sensor plan was a separate VL53L0X rangefinder + ADNS-3080 optical flow; the VL53L0X was destroyed in testing and the ADNS-3080 had an SPI pin-mux conflict with the FC -> consolidated both functions into the MTF-02P over a single UART, reducing wiring complexity.

Development was validated in ArduPilot SITL + Gazebo before field flights.

---

## Known limitations (stated on purpose)

- **Dead-reckoning transit** accumulates error from wind drift and optical-flow velocity error during altitude changes; the SEARCH/SEARCH_YAW states exist to absorb that error at each waypoint.
- Tested in calm outdoor conditions at ~1 m altitude; robustness in wind and at higher altitudes is future work.
- The COMMIT vote trades ~0.3 s of decision latency for classification reliability — the right trade at 0.25 m/s, revisit at higher speeds.

---

## Repository structure

```
AETHER/
├── aether_controller.py          # the full flight controller (this is what flew)
├── README.md
├── Docs/
│   ├── Project_Report_1.docx
│   ├── Project_Report_2.docx
│   └── Media/
│       ├── Flowcharts+Diagrams/
│       ├── 2nd Position Award.jpeg
│       └── Brochure.png
└── Model/
    ├── Model_1/                  # campus landmark detector — training results & curves
    └── Model_2/                  # deployed proxy-target detector
        ├── Weights/
        │   └── best.pt           # deployed YOLOv8-Nano weights (4 classes: box_white/green/red/blue)
        └── Result Images/        # in-flight detection samples
```

---

## Team

| Person | Role |
|---|---|
| **Abdullah Malik** | System architecture, hardware integration & debugging, model training, parameter design, field testing — AI-assisted implementation under his direction |
| **Muhammad Ans** | Simulation (SITL/Gazebo testing) |
| **Faiz Jilani** | Physical assembly and parts sourcing |
| Dr. Babar Zaman | Supervisor |
| Engr. Hamza Naeem | Co-supervisor |

GIK Institute of Engineering Sciences and Technology · Final Year Project · 2026 · (presented under the working title "Intellifly")
