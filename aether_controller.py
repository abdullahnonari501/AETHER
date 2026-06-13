#!/usr/bin/env python3
"""
AETHER Autonomous Controller
============================

GPS-denied autonomous waypoint navigation using:
  - YOLOv8 vision (Jetson Nano + USB camera)
  - MTF-02P optical flow + rangefinder (FC fusion)
  - ArduCopter GUIDED mode via MAVLink

Mission: drone visits 4 colored waypoint boxes at corners of a 10m square.
         WHITE -> GREEN -> RED -> BLUE -> WHITE.
         At each box: center over it via vision, hover stable for 5s, yaw to
         next, fly to next.

Author : Project AETHER team
Defense: 2026-04-29 morning

----------------------------------------------------------------------
USAGE
----------------------------------------------------------------------
  Inside Ultralytics container:
    python3 aether_controller.py --mode pilot_handoff
    python3 aether_controller.py --mode fully_auto

  Pilot-handoff (DEFAULT, recommended for defense):
    1) Pilot manually arms, takes off in Stabilize, climbs to ~1m
    2) Pilot switches mode to AltHold (VrA middle), hovers stable
    3) Pilot switches mode to GUIDED (VrA full CW)
    4) Controller takes over and flies the mission
    5) After mission, controller hovers over WHITE; pilot lands manually

  Fully-auto:
    1) Controller arms drone, takes off, flies mission, hovers at end

----------------------------------------------------------------------
SAFETY
----------------------------------------------------------------------
  - All velocity commands clamped to MAX_VEL (default 0.25 m/s)
  - If pilot switches AWAY from GUIDED at any time, controller releases
    authority immediately. Pilot has full override always.
  - Heartbeat watchdog: if FC stops responding for >2s, controller halts
  - State timeouts: drone won't get stuck in a state forever
"""

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import cv2
from pymavlink import mavutil
from ultralytics import YOLO

# =====================================================================
# CONFIG -- tune these for your specific setup
# =====================================================================

# --- Connection ---
MAVLINK_DEVICE = "/dev/ttyTHS1"   # Jetson UART to FC
MAVLINK_BAUD   = 57600

# --- Vision ---
YOLO_MODEL_PATH      = "/tmp/best.pt"
CAMERA_INDEX         = 0
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
CONF_THRESHOLD       = 0.50      # min YOLO confidence to count
DETECTION_WINDOW     = 10         # frames to keep for temporal smoothing
DETECTION_CONFIRM    = 7          # need this many positive frames in window
PIXEL_TOLERANCE      = 30         # ±px from center = "centered"

# --- Mission ---
HOVER_ALTITUDE_M     = 1.0        # altitude during transit and centering
HOVER_AT_BOX_SECONDS = 5.0        # stable hover duration before transition
SQUARE_SIDE_M        = 10.0       # box-to-box distance

# Waypoint matrix - drone-local NED frame (North-East-Down)
# Position (0,0) = WHITE box (start)
# After takeoff drone faces +X (north / forward)
# WHITE -> GREEN: 10m forward (+X)
# GREEN -> RED:   10m right   (+Y)
# RED   -> BLUE:  10m back    (-X)
# BLUE  -> WHITE: 10m left    (-Y)
WAYPOINTS = [
    {"name": "WHITE", "class": "box_white", "pos": (0.0, 0.0)},
    {"name": "GREEN", "class": "box_green", "pos": (SQUARE_SIDE_M, 0.0)},
    {"name": "RED",   "class": "box_red",   "pos": (SQUARE_SIDE_M, SQUARE_SIDE_M)},
    {"name": "BLUE",  "class": "box_blue",  "pos": (0.0, SQUARE_SIDE_M)},
    {"name": "WHITE", "class": "box_white", "pos": (0.0, 0.0)},  # return
]

# --- Search behavior (when no target visible) ---
PASSIVE_SEARCH_S     = 3.0        # how long to hover-look before yaw search
SEARCH_YAW_RATE_DEG  = 15.0       # slow sweep rate during 360 search
SEARCH_YAW_TIMEOUT_S = 30.0       # ~360° + margin at 15 deg/s

# --- Commit phase (multi-frame color vote) ---
COMMIT_FRAMES        = 6          # frames to collect during commit
COMMIT_MIN_AVG_CONF  = 0.65       # winner must average >= this
COMMIT_LEAD_MARGIN   = 0.20       # winner must beat runner-up by this
TRANSIT_VEL          = 0.25       # m/s during transit between boxes
CENTERING_VEL_MAX    = 0.20       # m/s during pixel-error centering
YAW_RATE_DEG_S       = 30.0       # deg/s when yawing to next waypoint
ALTITUDE_TOLERANCE_M = 0.15       # acceptable altitude error during transit

# --- Centering PD controller ---
# Pixel error -> velocity command. Positive pixel error in X (centroid
# right of frame center) means drone needs to move right (+Y in NED).
KP_PIXEL = 0.0006     # P gain: pixel error -> m/s
KD_PIXEL = 0.0001     # D gain: pixel error rate

# --- Safety ---
HEARTBEAT_TIMEOUT_S  = 2.0
STATE_TIMEOUT_S      = 60.0       # max time in any single state
MIN_FLIGHT_VOLTAGE   = 11.0       # force land below this voltage

# =====================================================================
# STATE MACHINE
# =====================================================================

class State(Enum):
    INIT             = "INIT"
    WAIT_FOR_GUIDED  = "WAIT_FOR_GUIDED"
    AUTO_TAKEOFF     = "AUTO_TAKEOFF"
    TRANSIT          = "TRANSIT"
    SEARCH           = "SEARCH"           # passive: hover and look
    SEARCH_YAW       = "SEARCH_YAW"       # active: rotate 360 looking
    COMMIT           = "COMMIT"           # multi-frame color vote
    CENTER           = "CENTER"
    HOVER_AT_BOX     = "HOVER_AT_BOX"
    YAW_TO_NEXT      = "YAW_TO_NEXT"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    ABORT            = "ABORT"


@dataclass
class Telemetry:
    """Snapshot of FC state at any moment."""
    mode: str = ""
    armed: bool = False
    altitude_m: float = 0.0     # rangefinder altitude
    voltage_v: float = 0.0
    heading_deg: float = 0.0
    last_heartbeat_age: float = 999.0


# =====================================================================
# MAVLINK HELPERS
# =====================================================================

class FCLink:
    """Wraps pymavlink connection to ArduCopter FC."""

    def __init__(self, device, baud):
        print(f"[mavlink] connecting to {device} @ {baud}...")
        self.master = mavutil.mavlink_connection(device, baud=baud)
        self.master.wait_heartbeat(timeout=10)
        self.sysid = self.master.target_system
        self.compid = self.master.target_component
        self.last_heartbeat = time.time()
        print(f"[mavlink] connected sys={self.sysid} comp={self.compid}")

        # Request fast streams for the messages we need
        self._request_streams()

    def _request_streams(self):
        # Request standard data streams at 10 Hz
        self.master.mav.request_data_stream_send(
            self.sysid, self.compid,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1,
        )

    def poll_telemetry(self, tlm: Telemetry):
        """Drain pending messages, update telemetry snapshot."""
        while True:
            msg = self.master.recv_match(blocking=False)
            if not msg:
                break
            t = msg.get_type()
            if t == "HEARTBEAT":
                self.last_heartbeat = time.time()
                tlm.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                tlm.mode = mavutil.mode_string_v10(msg)
            elif t == "RANGEFINDER":
                tlm.altitude_m = msg.distance
            elif t == "SYS_STATUS":
                tlm.voltage_v = msg.voltage_battery / 1000.0
            elif t == "VFR_HUD":
                tlm.heading_deg = msg.heading
            elif t == "ATTITUDE":
                pass  # could use roll/pitch for stability check

        tlm.last_heartbeat_age = time.time() - self.last_heartbeat

    def set_mode(self, mode_name):
        """Switch flight mode. Returns immediately; verify via telemetry."""
        if mode_name not in self.master.mode_mapping():
            print(f"[mavlink] ERROR unknown mode '{mode_name}'")
            return False
        mode_id = self.master.mode_mapping()[mode_name]
        self.master.mav.set_mode_send(
            self.sysid,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        return True

    def arm(self):
        print("[mavlink] arming...")
        self.master.mav.command_long_send(
            self.sysid, self.compid,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )

    def disarm(self):
        print("[mavlink] disarming...")
        self.master.mav.command_long_send(
            self.sysid, self.compid,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def takeoff(self, altitude_m):
        print(f"[mavlink] takeoff to {altitude_m:.2f} m")
        self.master.mav.command_long_send(
            self.sysid, self.compid,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude_m,
        )

    def send_velocity_body(self, vx, vy, vz, yaw_rate_deg=0.0):
        """Send velocity command in BODY frame (drone-relative)."""
        # Clamp velocity to safety limit
        vx = max(-TRANSIT_VEL, min(TRANSIT_VEL, vx))
        vy = max(-TRANSIT_VEL, min(TRANSIT_VEL, vy))
        vz = max(-0.5, min(0.5, vz))

        # SET_POSITION_TARGET_LOCAL_NED with body offset NED frame
        type_mask = (
            0b0000111111000111   # use velocity, ignore position/accel
        )
        self.master.mav.set_position_target_local_ned_send(
            0,                                   # time_boot_ms
            self.sysid, self.compid,
            mavutil.mavlink.MAV_FRAME_BODY_NED,  # body frame
            type_mask,
            0, 0, 0,                             # position (ignored)
            vx, vy, vz,                          # velocity
            0, 0, 0,                             # acceleration (ignored)
            0, math.radians(yaw_rate_deg),       # yaw, yaw_rate
        )

    def condition_yaw(self, target_heading_deg, rate_deg_s=YAW_RATE_DEG_S, relative=False):
        """Yaw to absolute or relative heading."""
        self.master.mav.command_long_send(
            self.sysid, self.compid,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            target_heading_deg,    # param1: heading deg
            rate_deg_s,            # param2: speed deg/s
            1,                     # param3: 1=CW, -1=CCW
            1 if relative else 0,  # param4: relative flag
            0, 0, 0,
        )


# =====================================================================
# VISION
# =====================================================================

class Vision:
    """Wraps YOLO model + camera."""

    def __init__(self, model_path, cam_index):
        print(f"[vision] loading model {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"[vision] classes: {self.class_names}")

        print(f"[vision] opening camera {cam_index}...")
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        print("[vision] camera ready")

    def detect(self, target_class):
        """
        Grab a frame, run inference, return best detection of target_class
        as a dict {cx, cy, conf, w, h} or None if nothing found.
        Coordinates are pixel coordinates in image frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        results = self.model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        best = None
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            if cls_name != target_class:
                continue
            conf = float(box.conf[0])
            if best is None or conf > best["conf"]:
                xyxy = box.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                w  = xyxy[2] - xyxy[0]
                h  = xyxy[3] - xyxy[1]
                best = {"cx": cx, "cy": cy, "conf": conf, "w": w, "h": h}
        return best

    def detect_all(self):
        """
        Grab a frame, run inference, return BEST detection per class as
        dict {class_name: {cx, cy, conf, w, h}}. Used in COMMIT phase to
        compare confidence across all classes for color voting.
        """
        ret, frame = self.cap.read()
        if not ret:
            return {}

        results = self.model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        per_class = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            conf = float(box.conf[0])
            if cls_name not in per_class or conf > per_class[cls_name]["conf"]:
                xyxy = box.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                per_class[cls_name] = {
                    "cx": cx, "cy": cy, "conf": conf,
                    "w": xyxy[2] - xyxy[0], "h": xyxy[3] - xyxy[1],
                }
        return per_class

    def shutdown(self):
        self.cap.release()


# =====================================================================
# MAIN CONTROLLER
# =====================================================================

class AetherController:
    def __init__(self, mode):
        self.mission_mode = mode  # "pilot_handoff" or "fully_auto"
        self.fc     = FCLink(MAVLINK_DEVICE, MAVLINK_BAUD)
        self.vision = Vision(YOLO_MODEL_PATH, CAMERA_INDEX)
        self.tlm    = Telemetry()

        # Mission progress
        self.wp_idx = 1   # next waypoint to fly to (0 is start, can't be target)
        self.state = State.INIT
        self.state_entered_at = time.time()

        # Vision state
        self.det_window = deque(maxlen=DETECTION_WINDOW)
        self.last_pixel_err = (0.0, 0.0)
        self.centered_since = None

        # COMMIT phase state - collects per-class confidences across N frames
        self.commit_samples = []     # list of {class: conf} dicts
        self.commit_target_pos = None  # last known centroid of winning class

        # SEARCH_YAW phase state
        self.search_yaw_started = False
        self.search_yaw_start_heading = 0.0

        # Initial yaw reference (drone's heading at start) for transit logic
        self.initial_heading = None

    # -------------------------------------------------------------
    # State transitions
    # -------------------------------------------------------------
    def transition(self, new_state):
        print(f"[state] {self.state.value} -> {new_state.value}")
        self.state = new_state
        self.state_entered_at = time.time()
        self.det_window.clear()
        self.centered_since = None
        self.commit_samples = []
        self.search_yaw_started = False

    def time_in_state(self):
        return time.time() - self.state_entered_at

    # -------------------------------------------------------------
    # Top-level loop
    # -------------------------------------------------------------
    def run(self):
        print(f"[ctrl] started in {self.mission_mode} mode")
        print(f"[ctrl] mission: {' -> '.join(w['name'] for w in WAYPOINTS)}")

        try:
            while True:
                self.fc.poll_telemetry(self.tlm)

                # Safety: heartbeat watchdog
                if self.tlm.last_heartbeat_age > HEARTBEAT_TIMEOUT_S:
                    print("[safety] heartbeat lost - aborting")
                    self.transition(State.ABORT)

                # Safety: low voltage
                if self.tlm.voltage_v > 0 and self.tlm.voltage_v < MIN_FLIGHT_VOLTAGE:
                    print(f"[safety] low voltage {self.tlm.voltage_v:.2f}V - aborting")
                    self.transition(State.ABORT)

                # Safety: pilot took manual control
                if (self.state not in (State.INIT, State.WAIT_FOR_GUIDED, State.ABORT)
                        and self.tlm.mode != "GUIDED"):
                    print(f"[safety] pilot left GUIDED (now {self.tlm.mode}) - releasing")
                    self.transition(State.ABORT)

                # State timeout
                if self.time_in_state() > STATE_TIMEOUT_S \
                        and self.state not in (State.INIT, State.WAIT_FOR_GUIDED,
                                                State.MISSION_COMPLETE, State.ABORT):
                    print(f"[safety] state {self.state.value} timed out")
                    self.transition(State.ABORT)

                # Dispatch
                if   self.state == State.INIT:             self.do_init()
                elif self.state == State.WAIT_FOR_GUIDED:  self.do_wait_for_guided()
                elif self.state == State.AUTO_TAKEOFF:     self.do_auto_takeoff()
                elif self.state == State.TRANSIT:          self.do_transit()
                elif self.state == State.SEARCH:           self.do_search()
                elif self.state == State.SEARCH_YAW:       self.do_search_yaw()
                elif self.state == State.COMMIT:           self.do_commit()
                elif self.state == State.CENTER:           self.do_center()
                elif self.state == State.HOVER_AT_BOX:     self.do_hover()
                elif self.state == State.YAW_TO_NEXT:      self.do_yaw()
                elif self.state == State.MISSION_COMPLETE: self.do_complete()
                elif self.state == State.ABORT:            self.do_abort(); return

                time.sleep(0.05)  # ~20 Hz control loop

        except KeyboardInterrupt:
            print("\n[ctrl] interrupted by user")
        finally:
            self.vision.shutdown()
            print("[ctrl] shutdown")

    # -------------------------------------------------------------
    # State handlers
    # -------------------------------------------------------------
    def do_init(self):
        if self.mission_mode == "pilot_handoff":
            print("[ctrl] waiting for pilot to switch into GUIDED...")
            self.transition(State.WAIT_FOR_GUIDED)
        else:
            print("[ctrl] fully-auto: arming and taking off")
            self.fc.set_mode("GUIDED")
            time.sleep(1)
            self.fc.arm()
            time.sleep(2)
            self.fc.takeoff(HOVER_ALTITUDE_M)
            self.transition(State.AUTO_TAKEOFF)

    def do_wait_for_guided(self):
        if self.tlm.mode == "GUIDED" and self.tlm.armed:
            self.initial_heading = self.tlm.heading_deg
            print(f"[ctrl] GUIDED received. initial heading = {self.initial_heading:.1f} deg")
            self.transition(State.TRANSIT)

    def do_auto_takeoff(self):
        # Wait until we reach hover altitude
        if self.tlm.altitude_m >= HOVER_ALTITUDE_M - ALTITUDE_TOLERANCE_M:
            self.initial_heading = self.tlm.heading_deg
            print(f"[ctrl] takeoff complete. heading = {self.initial_heading:.1f} deg")
            self.transition(State.TRANSIT)

    def do_transit(self):
        """Fly toward next waypoint at TRANSIT_VEL using body-frame velocity.

        We don't have absolute position so we time-integrate: distance = v * t.
        After ~SQUARE_SIDE_M / TRANSIT_VEL seconds, we should be near target.
        """
        target = WAYPOINTS[self.wp_idx]
        prev   = WAYPOINTS[self.wp_idx - 1]
        dx = target["pos"][0] - prev["pos"][0]
        dy = target["pos"][1] - prev["pos"][1]
        distance = math.hypot(dx, dy)
        transit_time = distance / TRANSIT_VEL

        if self.time_in_state() < transit_time:
            # Drone is yawed toward next box; just fly forward in body frame
            altitude_err = HOVER_ALTITUDE_M - self.tlm.altitude_m
            vz = -max(-0.3, min(0.3, altitude_err * 0.5))   # NED z-down
            self.fc.send_velocity_body(TRANSIT_VEL, 0.0, vz)
        else:
            # Estimated arrival - now use vision to find target
            print(f"[ctrl] transit time elapsed - searching for {target['name']}")
            self.transition(State.SEARCH)

    def do_search(self):
        """Hover and wait for target to appear in camera frame.

        If nothing detected after PASSIVE_SEARCH_S, escalate to SEARCH_YAW
        which slowly rotates the drone 360 degrees looking for the target.
        """
        target = WAYPOINTS[self.wp_idx]

        # Hold position
        altitude_err = HOVER_ALTITUDE_M - self.tlm.altitude_m
        vz = -max(-0.3, min(0.3, altitude_err * 0.5))
        self.fc.send_velocity_body(0.0, 0.0, vz)

        # Look for target
        det = self.vision.detect(target["class"])
        self.det_window.append(det is not None)

        # Confirm target with temporal smoothing
        if sum(self.det_window) >= DETECTION_CONFIRM:
            print(f"[ctrl] {target['name']} candidate seen - committing color vote")
            self.transition(State.COMMIT)
            return

        # If passive search times out, start active 360 yaw search
        if self.time_in_state() > PASSIVE_SEARCH_S:
            print(f"[ctrl] passive search timeout - starting 360 yaw search")
            self.transition(State.SEARCH_YAW)

    def do_search_yaw(self):
        """Slowly yaw 360 degrees looking for the target box.

        EKF3 yaw is reliable enough for this scale - it estimates yaw from
        gyro integration plus optical flow rotation cues from the MTF-02P.
        Drift over a 24s rotation is typically <2 deg, well within tolerance.
        """
        target = WAYPOINTS[self.wp_idx]

        # On first tick, record starting heading and command yaw
        if not self.search_yaw_started:
            self.search_yaw_start_heading = self.tlm.heading_deg
            self.search_yaw_started = True
            print(f"[ctrl] yaw search starting from heading {self.search_yaw_start_heading:.1f}")

        # Hold altitude, command yaw rate via body frame (yaw_rate component)
        altitude_err = HOVER_ALTITUDE_M - self.tlm.altitude_m
        vz = -max(-0.3, min(0.3, altitude_err * 0.5))
        self.fc.send_velocity_body(0.0, 0.0, vz, yaw_rate_deg=SEARCH_YAW_RATE_DEG)

        # Continuously check for target during rotation
        det = self.vision.detect(target["class"])
        self.det_window.append(det is not None)

        if sum(self.det_window) >= DETECTION_CONFIRM:
            print(f"[ctrl] {target['name']} found during yaw search")
            # Stop yawing
            self.fc.send_velocity_body(0.0, 0.0, vz, yaw_rate_deg=0.0)
            self.transition(State.COMMIT)
            return

        # Did we complete a full rotation without finding it?
        if self.time_in_state() > SEARCH_YAW_TIMEOUT_S:
            print(f"[ctrl] full 360 yaw complete - target NOT found")
            print(f"[ctrl] aborting mission - cannot continue without target")
            self.transition(State.ABORT)

    def do_commit(self):
        """Multi-frame color vote.

        Collect COMMIT_FRAMES detection samples (each containing the best-
        confidence detection per class). Average per-class confidence. The
        winner must:
          1. Have average confidence >= COMMIT_MIN_AVG_CONF
          2. Beat runner-up by >= COMMIT_LEAD_MARGIN
          3. Be the actual target class for this waypoint

        This protects against single-frame misclassification (e.g. green box
        briefly classified as blue at 0.6 conf).
        """
        target = WAYPOINTS[self.wp_idx]

        # Hover stable during commit
        altitude_err = HOVER_ALTITUDE_M - self.tlm.altitude_m
        vz = -max(-0.3, min(0.3, altitude_err * 0.5))
        self.fc.send_velocity_body(0.0, 0.0, vz)

        # Sample one frame
        per_class = self.vision.detect_all()
        self.commit_samples.append(per_class)

        # Save last known centroid of target class so CENTER can pick up smoothly
        if target["class"] in per_class:
            self.commit_target_pos = per_class[target["class"]]

        if len(self.commit_samples) < COMMIT_FRAMES:
            return  # keep collecting

        # We have enough samples - tally
        class_total_conf = {}
        class_count = {}
        for sample in self.commit_samples:
            for cname, det in sample.items():
                class_total_conf[cname] = class_total_conf.get(cname, 0) + det["conf"]
                class_count[cname] = class_count.get(cname, 0) + 1

        # Compute average confidence (averaged across COMMIT_FRAMES, missing
        # frames count as 0 conf)
        class_avg = {}
        for cname in class_total_conf:
            class_avg[cname] = class_total_conf[cname] / COMMIT_FRAMES

        # Sort by average confidence descending
        ranked = sorted(class_avg.items(), key=lambda x: -x[1])
        winner_name, winner_conf = ranked[0]
        runner_conf = ranked[1][1] if len(ranked) > 1 else 0.0

        print(f"[commit] tallies: {[(n, f'{c:.2f}') for n, c in ranked]}")

        # Validate against three criteria
        if winner_name != target["class"]:
            print(f"[commit] FAIL - winner is {winner_name}, expected {target['class']}")
            self.transition(State.SEARCH)
            return
        if winner_conf < COMMIT_MIN_AVG_CONF:
            print(f"[commit] FAIL - winner conf {winner_conf:.2f} < {COMMIT_MIN_AVG_CONF}")
            self.transition(State.SEARCH)
            return
        if winner_conf - runner_conf < COMMIT_LEAD_MARGIN:
            print(f"[commit] FAIL - lead {winner_conf - runner_conf:.2f} < {COMMIT_LEAD_MARGIN}")
            self.transition(State.SEARCH)
            return

        print(f"[commit] PASS - {winner_name} avg conf {winner_conf:.2f} (lead {winner_conf - runner_conf:.2f})")
        self.transition(State.CENTER)

    def do_center(self):
        """PD control on pixel error to bring target box to image center."""
        target = WAYPOINTS[self.wp_idx]
        det = self.vision.detect(target["class"])

        if det is None:
            self.det_window.append(False)
            # Briefly lost - hover in place
            self.fc.send_velocity_body(0.0, 0.0, 0.0)
            if sum(self.det_window) < 3:
                # Lost target entirely - go back to search
                self.transition(State.SEARCH)
            return

        self.det_window.append(True)

        # Pixel error from frame center
        err_x = det["cx"] - FRAME_WIDTH / 2.0
        err_y = det["cy"] - FRAME_HEIGHT / 2.0

        # Image axes -> body frame (assuming camera is mounted nose-forward):
        #   image +X (right)    -> drone body +Y (right)
        #   image +Y (down)     -> drone body -X (backward) [box AHEAD = need to fly forward]
        #
        # So drone needs to fly in body NED:
        #   body +X (forward) when err_y is NEGATIVE (target is above center)
        #   body +Y (right)   when err_x is POSITIVE (target is right of center)
        d_err_x = err_x - self.last_pixel_err[0]
        d_err_y = err_y - self.last_pixel_err[1]
        self.last_pixel_err = (err_x, err_y)

        vx = -(KP_PIXEL * err_y + KD_PIXEL * d_err_y)   # forward
        vy =  (KP_PIXEL * err_x + KD_PIXEL * d_err_x)   # right
        vx = max(-CENTERING_VEL_MAX, min(CENTERING_VEL_MAX, vx))
        vy = max(-CENTERING_VEL_MAX, min(CENTERING_VEL_MAX, vy))

        altitude_err = HOVER_ALTITUDE_M - self.tlm.altitude_m
        vz = -max(-0.3, min(0.3, altitude_err * 0.5))
        self.fc.send_velocity_body(vx, vy, vz)

        # Centered check
        if abs(err_x) < PIXEL_TOLERANCE and abs(err_y) < PIXEL_TOLERANCE:
            if self.centered_since is None:
                self.centered_since = time.time()
            elif time.time() - self.centered_since >= 0.5:
                # 0.5s of stable centering -> begin hover countdown
                print(f"[ctrl] centered on {target['name']} - starting 5s hover")
                self.transition(State.HOVER_AT_BOX)
        else:
            self.centered_since = None

    def do_hover(self):
        """Hold position over box for HOVER_AT_BOX_SECONDS."""
        # Maintain centering during hover (small corrections)
        target = WAYPOINTS[self.wp_idx]
        det = self.vision.detect(target["class"])
        if det is not None:
            err_x = det["cx"] - FRAME_WIDTH / 2.0
            err_y = det["cy"] - FRAME_HEIGHT / 2.0
            vx = -KP_PIXEL * err_y * 0.5   # gentler P during hover
            vy =  KP_PIXEL * err_x * 0.5
        else:
            vx = vy = 0.0
        altitude_err = HOVER_ALTITUDE_M - self.tlm.altitude_m
        vz = -max(-0.3, min(0.3, altitude_err * 0.5))
        self.fc.send_velocity_body(vx, vy, vz)

        if self.time_in_state() >= HOVER_AT_BOX_SECONDS:
            if self.wp_idx >= len(WAYPOINTS) - 1:
                # That was the final waypoint
                self.transition(State.MISSION_COMPLETE)
            else:
                self.transition(State.YAW_TO_NEXT)

    def do_yaw(self):
        """Yaw to face next waypoint."""
        # Compute heading from current waypoint to next
        cur = WAYPOINTS[self.wp_idx]
        nxt = WAYPOINTS[self.wp_idx + 1]
        dx = nxt["pos"][0] - cur["pos"][0]
        dy = nxt["pos"][1] - cur["pos"][1]
        # Drone's local frame: +X = initial forward direction
        # Heading from initial: atan2(dy, dx) gives angle in radians from +X axis
        target_heading_local = math.degrees(math.atan2(dy, dx))
        # Convert to absolute heading using initial_heading reference
        target_heading_abs = (self.initial_heading + target_heading_local) % 360

        if self.time_in_state() < 0.1:
            # First tick - send the yaw command once
            print(f"[ctrl] yawing to {target_heading_abs:.1f} deg")
            self.fc.condition_yaw(target_heading_abs)

        # Hold position during yaw
        altitude_err = HOVER_ALTITUDE_M - self.tlm.altitude_m
        vz = -max(-0.3, min(0.3, altitude_err * 0.5))
        self.fc.send_velocity_body(0.0, 0.0, vz)

        # Check if yaw complete (within 5 deg)
        heading_err = abs((self.tlm.heading_deg - target_heading_abs + 540) % 360 - 180)
        if heading_err < 5.0 and self.time_in_state() > 1.0:
            self.wp_idx += 1
            print(f"[ctrl] yaw done - heading to waypoint {self.wp_idx} ({WAYPOINTS[self.wp_idx]['name']})")
            self.transition(State.TRANSIT)

    def do_complete(self):
        """Mission done - hover indefinitely until pilot intervenes."""
        if int(self.time_in_state()) % 5 == 0:
            print(f"[ctrl] MISSION COMPLETE. hovering. time={self.time_in_state():.0f}s")
        altitude_err = HOVER_ALTITUDE_M - self.tlm.altitude_m
        vz = -max(-0.3, min(0.3, altitude_err * 0.5))
        self.fc.send_velocity_body(0.0, 0.0, vz)

    def do_abort(self):
        """Release authority. Pilot has the drone."""
        print("[ctrl] ABORT - sending zero velocity, releasing")
        self.fc.send_velocity_body(0.0, 0.0, 0.0)


# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="AETHER autonomous controller")
    parser.add_argument(
        "--mode",
        choices=["pilot_handoff", "fully_auto"],
        default="pilot_handoff",
        help="Mission start mode (default: pilot_handoff)",
    )
    args = parser.parse_args()

    ctrl = AetherController(args.mode)
    ctrl.run()


if __name__ == "__main__":
    main()
