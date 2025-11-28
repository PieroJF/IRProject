# Contributor: Inika Kumar
Role: Path Planning, Hazard Detection & Evaluation (Person D)

## 1. Hardware Integration & Robustness
**Files:** `controllers/moose_path_following/moose_path_following.c`

To enable slope detection, I integrated the robot's Inertial Measurement Unit (IMU). The initial controller crashed because it could not locate the device by its default name.

Robust Device Discovery: Instead of hardcoding the device name (which caused runtime errors), I implemented a failsafe initialization routine. The controller now iterates through the robot's device list to identify the IMU dynamically by its internal Node Type (`WB_NODE_INERTIAL_UNIT`). This ensures the code is portable and robust even if the robot's specific sensor names change.

## 2. Hazard Detection Algorithm
File: `controllers/moose_path_following/moose_path_following.c`

I implemented the safety layer that runs at high frequency (32ms) to prevent the robot from tipping over on uneven terrain.

Logic (`is_hazardous_state`): Monitors the robot's Roll and Pitch in real-time.
Thresholding: If the tilt exceeds `0.35` rad (~20°), the system triggers an emergency stop by overriding motor commands to 0.
Log Throttling: Implemented a static counter to throttle warning logs to 1Hz, keeping the console clean for EKF data while maintaining 32ms safety checks.

## 3. Slope-Aware Navigation & Cost Mapping
File: `controllers/moose_path_following/moose_path_following.c`

I developed the navigation logic that integrates the EKF Pose (from Person B) to make autonomous driving decisions.

Mock Cost Map (`get_mock_map_cost`): Since the 3D Mapping module (Person C) is in progress, I implemented a virtual cost layer that simulates a steep "Hill Obstacle" at coordinates (5.0, 2.0).
Reactive Planner (`calculate_navigation`):
    Lookahead: Projects the robot's EKF position 2.0 meters forward based on current heading ($\theta$).
    Avoidance: If the projected point hits a High Cost area (>50), the planner overrides the path to execute an evasive turn.
    Goal Seeking: Uses a P-Controller to steer the robot toward the target $(10.0, 2.0)$ when the path is clear.

## 4. Evaluation & Output
The controller now outputs specific navigation states to verify decision making:

Pos: (2.59, 3.14) -> Goal: (10.00, 2.00) | Map Cost Ahead: 0
[Nav] High Slope Detected ahead! Turning Left to avoid.
!!! DANGER: Tipping Hazard Detected! Roll: 0.07, Pitch: -0.35 !!!

Need to complete an integration step to merge Piero's map.