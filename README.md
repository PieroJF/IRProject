# Autonomous Navigation and 3D Mapping for Grass-Cutting Robots on Inclined Terrains

## Project Overview
This project enhances the autonomy of a grass-cutting robot (Clearpath Moose) to operate on uneven, hilly terrain. The system addresses slope hazards through three core software components:
1. **Extended Kalman Filter (EKF):** Fuses wheel odometry with compass data for accurate localization.
2. **3D LiDAR Mapping:** Builds an OctoMap-based occupancy grid to represent complex terrain.
3. **Slope-Aware Navigation:** Uses an A* planner with a custom cost function to identify and avoid hazardous gradients (>20°) while maintaining coverage.

---

### Prerequisites
* **Webots R2022a**

---

# System Architecture

## 1. Kinematics & Low-Level Control
* **Modified Hardware Interface:** Custom `Moose.proto` with 8-wheel PositionSensors enabled closed-loop odometry.
* **Differential Drive Logic:** Converts raw encoder ticks into metric state estimates (x, y, θ) using skid-steering kinematics.

## 2. Sensor Fusion (EKF)
* **State Prediction:** Propagates system uncertainty (Covariance Matrix P) based on motion model.
* **Update Step:** Fuses magnetometer (compass) yaw with the prediction to eliminate odometric drift.
* **Auto-Calibration:** Solves coordinate frame mismatch by aligning odometry θ with the initial compass reading.

## 3. 3D Mapping (OctoMap)
* **LiDAR Pipeline:** Filters point clouds (0.3m < d < 50m) and downsamples using a 2003 voxel grid.
* **Occupancy Grid:** Probabilistic log-odds update model with dynamic map recentring for unbounded exploration.
* **Traversability Analysis:** Converts 3D data into a 2D cost map, penalizing surface roughness and slopes.

## 4. Navigation & Planning
* **Slope-Aware A*:** Global planner that treats gradients > 20° as obstacles.
* **Hazard Detection:** High-frequency (32ms) safety layer that monitors IMU Roll/Pitch to prevent tipping.
* **Stuck Recovery:** Monitors displacement over 5s windows; triggers reversing manoeuvres if the robot is physically stuck.

---

# Authorship & Contributions

This project was a collaborative effort. The specific contributions for assessment purposes are listed below.

### Yichang Chao
**Role:** Robot Modelling, Low-level Control & Kinematics
**File:** `protos/Moose.proto`, `controllers/moose_path_following/moose_path_following.c`
**Key Contributions:**
* Modified `Moose.proto` to add 8-wheel feedback sensors.
* Implemented low-level C driver for motor/sensor interfacing.
* Developed Differential Drive Odometry and EKF Prediction step.

### Ran Zhang
**Role:** Sensor Fusion, Calibration & Validation
**File:** `controllers/moose_path_following/moose_path_following.c`
**Key Contributions:**
* Implemented EKF Update step (Compass fusion).
* Developed the "Auto-Alignment" routine to fix coordinate frame inversion.
* Built the GPS Ground Truth validation system.

### Piero Flores López
**Role:** 3D Mapping & Environment Reconstruction
**File:** `controllers/moose_path_following/moose_path_following.c`
**Key Contributions:**
* Built the LiDAR processing pipeline (Filtering, Voxel Grid, Transform).
* Implemented OctoMap-style occupancy grid and ICP Scan Matching.
* Developed the Traversability Cost function (Slope + Roughness).

### Inika Kumar
**Role:** Path Planning, Hazard Detection & Evaluation
**File:** `controllers/moose_path_following/moose_path_following.c`
**Key Contributions:**
* Implemented the slope-aware A* global planner.
* Developed the IMU-based Hazard Detection safety layer.
* Integrated the Navigation Controller (Path Following + Performance Metrics).
