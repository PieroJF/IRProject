# Contributor: Piero Flores López

- **Role:** 3D Mapping and Environment Reconstruction
- **Files:** `controllers/moose_path_following/moose_path_following.c`

## 1. LiDAR Processing Pipeline

To build 3D maps, I implemented a complete point cloud processing pipeline that filters and transforms raw LiDAR data into usable spatial information.

- **Point Filtering:** Removes invalid measurements (NaN/Inf), near-field noise (`MIN_POINT_DISTANCE = 0.3m`), and distant unreliable returns (`MAX_POINT_DISTANCE = 50m`). Ground plane rejection filters points below `GROUND_HEIGHT_THRESH`.

- **Voxel Downsampling:** Reduces point density using a pre-allocated 200³ voxel grid with 0.1m leaf size, improving computational efficiency without repeated memory allocation.

- **EKF Pose Integration:** Transforms points from sensor frame to global frame using the robot's 6-DOF pose estimate, enabling proper spatial registration as the robot moves.

## 2. OctoMap Occupancy Mapping

I implemented an octree-based 3D occupancy grid following the OctoMap framework for memory-efficient volumetric representation.

- **Octree Structure (`octomap_init`, `octomap_insert_point`):** Hierarchical spatial subdivision with maximum depth 12 and 0.1m base resolution, supporting a 40m × 40m × 10m environment.

- **Probabilistic Updates:** Log-odds occupancy model with configurable hit/miss probabilities (`L_HIT = 0.85`, `L_MISS = -0.4`). Values are clamped to [-2.0, 3.5] to prevent saturation and allow map correction.

- **Dynamic Recentering (`octomap_recenter`):** Enables unbounded exploration by shifting the map origin when the robot approaches boundaries.

## 3. ICP Scan Matching

To correct odometric drift, I implemented an Iterative Closest Point algorithm that aligns consecutive LiDAR scans.

- **Point-to-Point ICP (`icp_align`):** Computes optimal rigid transformation between scan pairs using SVD-based estimation. Configured with 15 max iterations, convergence tolerance 0.005, and 1.0m max correspondence distance.

- **Cumulative Correction:** Maintains running transformation estimate applied to subsequent pose predictions.

## 4. Elevation Grid & Traversability

The system extracts 2D terrain information from 3D data for integration with the path planning module.

- **Elevation Grid:** 200 × 200 cell grid at 0.2m resolution aggregating height information. Achieved 81.7% coverage (32,666/40,000 cells) during testing.

- **Slope Computation:** Calculates terrain slope via central differences on the elevation grid.

- **Traversability Cost (`compute_traversability`):** Combines slope and surface roughness into a weighted cost function consumed by Inika's A* planner. Cells exceeding 20° slope are marked as obstacles.

## 5. Map Export & Visualization

I added export capabilities for offline analysis and debugging.

- **PGM Export (`mapping_export_pgm`):** Outputs 2D traversability maps compatible with standard visualization tools.

- **PLY Export (`mapping_export_ply`):** Outputs 3D point clouds for visualization in CloudCompare or MeshLab.

## 6. Random Wandering Navigation

To enable autonomous exploration, I implemented a state machine-based navigation controller.

- **FSM Controller (`random_wander_controller_v2`):** Three-state machine (NORMAL → REVERSING → TURNING) that generates random goals and handles obstacle avoidance.

- **Stuck Detection:** Monitors displacement over 5-second windows. If movement < 0.5m, triggers recovery sequence.

- **Recovery Maneuvers:** Executes 3-second reverse at -2.0 m/s followed by random turn (45°-270°) to escape local minima.

- **Goal Validation (`generate_random_goal`):** Validates random goals against traversability map before acceptance, ensuring reachable destinations.

## 7. Known Issues & Future Work

During testing, the OctoMap showed 0% point acceptance rate while the elevation grid worked correctly (81.7% coverage). Root cause analysis indicates `GROUND_HEIGHT_THRESH = -5.0m` interacts incorrectly with the sensor coordinate frame, causing valid points to be rejected.

**Recommended fixes:**
- Log raw point cloud Z-range before filtering
- Set ground threshold relative to sensor height
- Add per-filter rejection counters for debugging
