/*
 * ============================================================================
 * Moose Robot - 3D Navigation and Mapping System
 * Extended Kalman Filter (6 DOF) + Velodyne LiDAR + 3D Occupancy Mapping
 * 
 * IMPROVED VERSION - Piero Flores López 3D Mapping Module Integrated
 * 
 * New Features:
 * - OctoMap-style 3D Occupancy Grid (Octree-based)
 * - ICP Scan Matching for drift correction (SLAM-like)
 * - Traversability cost map derived from real sensor data
 * - Roughness and slope computation
 * - Map export to PGM/PLY for visualization
 * ============================================================================
 */

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/compass.h>
#include <webots/position_sensor.h>
#include <webots/gps.h>
#include <webots/inertial_unit.h>
#include <webots/lidar.h>
#include <webots/device.h>
#include <webots/nodes.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ============================================================================
// CONFIGURATION PARAMETERS
// ============================================================================

#define TIME_STEP 32
#define STATE_SIZE 6  // [x, y, z, roll, pitch, yaw]

// Moose wheel parameters
#define WHEEL_RADIUS 0.32 
#define WHEEL_BASE   1.1

// Hazard Thresholds (Radians) - ~20 degrees
#define MAX_SAFE_PITCH 0.35 
#define MAX_SAFE_ROLL  0.35

// ============================================================================
// 3D MAPPING PARAMETERS (Piero Flores López)
// ============================================================================

// OctoMap parameters
#define OCTOMAP_RESOLUTION 0.1      // 10cm voxel resolution
#define OCTOMAP_MAX_DEPTH 12        // Maximum octree depth
#define OCTOMAP_SIZE 40.0           // 40m x 40m x 10m map

// Occupancy probabilities (log-odds)
#define PROB_HIT_LOG 0.85f          // Log-odds for hit
#define PROB_MISS_LOG -0.4f         // Log-odds for miss  
#define PROB_THRESH_MIN -2.0f       // Clamping minimum
#define PROB_THRESH_MAX 3.5f        // Clamping maximum
#define PROB_OCCUPIED_THRESH 0.5f   // Threshold for occupied

// Elevation grid parameters
#define MAP_RESOLUTION 0.2          // 20cm per cell
#define MAP_WIDTH 200               // 40m x 40m
#define MAP_HEIGHT 200

// Point cloud filtering
#define MIN_POINT_DISTANCE 0.2      // Minimum distance filter
#define MAX_POINT_DISTANCE 20.0     // Maximum distance filter
#define GROUND_HEIGHT_THRESH -0.5   // Ground plane threshold
#define VOXEL_LEAF_SIZE 0.1         // Voxel grid filter size

// ICP parameters
#define ICP_MAX_ITERATIONS 15
#define ICP_TOLERANCE 0.005
#define ICP_MAX_CORRESPONDENCE_DIST 1.0

// Downsampling grid
#define VOXEL_GRID_DIM 200

// A* Path Planning
#define PATH_MAP_SIZE 200   // grid size for path planning
#define PATH_MAP_RESOLUTION 0.2  // 0.2m per cell
#define MAX_PATH_LENGTH 500    // maximum waypoints in a path

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// 3D Point structure
typedef struct {
    float x, y, z;
} Point3D;

// Point cloud container
typedef struct {
    Point3D* points;
    int count;
    int capacity;
} PointCloud;

// Octree node for OctoMap
typedef struct OctreeNode {
    float occupancy_log;            // Log-odds occupancy probability
    float elevation_sum;            // For elevation mapping
    int hit_count;                  // Number of observations
    struct OctreeNode* children[8]; // 8 children for octree
    unsigned char is_leaf;          // Leaf node flag
    unsigned char depth;            // Node depth in tree
} OctreeNode;

// OctoMap structure (3D Occupancy Grid)
typedef struct {
    OctreeNode* root;
    double resolution;
    double origin_x, origin_y, origin_z;
    double size;
    int max_depth;
    int total_nodes;
    int leaf_nodes;
    int occupied_nodes;
} OctoMap;

// 2D Elevation Grid with Traversability
typedef struct {
    float elevation[MAP_WIDTH][MAP_HEIGHT];
    float slope[MAP_WIDTH][MAP_HEIGHT];
    float roughness[MAP_WIDTH][MAP_HEIGHT];
    unsigned char traversability[MAP_WIDTH][MAP_HEIGHT];  // 0=free, 255=obstacle, 128=unknown
    int hit_count[MAP_WIDTH][MAP_HEIGHT];
} ElevationGrid;

// 4x4 Transformation matrix
typedef struct {
    double m[4][4];
} Transform3D;

// ICP result
typedef struct {
    Transform3D transform;
    double fitness_score;
    int converged;
    int iterations;
} ICPResult;

// Odometry data
typedef struct {
    double x, y, z;
    double roll, pitch, yaw;
    double prev_left_enc;
    double prev_right_enc;
} OdometryData;

// Extended Kalman Filter
typedef struct {
    double state[STATE_SIZE];
    double covariance[STATE_SIZE][STATE_SIZE];
    double Q[STATE_SIZE][STATE_SIZE];
    double R[STATE_SIZE][STATE_SIZE];
} EKF;

// Sensor readings
typedef struct {
    double compass_yaw;    
    double gps_x, gps_y, gps_z;
    double roll, pitch;
    double left_pos[4];
    double right_pos[4];
} SensorData;

// Robot devices
typedef struct {
    WbDeviceTag left_sensors[4];  
    WbDeviceTag right_sensors[4];
    WbDeviceTag left_motors[4];
    WbDeviceTag right_motors[4];
    WbDeviceTag compass;
    WbDeviceTag gps;
    WbDeviceTag imu;
    WbDeviceTag velodyne;
} RobotDevices;

// Mapping state (Piero)
typedef struct {
    OctoMap* octomap;
    PointCloud* previous_scan;
    Transform3D cumulative_correction;
    int frame_count;
    int icp_enabled;
} MappingState;

// path node for A* algorithm
typedef struct PathNode {
    int x, y;  // coordinates
    double g_cost;  // cost
    double h_cost;  // heuristic
    double f_cost;   // total cost (g + h)
    struct PathNode* parent;  // for path reconstruction
} PathNode;

// path structure
typedef struct {
    int x[MAX_PATH_LENGTH];  // X coordinates
    int y[MAX_PATH_LENGTH];  // Y coordinates
    int length;  // number of waypoints
    int current_waypoint;  // Current target waypoint index
} Path;

// priority queue for A*
typedef struct {
    PathNode* nodes[MAX_PATH_LENGTH * 2];
    int size;
} PriorityQueue;

// hazard types detected from sensors
typedef enum {
    HAZARD_NONE = 0,
    HAZARD_STEEP_SLOPE = 1,  
    HAZARD_ROCK = 2,  
    HAZARD_HOLE = 3,  
    HAZARD_TIPPING = 4  
} HazardType;

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

static RobotDevices devices;
static SensorData sensor_data;
static OdometryData odometry;
static EKF ekf_filter;
static ElevationGrid elevation_grid;
static MappingState mapping_state;

// GPS offsets
static double gps_offset_x = 0.0;
static double gps_offset_y = 0.0;
static double gps_offset_z = 0.0;

// Voxel grid for downsampling (allocated once)
static char* g_voxel_grid = NULL;

static Path current_path; // for A* path planning
// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// EKF functions
void ekf_init(EKF *ekf);
void ekf_predict(double v, double omega, double dt);
void ekf_update_full(void);

// Odometry
void odometry_update(double d_left_rad, double d_right_rad);

// Device operations
void initialize_devices(void);
void read_sensors_data(void);

// 3D Mapping (Piero)
void mapping_init(void);
void mapping_cleanup(void);
void mapping_process_scan(const WbLidarPoint* raw_points, int num_points);
unsigned char mapping_get_traversability(double x, double y);
float mapping_get_slope(double x, double y);
float mapping_get_elevation(double x, double y);
void mapping_export_pgm(const char* filename);
void mapping_print_stats(void);

// OctoMap functions
OctoMap* octomap_create(double resolution, double size);
void octomap_destroy(OctoMap* map);
void octomap_insert_point(OctoMap* map, double px, double py, double pz,
                          double sensor_x, double sensor_y, double sensor_z);
float octomap_get_occupancy(OctoMap* map, double x, double y, double z);

// A* Path Planning functions
int astar_plan_path(int start_x, int start_y, int goal_x, int goal_y, Path* path);
double follow_path(Path* path, double robot_x, double robot_y, double robot_theta);

// hazard detection function
HazardType detect_hazards_ahead(double robot_x, double robot_y, double robot_theta);

// Point cloud functions
PointCloud* pointcloud_create(int capacity);
void pointcloud_destroy(PointCloud* cloud);
void pointcloud_add(PointCloud* cloud, float x, float y, float z);
PointCloud* pointcloud_filter_and_downsample(const WbLidarPoint* raw, int count);
PointCloud* pointcloud_transform(PointCloud* input, Transform3D* t);

// Transform functions
Transform3D transform_identity(void);
Transform3D transform_from_pose(double x, double y, double z, double roll, double pitch, double yaw);
Transform3D transform_multiply(Transform3D* a, Transform3D* b);
Point3D transform_point(Transform3D* t, Point3D* p);

// ICP
ICPResult icp_align(PointCloud* source, PointCloud* target);

// Navigation
double get_map_cost(double x, double y);
double calculate_navigation(double rob_x, double rob_y, double rob_theta, double goal_x, double goal_y);
int is_hazardous_state(void);

// ============================================================================
// TRANSFORM OPERATIONS
// ============================================================================

Transform3D transform_identity(void) {
    Transform3D t;
    memset(&t, 0, sizeof(Transform3D));
    t.m[0][0] = 1.0; t.m[1][1] = 1.0; t.m[2][2] = 1.0; t.m[3][3] = 1.0;
    return t;
}

Transform3D transform_from_pose(double x, double y, double z,
                                 double roll, double pitch, double yaw) {
    Transform3D t = transform_identity();
    
    double cy = cos(yaw), sy = sin(yaw);
    double cp = cos(pitch), sp = sin(pitch);
    double cr = cos(roll), sr = sin(roll);
    
    t.m[0][0] = cy * cp;
    t.m[0][1] = cy * sp * sr - sy * cr;
    t.m[0][2] = cy * sp * cr + sy * sr;
    t.m[0][3] = x;
    
    t.m[1][0] = sy * cp;
    t.m[1][1] = sy * sp * sr + cy * cr;
    t.m[1][2] = sy * sp * cr - cy * sr;
    t.m[1][3] = y;
    
    t.m[2][0] = -sp;
    t.m[2][1] = cp * sr;
    t.m[2][2] = cp * cr;
    t.m[2][3] = z;
    
    return t;
}

Transform3D transform_multiply(Transform3D* a, Transform3D* b) {
    Transform3D r;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            r.m[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                r.m[i][j] += a->m[i][k] * b->m[k][j];
            }
        }
    }
    return r;
}

Point3D transform_point(Transform3D* t, Point3D* p) {
    Point3D r;
    r.x = t->m[0][0] * p->x + t->m[0][1] * p->y + t->m[0][2] * p->z + t->m[0][3];
    r.y = t->m[1][0] * p->x + t->m[1][1] * p->y + t->m[1][2] * p->z + t->m[1][3];
    r.z = t->m[2][0] * p->x + t->m[2][1] * p->y + t->m[2][2] * p->z + t->m[2][3];
    return r;
}

// ============================================================================
// POINT CLOUD OPERATIONS
// ============================================================================

PointCloud* pointcloud_create(int capacity) {
    PointCloud* cloud = (PointCloud*)malloc(sizeof(PointCloud));
    if (!cloud) return NULL;
    
    cloud->points = (Point3D*)malloc(sizeof(Point3D) * capacity);
    if (!cloud->points) {
        free(cloud);
        return NULL;
    }
    
    cloud->count = 0;
    cloud->capacity = capacity;
    return cloud;
}

void pointcloud_destroy(PointCloud* cloud) {
    if (!cloud) return;
    if (cloud->points) free(cloud->points);
    free(cloud);
}

void pointcloud_add(PointCloud* cloud, float x, float y, float z) {
    if (!cloud || cloud->count >= cloud->capacity) return;
    cloud->points[cloud->count].x = x;
    cloud->points[cloud->count].y = y;
    cloud->points[cloud->count].z = z;
    cloud->count++;
}

PointCloud* pointcloud_filter_and_downsample(const WbLidarPoint* raw, int count) {
    if (!raw || count == 0) return NULL;
    
    // Allocate voxel grid once
    if (!g_voxel_grid) {
        g_voxel_grid = (char*)calloc(VOXEL_GRID_DIM * VOXEL_GRID_DIM * VOXEL_GRID_DIM, sizeof(char));
        if (!g_voxel_grid) {
            printf("[Mapping] ERROR: Failed to allocate voxel grid!\n");
            return NULL;
        }
    }
    
    // Clear grid
    memset(g_voxel_grid, 0, VOXEL_GRID_DIM * VOXEL_GRID_DIM * VOXEL_GRID_DIM);
    
    PointCloud* cloud = pointcloud_create(count);
    if (!cloud) return NULL;
    
    for (int i = 0; i < count; i++) {
        float px = raw[i].x;
        float py = raw[i].y;
        float pz = raw[i].z;
        
        // Distance filter
        float dist = sqrtf(px*px + py*py + pz*pz);
        if (dist < MIN_POINT_DISTANCE || dist > MAX_POINT_DISTANCE) continue;
        
        // Ground filter
        if (pz < GROUND_HEIGHT_THRESH) continue;
        
        // Voxel grid downsampling
        int vx = (int)((px + 20.0f) / VOXEL_LEAF_SIZE);
        int vy = (int)((py + 20.0f) / VOXEL_LEAF_SIZE);
        int vz = (int)((pz + 10.0f) / VOXEL_LEAF_SIZE);
        
        if (vx < 0 || vx >= VOXEL_GRID_DIM ||
            vy < 0 || vy >= VOXEL_GRID_DIM ||
            vz < 0 || vz >= VOXEL_GRID_DIM) continue;
        
        int idx = vx * VOXEL_GRID_DIM * VOXEL_GRID_DIM + vy * VOXEL_GRID_DIM + vz;
        
        if (g_voxel_grid[idx] == 0) {
            pointcloud_add(cloud, px, py, pz);
            g_voxel_grid[idx] = 1;
        }
    }
    
    return cloud;
}

PointCloud* pointcloud_transform(PointCloud* input, Transform3D* t) {
    if (!input || !t) return NULL;
    
    PointCloud* output = pointcloud_create(input->count);
    if (!output) return NULL;
    
    for (int i = 0; i < input->count; i++) {
        Point3D tp = transform_point(t, &input->points[i]);
        pointcloud_add(output, tp.x, tp.y, tp.z);
    }
    
    return output;
}

// ============================================================================
// OCTOMAP IMPLEMENTATION (Octree-based 3D Occupancy Grid)
// ============================================================================

static OctreeNode* octree_node_create(unsigned char depth) {
    OctreeNode* node = (OctreeNode*)calloc(1, sizeof(OctreeNode));
    if (!node) return NULL;
    
    node->occupancy_log = 0.0f;
    node->elevation_sum = 0.0f;
    node->hit_count = 0;
    node->is_leaf = 1;
    node->depth = depth;
    
    for (int i = 0; i < 8; i++) {
        node->children[i] = NULL;
    }
    
    return node;
}

static void octree_node_destroy(OctreeNode* node) {
    if (!node) return;
    for (int i = 0; i < 8; i++) {
        if (node->children[i]) {
            octree_node_destroy(node->children[i]);
        }
    }
    free(node);
}

static int get_child_index(double x, double y, double z, double cx, double cy, double cz) {
    int idx = 0;
    if (x >= cx) idx |= 1;
    if (y >= cy) idx |= 2;
    if (z >= cz) idx |= 4;
    return idx;
}

OctoMap* octomap_create(double resolution, double size) {
    OctoMap* map = (OctoMap*)calloc(1, sizeof(OctoMap));
    if (!map) return NULL;
    
    map->resolution = resolution;
    map->size = size;
    map->origin_x = -size / 2.0;
    map->origin_y = -size / 2.0;
    map->origin_z = -5.0;
    map->max_depth = OCTOMAP_MAX_DEPTH;
    
    map->root = octree_node_create(0);
    map->total_nodes = 1;
    map->leaf_nodes = 1;
    map->occupied_nodes = 0;
    
    printf("[OctoMap] Created: resolution=%.2fm, size=%.1fm, max_depth=%d\n", 
           resolution, size, OCTOMAP_MAX_DEPTH);
    
    return map;
}

void octomap_destroy(OctoMap* map) {
    if (!map) return;
    octree_node_destroy(map->root);
    free(map);
}

static void octomap_update_node(OctoMap* map, double x, double y, double z, int occupied) {
    if (!map || !map->root) return;
    
    // Bounds check
    double max_z = map->origin_z + map->size;
    if (x < map->origin_x || x >= map->origin_x + map->size ||
        y < map->origin_y || y >= map->origin_y + map->size ||
        z < map->origin_z || z >= max_z) {
        return;
    }
    
    OctreeNode* node = map->root;
    double node_size = map->size;
    double cx = map->origin_x + node_size / 2.0;
    double cy = map->origin_y + node_size / 2.0;
    double cz = map->origin_z + node_size / 2.0;
    
    int target_depth = (int)(log2(map->size / map->resolution));
    if (target_depth > map->max_depth) target_depth = map->max_depth;
    
    for (int d = 0; d < target_depth; d++) {
        int child_idx = get_child_index(x, y, z, cx, cy, cz);
        
        if (!node->children[child_idx]) {
            node->children[child_idx] = octree_node_create(d + 1);
            if (node->is_leaf) {
                node->is_leaf = 0;
                map->leaf_nodes--;
            }
            map->total_nodes++;
            map->leaf_nodes++;
        }
        
        node_size /= 2.0;
        cx += ((child_idx & 1) ? node_size/2.0 : -node_size/2.0);
        cy += ((child_idx & 2) ? node_size/2.0 : -node_size/2.0);
        cz += ((child_idx & 4) ? node_size/2.0 : -node_size/2.0);
        
        node = node->children[child_idx];
    }
    
    // Update occupancy with log-odds
    float prev_log = node->occupancy_log;
    if (occupied) {
        node->occupancy_log += PROB_HIT_LOG;
        node->elevation_sum += (float)z;
        node->hit_count++;
    } else {
        node->occupancy_log += PROB_MISS_LOG;
    }
    
    // Clamp
    if (node->occupancy_log < PROB_THRESH_MIN) node->occupancy_log = PROB_THRESH_MIN;
    if (node->occupancy_log > PROB_THRESH_MAX) node->occupancy_log = PROB_THRESH_MAX;
    
    // Track occupied count
    float prob = 1.0f / (1.0f + expf(-node->occupancy_log));
    float prev_prob = 1.0f / (1.0f + expf(-prev_log));
    
    if (prob >= PROB_OCCUPIED_THRESH && prev_prob < PROB_OCCUPIED_THRESH) {
        map->occupied_nodes++;
    } else if (prob < PROB_OCCUPIED_THRESH && prev_prob >= PROB_OCCUPIED_THRESH) {
        map->occupied_nodes--;
    }
}

void octomap_insert_point(OctoMap* map, double px, double py, double pz,
                          double sensor_x, double sensor_y, double sensor_z) {
    if (!map) return;
    
    // Mark endpoint as occupied
    octomap_update_node(map, px, py, pz, 1);
    
    // Ray casting: mark cells along ray as free (simplified)
    double dx = px - sensor_x;
    double dy = py - sensor_y;
    double dz = pz - sensor_z;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    
    if (dist < 0.2) return;
    
    int num_steps = (int)(dist / (map->resolution * 2));
    if (num_steps > 50) num_steps = 50;  // Limit for performance
    
    for (int s = 1; s < num_steps; s++) {
        double t = (double)s / (double)num_steps;
        double rx = sensor_x + dx * t;
        double ry = sensor_y + dy * t;
        double rz = sensor_z + dz * t;
        octomap_update_node(map, rx, ry, rz, 0);
    }
}

float octomap_get_occupancy(OctoMap* map, double x, double y, double z) {
    if (!map || !map->root) return 0.5f;
    
    OctreeNode* node = map->root;
    double node_size = map->size;
    double cx = map->origin_x + node_size / 2.0;
    double cy = map->origin_y + node_size / 2.0;
    double cz = map->origin_z + node_size / 2.0;
    
    while (node && !node->is_leaf) {
        int child_idx = get_child_index(x, y, z, cx, cy, cz);
        
        if (!node->children[child_idx]) return 0.5f;
        
        node_size /= 2.0;
        cx += ((child_idx & 1) ? node_size/2.0 : -node_size/2.0);
        cy += ((child_idx & 2) ? node_size/2.0 : -node_size/2.0);
        cz += ((child_idx & 4) ? node_size/2.0 : -node_size/2.0);
        
        node = node->children[child_idx];
    }
    
    if (!node) return 0.5f;
    return 1.0f / (1.0f + expf(-node->occupancy_log));
}

// ============================================================================
// ICP SCAN MATCHING (Point-to-Point, Simplified)
// ============================================================================

static int find_nearest_neighbor(Point3D* query, PointCloud* target, float* dist_out) {
    int best_idx = -1;
    float best_dist_sq = 1e10f;
    
    for (int i = 0; i < target->count; i++) {
        float dx = query->x - target->points[i].x;
        float dy = query->y - target->points[i].y;
        float dz = query->z - target->points[i].z;
        float d_sq = dx*dx + dy*dy + dz*dz;
        
        if (d_sq < best_dist_sq) {
            best_dist_sq = d_sq;
            best_idx = i;
        }
    }
    
    *dist_out = sqrtf(best_dist_sq);
    return best_idx;
}

ICPResult icp_align(PointCloud* source, PointCloud* target) {
    ICPResult result;
    result.converged = 0;
    result.iterations = 0;
    result.fitness_score = 1e10;
    result.transform = transform_identity();
    
    if (!source || !target || source->count < 20 || target->count < 20) {
        return result;
    }
    
    // Translation accumulator
    double tx = 0, ty = 0, tz = 0;
    
    for (int iter = 0; iter < ICP_MAX_ITERATIONS; iter++) {
        result.iterations = iter + 1;
        
        double sum_sx = 0, sum_sy = 0, sum_sz = 0;
        double sum_tx = 0, sum_ty = 0, sum_tz = 0;
        double total_error = 0;
        int num_corr = 0;
        
        for (int i = 0; i < source->count; i += 3) {  // Sample every 3rd point
            Point3D sp;
            sp.x = source->points[i].x + tx;
            sp.y = source->points[i].y + ty;
            sp.z = source->points[i].z + tz;
            
            float dist;
            int nn = find_nearest_neighbor(&sp, target, &dist);
            
            if (nn >= 0 && dist < ICP_MAX_CORRESPONDENCE_DIST) {
                sum_sx += source->points[i].x;
                sum_sy += source->points[i].y;
                sum_sz += source->points[i].z;
                sum_tx += target->points[nn].x;
                sum_ty += target->points[nn].y;
                sum_tz += target->points[nn].z;
                total_error += dist * dist;
                num_corr++;
            }
        }
        
        if (num_corr < 10) break;
        
        result.fitness_score = total_error / num_corr;
        
        // Compute centroid difference
        double mean_sx = sum_sx / num_corr;
        double mean_sy = sum_sy / num_corr;
        double mean_sz = sum_sz / num_corr;
        double mean_tx = sum_tx / num_corr;
        double mean_ty = sum_ty / num_corr;
        double mean_tz = sum_tz / num_corr;
        
        double dx = mean_tx - (mean_sx + tx);
        double dy = mean_ty - (mean_sy + ty);
        double dz = mean_tz - (mean_sz + tz);
        
        // Update translation
        tx += dx * 0.5;
        ty += dy * 0.5;
        tz += dz * 0.5;
        
        double delta = sqrt(dx*dx + dy*dy + dz*dz);
        if (delta < ICP_TOLERANCE) {
            result.converged = 1;
            break;
        }
    }
    
    result.transform.m[0][3] = tx;
    result.transform.m[1][3] = ty;
    result.transform.m[2][3] = tz;
    
    return result;
}

// ============================================================================
// ELEVATION GRID OPERATIONS
// ============================================================================

static void elevation_grid_init(void) {
    memset(&elevation_grid, 0, sizeof(ElevationGrid));
    
    // Initialize traversability as unknown (128)
    for (int x = 0; x < MAP_WIDTH; x++) {
        for (int y = 0; y < MAP_HEIGHT; y++) {
            elevation_grid.traversability[x][y] = 128;
        }
    }
    
    printf("[ElevationGrid] Initialized: %dx%d (%.1fm x %.1fm)\n",
           MAP_WIDTH, MAP_HEIGHT, MAP_WIDTH * MAP_RESOLUTION, MAP_HEIGHT * MAP_RESOLUTION);
}

static void elevation_grid_update(double gx, double gy, double gz) {
    int mx = (int)((gx + 20.0) / MAP_RESOLUTION);
    int my = (int)((gy + 20.0) / MAP_RESOLUTION);
    
    if (mx < 0 || mx >= MAP_WIDTH || my < 0 || my >= MAP_HEIGHT) return;
    
    // Running average for elevation
    int count = elevation_grid.hit_count[mx][my];
    elevation_grid.elevation[mx][my] = 
        (elevation_grid.elevation[mx][my] * count + gz) / (count + 1);
    elevation_grid.hit_count[mx][my]++;
}

static void elevation_grid_compute_slopes(void) {
    for (int x = 1; x < MAP_WIDTH - 1; x++) {
        for (int y = 1; y < MAP_HEIGHT - 1; y++) {
            if (elevation_grid.hit_count[x][y] < 3) continue;
            
            // Gradient using central differences
            float dz_dx = 0.0f, dz_dy = 0.0f;
            
            if (elevation_grid.hit_count[x+1][y] > 0 && elevation_grid.hit_count[x-1][y] > 0) {
                dz_dx = (elevation_grid.elevation[x+1][y] - elevation_grid.elevation[x-1][y]) 
                        / (2.0f * MAP_RESOLUTION);
            }
            
            if (elevation_grid.hit_count[x][y+1] > 0 && elevation_grid.hit_count[x][y-1] > 0) {
                dz_dy = (elevation_grid.elevation[x][y+1] - elevation_grid.elevation[x][y-1]) 
                        / (2.0f * MAP_RESOLUTION);
            }
            
            // Slope magnitude
            elevation_grid.slope[x][y] = sqrtf(dz_dx * dz_dx + dz_dy * dz_dy);
            
            // Roughness (local variance)
            float sum_sq = 0.0f;
            int neighbors = 0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx, ny = y + dy;
                    if (elevation_grid.hit_count[nx][ny] > 0) {
                        float diff = elevation_grid.elevation[nx][ny] - elevation_grid.elevation[x][y];
                        sum_sq += diff * diff;
                        neighbors++;
                    }
                }
            }
            if (neighbors > 0) {
                elevation_grid.roughness[x][y] = sqrtf(sum_sq / neighbors);
            }
        }
    }
}

static void elevation_grid_compute_traversability(void) {
    const float max_slope = 0.5f;      // ~27 degrees
    const float max_roughness = 0.15f;
    
    int traversable = 0, obstacles = 0, unknown = 0;
    
    for (int x = 0; x < MAP_WIDTH; x++) {
        for (int y = 0; y < MAP_HEIGHT; y++) {
            if (elevation_grid.hit_count[x][y] < 3) {
                elevation_grid.traversability[x][y] = 128;  // Unknown
                unknown++;
                continue;
            }
            
            float slope = elevation_grid.slope[x][y];
            float roughness = elevation_grid.roughness[x][y];
            
            // Cost function
            float slope_cost = (slope / max_slope) * 200.0f;
            float rough_cost = (roughness / max_roughness) * 55.0f;
            float total_cost = slope_cost + rough_cost;
            
            if (total_cost > 255.0f) total_cost = 255.0f;
            if (total_cost < 0.0f) total_cost = 0.0f;
            
            elevation_grid.traversability[x][y] = (unsigned char)total_cost;
            
            if (total_cost > 200) obstacles++;
            else traversable++;
        }
    }
    
    printf("[Traversability] %d traversable, %d obstacles, %d unknown\n",
           traversable, obstacles, unknown);
}

// ============================================================================
// MAIN MAPPING INTERFACE (Piero Flores López)
// ============================================================================

void mapping_init(void) {
    mapping_state.octomap = octomap_create(OCTOMAP_RESOLUTION, OCTOMAP_SIZE);
    mapping_state.previous_scan = NULL;
    mapping_state.cumulative_correction = transform_identity();
    mapping_state.frame_count = 0;
    mapping_state.icp_enabled = 1;
    
    elevation_grid_init();
    
    printf("[Mapping] 3D Mapping system initialized (Piero Flores López)\n");
}

void mapping_cleanup(void) {
    if (mapping_state.octomap) {
        octomap_destroy(mapping_state.octomap);
        mapping_state.octomap = NULL;
    }
    
    if (mapping_state.previous_scan) {
        pointcloud_destroy(mapping_state.previous_scan);
        mapping_state.previous_scan = NULL;
    }
    
    if (g_voxel_grid) {
        free(g_voxel_grid);
        g_voxel_grid = NULL;
    }
    
    printf("[Mapping] Cleanup complete\n");
}

void mapping_process_scan(const WbLidarPoint* raw_points, int num_points) {
    if (!raw_points || num_points == 0) return;
    
    mapping_state.frame_count++;
    
    // Filter and downsample
    PointCloud* filtered = pointcloud_filter_and_downsample(raw_points, num_points);
    if (!filtered || filtered->count == 0) {
        if (filtered) pointcloud_destroy(filtered);
        return;
    }
    
    // Get robot pose from EKF
    double rx = ekf_filter.state[0];
    double ry = ekf_filter.state[1];
    double rz = ekf_filter.state[2];
    double roll = ekf_filter.state[3];
    double pitch = ekf_filter.state[4];
    double yaw = ekf_filter.state[5];
    
    Transform3D robot_pose = transform_from_pose(rx, ry, rz, roll, pitch, yaw);
    
    // ICP scan matching for drift correction
    if (mapping_state.icp_enabled && mapping_state.previous_scan && 
        mapping_state.previous_scan->count > 50) {
        
        ICPResult icp = icp_align(filtered, mapping_state.previous_scan);
        
        if (icp.converged && icp.fitness_score < 0.2) {
            // Apply ICP correction to cumulative transform
            mapping_state.cumulative_correction = 
                transform_multiply(&icp.transform, &mapping_state.cumulative_correction);
            
            if (mapping_state.frame_count % 100 == 0) {
                printf("[ICP] Correction applied: fitness=%.4f, iters=%d\n", 
                       icp.fitness_score, icp.iterations);
            }
        }
    }
    
    // Combine robot pose with ICP correction
    Transform3D corrected_pose = transform_multiply(&robot_pose, &mapping_state.cumulative_correction);
    
    // Transform points to global frame
    PointCloud* global_cloud = pointcloud_transform(filtered, &corrected_pose);
    
    if (global_cloud) {
        // Update OctoMap
        for (int i = 0; i < global_cloud->count; i++) {
            Point3D* p = &global_cloud->points[i];
            octomap_insert_point(mapping_state.octomap, p->x, p->y, p->z, rx, ry, rz);
            
            // Update elevation grid
            elevation_grid_update(p->x, p->y, p->z);
        }
        
        pointcloud_destroy(global_cloud);
    }
    
    // Store for next ICP
    if (mapping_state.previous_scan) {
        pointcloud_destroy(mapping_state.previous_scan);
    }
    mapping_state.previous_scan = filtered;
    
    // Periodic slope/traversability update
    if (mapping_state.frame_count % 50 == 0) {
        elevation_grid_compute_slopes();
        elevation_grid_compute_traversability();
    }
}

unsigned char mapping_get_traversability(double x, double y) {
    int mx = (int)((x + 20.0) / MAP_RESOLUTION);
    int my = (int)((y + 20.0) / MAP_RESOLUTION);
    
    if (mx < 0 || mx >= MAP_WIDTH || my < 0 || my >= MAP_HEIGHT) {
        return 128;  // Unknown
    }
    
    return elevation_grid.traversability[mx][my];
}

float mapping_get_slope(double x, double y) {
    int mx = (int)((x + 20.0) / MAP_RESOLUTION);
    int my = (int)((y + 20.0) / MAP_RESOLUTION);
    
    if (mx < 0 || mx >= MAP_WIDTH || my < 0 || my >= MAP_HEIGHT) {
        return 0.0f;
    }
    
    return elevation_grid.slope[mx][my];
}

float mapping_get_elevation(double x, double y) {
    int mx = (int)((x + 20.0) / MAP_RESOLUTION);
    int my = (int)((y + 20.0) / MAP_RESOLUTION);
    
    if (mx < 0 || mx >= MAP_WIDTH || my < 0 || my >= MAP_HEIGHT) {
        return 0.0f;
    }
    
    return elevation_grid.elevation[mx][my];
}

void mapping_export_pgm(const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("[Mapping] ERROR: Cannot create %s\n", filename);
        return;
    }
    
    fprintf(fp, "P5\n%d %d\n255\n", MAP_WIDTH, MAP_HEIGHT);
    
    for (int y = MAP_HEIGHT - 1; y >= 0; y--) {
        for (int x = 0; x < MAP_WIDTH; x++) {
            unsigned char val = 255 - elevation_grid.traversability[x][y];
            fwrite(&val, 1, 1, fp);
        }
    }
    
    fclose(fp);
    printf("[Mapping] Exported traversability map to %s\n", filename);
}

void mapping_print_stats(void) {
    printf("\n[3D Mapping Statistics - Piero Flores López]\n");
    printf("  Frames processed: %d\n", mapping_state.frame_count);
    
    if (mapping_state.octomap) {
        printf("  OctoMap: %d nodes, %d leaves, %d occupied\n",
               mapping_state.octomap->total_nodes,
               mapping_state.octomap->leaf_nodes,
               mapping_state.octomap->occupied_nodes);
    }
    
    int observed = 0;
    for (int x = 0; x < MAP_WIDTH; x++) {
        for (int y = 0; y < MAP_HEIGHT; y++) {
            if (elevation_grid.hit_count[x][y] > 0) observed++;
        }
    }
    
    printf("  Elevation grid: %d/%d cells observed (%.1f%%)\n",
           observed, MAP_WIDTH * MAP_HEIGHT,
           100.0 * observed / (MAP_WIDTH * MAP_HEIGHT));
}

// ============================================================================
// EKF IMPLEMENTATION
// ============================================================================

void ekf_init(EKF *ekf) {
    memset(ekf, 0, sizeof(EKF));
    
    ekf->covariance[0][0] = 0.1;
    ekf->covariance[1][1] = 0.1;
    ekf->covariance[2][2] = 0.1;
    ekf->covariance[3][3] = 0.05;
    ekf->covariance[4][4] = 0.05;
    ekf->covariance[5][5] = 0.1;
    
    for(int i = 0; i < 6; i++) {
        ekf->Q[i][i] = 0.01;
    }
    
    ekf->R[0][0] = 0.5;
    ekf->R[1][1] = 0.5;
    ekf->R[2][2] = 0.5;
    ekf->R[3][3] = 0.05;
    ekf->R[4][4] = 0.05;
    ekf->R[5][5] = 0.05;
}

void ekf_predict(double v, double omega, double dt) {
    double yaw = ekf_filter.state[5];
    
    ekf_filter.state[0] += v * cos(yaw) * dt;
    ekf_filter.state[1] += v * sin(yaw) * dt;
    ekf_filter.state[3] = sensor_data.roll;
    ekf_filter.state[4] = sensor_data.pitch;
    ekf_filter.state[5] += omega * dt;
    
    for(int i = 3; i < 6; i++) {
        while (ekf_filter.state[i] > M_PI) ekf_filter.state[i] -= 2.0 * M_PI;
        while (ekf_filter.state[i] < -M_PI) ekf_filter.state[i] += 2.0 * M_PI;
    }
    
    for(int i = 0; i < 6; i++) {
        ekf_filter.covariance[i][i] += ekf_filter.Q[i][i];
    }
}

void ekf_update_full(void) {
    // GPS update
    if(devices.gps) {
        for(int i = 0; i < 3; i++) {
            double measurement;
            if(i == 0) measurement = sensor_data.gps_x - gps_offset_x;
            else if(i == 1) measurement = sensor_data.gps_y - gps_offset_y;
            else measurement = sensor_data.gps_z - gps_offset_z;
            
            double innovation = measurement - ekf_filter.state[i];
            double S = ekf_filter.covariance[i][i] + ekf_filter.R[i][i];
            double K = ekf_filter.covariance[i][i] / S;
            
            ekf_filter.state[i] += K * innovation;
            ekf_filter.covariance[i][i] = (1 - K) * ekf_filter.covariance[i][i];
        }
    }
    
    // IMU update (Roll, Pitch)
    if(devices.imu) {
        double innov_roll = sensor_data.roll - ekf_filter.state[3];
        while(innov_roll > M_PI) innov_roll -= 2*M_PI;
        while(innov_roll < -M_PI) innov_roll += 2*M_PI;
        double S = ekf_filter.covariance[3][3] + ekf_filter.R[3][3];
        double K = ekf_filter.covariance[3][3] / S;
        ekf_filter.state[3] += K * innov_roll;
        ekf_filter.covariance[3][3] = (1 - K) * ekf_filter.covariance[3][3];
        
        double innov_pitch = sensor_data.pitch - ekf_filter.state[4];
        while(innov_pitch > M_PI) innov_pitch -= 2*M_PI;
        while(innov_pitch < -M_PI) innov_pitch += 2*M_PI;
        S = ekf_filter.covariance[4][4] + ekf_filter.R[4][4];
        K = ekf_filter.covariance[4][4] / S;
        ekf_filter.state[4] += K * innov_pitch;
        ekf_filter.covariance[4][4] = (1 - K) * ekf_filter.covariance[4][4];
    }
    
    // Compass update (Yaw)
    if(devices.compass) {
        double innov_yaw = sensor_data.compass_yaw - ekf_filter.state[5];
        while(innov_yaw > M_PI) innov_yaw -= 2*M_PI;
        while(innov_yaw < -M_PI) innov_yaw += 2*M_PI;
        double S = ekf_filter.covariance[5][5] + ekf_filter.R[5][5];
        double K = ekf_filter.covariance[5][5] / S;
        ekf_filter.state[5] += K * innov_yaw;
        ekf_filter.covariance[5][5] = (1 - K) * ekf_filter.covariance[5][5];
    }
}

void odometry_update(double d_left_rad, double d_right_rad) {
    double dl = d_left_rad * WHEEL_RADIUS;
    double dr = d_right_rad * WHEEL_RADIUS;
    double dc = (dl + dr) / 2.0;
    double dth = (dr - dl) / WHEEL_BASE;

    odometry.yaw += dth;
    while(odometry.yaw > M_PI) odometry.yaw -= 2*M_PI;
    while(odometry.yaw < -M_PI) odometry.yaw += 2*M_PI;

    odometry.x += dc * cos(odometry.yaw);
    odometry.y += dc * sin(odometry.yaw);
}

// ============================================================================
// DEVICE OPERATIONS
// ============================================================================

void initialize_devices(void) {
    char name[32];
    
    for (int i = 0; i < 4; i++) {
        sprintf(name, "right motor %d", i+1);
        devices.right_motors[i] = wb_robot_get_device(name);
        if(devices.right_motors[i]) {
            wb_motor_set_position(devices.right_motors[i], INFINITY);
            wb_motor_set_velocity(devices.right_motors[i], 0.0);
        }
        
        sprintf(name, "right wheel sensor %d", i+1);
        devices.right_sensors[i] = wb_robot_get_device(name);
        if(devices.right_sensors[i]) wb_position_sensor_enable(devices.right_sensors[i], TIME_STEP);

        sprintf(name, "left motor %d", i+1);
        devices.left_motors[i] = wb_robot_get_device(name);
        if(devices.left_motors[i]) {
            wb_motor_set_position(devices.left_motors[i], INFINITY);
            wb_motor_set_velocity(devices.left_motors[i], 0.0);
        }
        
        sprintf(name, "left wheel sensor %d", i+1);
        devices.left_sensors[i] = wb_robot_get_device(name);
        if(devices.left_sensors[i]) wb_position_sensor_enable(devices.left_sensors[i], TIME_STEP);
    }

    devices.compass = wb_robot_get_device("compass");
    if(devices.compass) wb_compass_enable(devices.compass, TIME_STEP);
    
    devices.gps = wb_robot_get_device("gps");
    if(devices.gps) wb_gps_enable(devices.gps, TIME_STEP);
    
    devices.velodyne = wb_robot_get_device("velodyne");
    if(devices.velodyne) {
        wb_lidar_enable(devices.velodyne, TIME_STEP);
        wb_lidar_enable_point_cloud(devices.velodyne);
        printf("[Init] Velodyne LiDAR initialized\n");
    } else {
        printf("[Init] WARNING: No Velodyne LiDAR found\n");
    }
    
    devices.imu = 0;
    int n_devices = wb_robot_get_number_of_devices();
    
    for (int i = 0; i < n_devices; i++) {
        WbDeviceTag tag = wb_robot_get_device_by_index(i);
        if (wb_device_get_node_type(tag) == WB_NODE_INERTIAL_UNIT) {
            devices.imu = tag;
            printf("[Init] IMU found (Tag: %d)\n", tag);
            wb_inertial_unit_enable(devices.imu, TIME_STEP);
            break;
        }
    }

    if (!devices.imu) {
        printf("[Init] WARNING: No IMU found\n");
    }

    ekf_init(&ekf_filter);
}

void read_sensors_data(void) {
    for(int i=0; i<4; i++) {
        if(devices.left_sensors[i]) 
            sensor_data.left_pos[i] = wb_position_sensor_get_value(devices.left_sensors[i]);
        if(devices.right_sensors[i]) 
            sensor_data.right_pos[i] = wb_position_sensor_get_value(devices.right_sensors[i]);
    }
    
    if(devices.compass) {
        const double *north = wb_compass_get_values(devices.compass);
        sensor_data.compass_yaw = atan2(north[0], north[1]); 
    }
    
    if(devices.gps) {
        const double *pos = wb_gps_get_values(devices.gps);
        sensor_data.gps_x = pos[0];
        sensor_data.gps_y = pos[1];
        sensor_data.gps_z = pos[2];
    }

    if(devices.imu) {
        const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(devices.imu);
        sensor_data.roll = rpy[0];
        sensor_data.pitch = rpy[1];
    }
}

// ============================================================================
// A* PATH PLANNING
// ============================================================================

  static void world_to_path_grid(double wx, double wy, int* gx, int* gy) {
      *gx = (int)((wx + 20.0) / PATH_MAP_RESOLUTION);
      *gy = (int)((wy + 20.0) / PATH_MAP_RESOLUTION);
      
      if (*gx < 0) *gx = 0;
      if (*gx >= PATH_MAP_SIZE) *gx = PATH_MAP_SIZE - 1;
      if (*gy < 0) *gy = 0;
      if (*gy >= PATH_MAP_SIZE) *gy = PATH_MAP_SIZE - 1;
  }
  
  static void path_grid_to_world(int gx, int gy, double* wx, double* wy) {
      *wx = (gx * PATH_MAP_RESOLUTION) - 20.0;
      *wy = (gy * PATH_MAP_RESOLUTION) - 20.0;
  }
  
  // priority queues
  static void pq_init(PriorityQueue* pq) {
      pq->size = 0;
  }
  
  static void pq_push(PriorityQueue* pq, PathNode* node) {
    if (pq->size >= MAX_PATH_LENGTH * 2) return;
    pq->nodes[pq->size++] = node;
    
    // bubble up
    int i = pq->size - 1;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (pq->nodes[i]->f_cost >= pq->nodes[parent]->f_cost) break;
        
        PathNode* temp = pq->nodes[i];
        pq->nodes[i] = pq->nodes[parent];
        pq->nodes[parent] = temp;
        i = parent;
    }
  }
  
  
  static PathNode* pq_pop(PriorityQueue* pq) {
    if (pq->size == 0) return NULL;
    
    PathNode* result = pq->nodes[0];
    pq->nodes[0] = pq->nodes[--pq->size];
    
    // bubble down
    int i = 0;
    while (1) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;
        
        if (left < pq->size && pq->nodes[left]->f_cost < pq->nodes[smallest]->f_cost)
            smallest = left;
        if (right < pq->size && pq->nodes[right]->f_cost < pq->nodes[smallest]->f_cost)
            smallest = right;
            
        if (smallest == i) break;
        
        PathNode* temp = pq->nodes[i];
        pq->nodes[i] = pq->nodes[smallest];
        pq->nodes[smallest] = temp;
        i = smallest;
    }
    
    return result;
  }
  
  // heuristic
  static double path_heuristic(int x1, int y1, int x2, int y2) {
      return sqrt((double)((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)));
  }
  
  // A* Algorithm
  int astar_plan_path(int start_x, int start_y, int goal_x, int goal_y, Path* path) {
      static char visited[PATH_MAP_SIZE][PATH_MAP_SIZE];
      memset(visited, 0, sizeof(visited));
      
      PriorityQueue open_set;
      pq_init(&open_set);
      
      // allocate nodes for A* search
      PathNode* nodes = (PathNode*)malloc(sizeof(PathNode) * PATH_MAP_SIZE * PATH_MAP_SIZE);
      if (!nodes) {
          printf("[A* PATH PLANNING ERROR] Memory allocation failed\n");
          return 0;
      }
      int node_count = 0;
      
      // create start node
      PathNode* start_node = &nodes[node_count++];
      start_node->x = start_x;
      start_node->y = start_y;
      start_node->g_cost = 0;
      start_node->h_cost = path_heuristic(start_x, start_y, goal_x, goal_y);
      start_node->f_cost = start_node->h_cost;
      start_node->parent = NULL;
      
      pq_push(&open_set, start_node);
      
      PathNode* goal_node = NULL;
      int iterations = 0;
      int max_iterations = PATH_MAP_SIZE * PATH_MAP_SIZE / 4;
      
      // search loop
      while (open_set.size > 0 && iterations < max_iterations) {
          iterations++;
          PathNode* current = pq_pop(&open_set);
          
          // goal reached
          if (current->x == goal_x && current->y == goal_y) {
              goal_node = current;
              break;
          }
          
          visited[current->x][current->y] = 1;
          
          // explore neighbors
          int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
          int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
          
          for (int i = 0; i < 8; i++) {
              int nx = current->x + dx[i];
              int ny = current->y + dy[i];
              
              // bounds check
              if (nx < 0 || nx >= PATH_MAP_SIZE || ny < 0 || ny >= PATH_MAP_SIZE) continue;
              
              // skip visited
              if (visited[nx][ny]) continue;
              
              // get terrain cost from mapping module
              double wx, wy;
              path_grid_to_world(nx, ny, &wx, &wy);
              double terrain_cost = get_map_cost(wx, wy);
              
              // skip impassable cells 
              // cost > 200 = steep slope or obstacle
              if (terrain_cost > 200.0) continue;
              
              // calculate movement cost 
              // diagonal costs more
              double move_cost = (i < 4) ? 1.0 : 1.414;
              double new_g = current->g_cost + move_cost + (terrain_cost / 20.0);
              
              // create neighbor node
              if (node_count >= PATH_MAP_SIZE * PATH_MAP_SIZE) continue;
              PathNode* neighbor = &nodes[node_count++];
              neighbor->x = nx;
              neighbor->y = ny;
              neighbor->g_cost = new_g;
              neighbor->h_cost = path_heuristic(nx, ny, goal_x, goal_y);
              neighbor->f_cost = neighbor->g_cost + neighbor->h_cost;
              neighbor->parent = current;
              
              pq_push(&open_set, neighbor);
          }
      }
      
      // reconstruct path from goal to start
      if (goal_node) {
          path->length = 0;
          PathNode* current = goal_node;
          
          // backtrack through parents
          while (current != NULL && path->length < MAX_PATH_LENGTH) {
              path->x[path->length] = current->x;
              path->y[path->length] = current->y;
              path->length++;
              current = current->parent;
          }
          
          // reverse path (start -> goal)
          for (int i = 0; i < path->length / 2; i++) {
              int temp_x = path->x[i];
              int temp_y = path->y[i];
              path->x[i] = path->x[path->length - 1 - i];
              path->y[i] = path->y[path->length - 1 - i];
              path->x[path->length - 1 - i] = temp_x;
              path->y[path->length - 1 - i] = temp_y;
          }
          
          path->current_waypoint = 0;
          free(nodes);
          
          printf("[A*] Path found: %d waypoints, %d iterations\n", path->length, iterations);
          return 1;
      }
      
      // free memory allocated
      free(nodes);
      printf("[A*] No path found after %d iterations\n", iterations);
      return 0;
  }
  
  // path Following Controller
  double follow_path(Path* path, double robot_x, double robot_y, double robot_theta) {
      if (path->length == 0 || path->current_waypoint >= path->length) {
          return 0.0; // no path
      }
      
      // get current waypoint coordinates
      double wx, wy;
      path_grid_to_world(path->x[path->current_waypoint], 
                         path->y[path->current_waypoint], &wx, &wy);
      
      // check if waypoint reached
      double dist_to_waypoint = sqrt((wx - robot_x)*(wx - robot_x) + 
                                     (wy - robot_y)*(wy - robot_y));
      
      if (dist_to_waypoint < 0.5) { // within 0.5m
          path->current_waypoint++;
          if (path->current_waypoint >= path->length) {
              printf("[Path Follow] *** GOAL REACHED! ***\n");
              return 0.0;
          }
          // update to next waypoint
          path_grid_to_world(path->x[path->current_waypoint], 
                            path->y[path->current_waypoint], &wx, &wy);
      }
      
      // calculate steering command
      double dx = wx - robot_x;
      double dy = wy - robot_y;
      double target_angle = atan2(dy, dx);
      double error = target_angle - robot_theta;
      
      // normalise angle
      while(error > M_PI) error -= 2.0 * M_PI;
      while(error < -M_PI) error += 2.0 * M_PI;
      
      return error; // steering command
  }
  
  
  // ============================================================================
// HAZARD DETECTION 
// ============================================================================

// detect hazards from LIDAR + IMU data  
  HazardType detect_hazards_ahead(double robot_x, double robot_y, double robot_theta) {
      
      double lookahead = 2.0; // look 2m ahead
      double check_x = robot_x + cos(robot_theta) * lookahead;
      double check_y = robot_y + sin(robot_theta) * lookahead;
      
      // check for steep slopes
      unsigned char trav = mapping_get_traversability(check_x, check_y);
      float slope = mapping_get_slope(check_x, check_y);
      
      // high cost = obstacle/steep slope detected
      if (trav > 200 || slope > MAX_SAFE_PITCH) {
          return HAZARD_STEEP_SLOPE;
      }
      
      // check for rocks
      if (trav > 150 && slope < 0.2) {
          return HAZARD_ROCK;
      }
      
      // check for holes (unknown areas surrounded by known terrain)
      if (trav == 128) { // Unknown area
          // check if surrounded by known areas
          int known_neighbors = 0;
          for (double r = 0.5; r < 1.5; r += 0.5) {
              for (double a = 0; a < 2*M_PI; a += M_PI/4) {
                  double nx = check_x + r * cos(a);
                  double ny = check_y + r * sin(a);
                  if (mapping_get_traversability(nx, ny) != 128) {
                      known_neighbors++;
                  }
              }
          }
          // if mostly surrounded by known terrain, this unknown area is likely a hole
          if (known_neighbors > 6) {
              return HAZARD_HOLE;
          }
      }
      
      return HAZARD_NONE;
  }

// ============================================================================
// NAVIGATION (uses real map data from Piero's module)
// ============================================================================

double get_map_cost(double x, double y) {
    // Get traversability from real sensor-derived map
    unsigned char trav = mapping_get_traversability(x, y);
    
    // Unknown areas get medium cost
    if (trav == 128) {
        return 50.0;
    }
    
    // Add slope penalty
    float slope = mapping_get_slope(x, y);
    double cost = (double)trav;
    
    // Extra penalty for steep slopes (>20 degrees ≈ 0.36 rad)
    if (slope > 0.36) {
        cost += (slope - 0.36) * 200.0;
    }
    
    if (cost > 255.0) cost = 255.0;
    
    return cost;
}

double calculate_navigation(double rob_x, double rob_y, double rob_theta, 
                            double goal_x, double goal_y) {
    double lookahead = 2.0;
    
    // Check cost ahead
    double next_x = rob_x + cos(rob_theta) * lookahead;
    double next_y = rob_y + sin(rob_theta) * lookahead;
    double cost_ahead = get_map_cost(next_x, next_y);
    
    // If high cost, check alternatives
    if (cost_ahead > 100.0) {
        double left_x = rob_x + cos(rob_theta + 0.5) * lookahead;
        double left_y = rob_y + sin(rob_theta + 0.5) * lookahead;
        double cost_left = get_map_cost(left_x, left_y);
        
        double right_x = rob_x + cos(rob_theta - 0.5) * lookahead;
        double right_y = rob_y + sin(rob_theta - 0.5) * lookahead;
        double cost_right = get_map_cost(right_x, right_y);
        
        if (cost_left < cost_ahead && cost_left < cost_right) {
            printf("[Nav] Slope detected! Turning LEFT (cost: %.0f->%.0f)\n", cost_ahead, cost_left);
            return 0.8;
        } else if (cost_right < cost_ahead) {
            printf("[Nav] Slope detected! Turning RIGHT (cost: %.0f->%.0f)\n", cost_ahead, cost_right);
            return -0.8;
        }
    }
    
    // Navigate toward goal
    double dx = goal_x - rob_x;
    double dy = goal_y - rob_y;
    double target_angle = atan2(dy, dx);
    double error = target_angle - rob_theta;
    
    while(error > M_PI) error -= 2.0 * M_PI;
    while(error < -M_PI) error += 2.0 * M_PI;
    
    return error;
}

// detects if robot is currently tipping over
  int is_hazardous_state(void) {
     static int hazard_counter = 0;
  
      if (fabs(sensor_data.roll) > MAX_SAFE_ROLL || fabs(sensor_data.pitch) > MAX_SAFE_PITCH) {
          if (hazard_counter % 30 == 0) {
              printf("!!! HAZARD: Roll=%.2f°, Pitch=%.2f° !!!\n", 
                     sensor_data.roll * 180/M_PI, sensor_data.pitch * 180/M_PI);
          }
          hazard_counter++;
          return 1;
      }
  
      hazard_counter = 0;
      return 0;
  }

// ============================================================================
// MAIN CONTROL LOOP
// ============================================================================

int main(void) {
    wb_robot_init();

    printf("=== Dispositivos disponibles ===\n");
    int n = wb_robot_get_number_of_devices();
    for (int i = 0; i < n; i++) {
        WbDeviceTag tag = wb_robot_get_device_by_index(i);
        const char* name = wb_device_get_name(tag);
        int type = wb_device_get_node_type(tag);
        printf("  [%d] %s (tipo: %d)\n", i, name, type);
    }
    printf("================================\n");
    
    printf("\n============================================================\n");
    printf("  Grass Cutting Robot - 3D Navigation & Mapping System\n");
    printf("  EKF + OctoMap + ICP Scan Matching\n");
    printf("  3D Mapping: Piero Flores López\n");
    printf("============================================================\n\n");

    initialize_devices();
    mapping_init();
    
    double goal_x = 10.0;
    double goal_y = 2.0;
    
    // Warm up sensors
    printf("[Init] Warming up sensors...\n");
    for(int i = 0; i < 10; i++) {
        wb_robot_step(TIME_STEP);
        read_sensors_data();
    }
    
    // Calibrate initial pose
    printf("[Init] Calibrating initial pose...\n");
    gps_offset_x = sensor_data.gps_x;
    gps_offset_y = sensor_data.gps_y;
    gps_offset_z = sensor_data.gps_z;
    
    double start_l = 0, start_r = 0;
    int cnt_l = 0, cnt_r = 0;
    for(int i = 0; i < 4; i++) {
        if(devices.left_sensors[i]) { start_l += sensor_data.left_pos[i]; cnt_l++; }
        if(devices.right_sensors[i]) { start_r += sensor_data.right_pos[i]; cnt_r++; }
    }
    odometry.prev_left_enc = (cnt_l > 0) ? start_l/cnt_l : 0;
    odometry.prev_right_enc = (cnt_r > 0) ? start_r/cnt_r : 0;
    
    odometry.x = 0.0;
    odometry.y = 0.0;
    odometry.z = 0.0;
    odometry.roll = sensor_data.roll;
    odometry.pitch = sensor_data.pitch;
    odometry.yaw = sensor_data.compass_yaw;
    
    for (int i = 0; i < 6; i++) ekf_filter.state[i] = 0.0;
    ekf_filter.state[3] = sensor_data.roll;
    ekf_filter.state[4] = sensor_data.pitch;
    ekf_filter.state[5] = sensor_data.compass_yaw;
    
    printf("[Init] GPS offset: (%.2f, %.2f, %.2f)\n", gps_offset_x, gps_offset_y, gps_offset_z);
    printf("[Init] Initial orientation: Roll=%.1f°, Pitch=%.1f°, Yaw=%.1f°\n",
           sensor_data.roll * 180/M_PI, sensor_data.pitch * 180/M_PI, sensor_data.compass_yaw * 180/M_PI);
    printf("[Init] Calibration complete!\n\n");

    int step_count = 0;
    double dt = TIME_STEP / 1000.0;
    
    // ========== MAIN LOOP ==========
    while (wb_robot_step(TIME_STEP) != -1) {
        step_count++;
        read_sensors_data();
        
        // Safety check
        if (is_hazardous_state()) {
            for(int i = 0; i < 4; i++) {
                if(devices.left_motors[i]) wb_motor_set_velocity(devices.left_motors[i], 0);
                if(devices.right_motors[i]) wb_motor_set_velocity(devices.right_motors[i], 0);
            }
            continue;
        }
        
        // Encoder processing
        double curr_l = 0, curr_r = 0;
        cnt_l = 0; cnt_r = 0;
        for(int i = 0; i < 4; i++) {
            if(devices.left_sensors[i]) { curr_l += sensor_data.left_pos[i]; cnt_l++; }
            if(devices.right_sensors[i]) { curr_r += sensor_data.right_pos[i]; cnt_r++; }
        }
        if(cnt_l > 0) curr_l /= cnt_l;
        if(cnt_r > 0) curr_r /= cnt_r;
        
        double d_left = curr_l - odometry.prev_left_enc;
        double d_right = curr_r - odometry.prev_right_enc;
        odometry.prev_left_enc = curr_l;
        odometry.prev_right_enc = curr_r;
        
        // Odometry update
        odometry_update(d_left, d_right);
        
        double v_linear = ((d_left + d_right) / 2.0 * WHEEL_RADIUS) / dt;
        double v_angular = ((d_right - d_left) * WHEEL_RADIUS / WHEEL_BASE) / dt;
        
        // EKF
        ekf_predict(v_linear, v_angular, dt);
        ekf_update_full();
        
        // ========== 3D MAPPING (Piero) ==========
        if(devices.velodyne && step_count % 5 == 0) {
            const WbLidarPoint* raw_points = wb_lidar_get_point_cloud(devices.velodyne);
            int num_points = wb_lidar_get_number_of_points(devices.velodyne);
            
            if(num_points > 0 && raw_points != NULL) {
                mapping_process_scan(raw_points, num_points);
            }
        }
        
        // ========== NAVIGATION ==========
        double cx = ekf_filter.state[0];
        double cy = ekf_filter.state[1];
        double cz = ekf_filter.state[2];
        double cyaw = ekf_filter.state[5];

        double turn_cmd = calculate_navigation(cx, cy, cyaw, goal_x, goal_y);
        double fwd_cmd = 2.0;

        double left_speed = fwd_cmd - turn_cmd;
        double right_speed = fwd_cmd + turn_cmd;

        if(left_speed > 5.0) left_speed = 5.0;
        if(left_speed < -5.0) left_speed = -5.0;
        if(right_speed > 5.0) right_speed = 5.0;
        if(right_speed < -5.0) right_speed = -5.0;

        for(int i = 0; i < 4; i++) {
            if(devices.left_motors[i]) wb_motor_set_velocity(devices.left_motors[i], left_speed);
            if(devices.right_motors[i]) wb_motor_set_velocity(devices.right_motors[i], right_speed);
        }

        // ========== DEBUG OUTPUT ==========
        if(step_count % (1000 / TIME_STEP) == 0) {
            double lookahead_x = cx + cos(cyaw) * 2.0;
            double lookahead_y = cy + sin(cyaw) * 2.0;
            double map_cost = get_map_cost(lookahead_x, lookahead_y);
            
            double gps_local_x = sensor_data.gps_x - gps_offset_x;
            double gps_local_y = sensor_data.gps_y - gps_offset_y;
            double gps_local_z = sensor_data.gps_z - gps_offset_z;
            
            double dist_err_2d = sqrt(pow(ekf_filter.state[0] - gps_local_x, 2) + 
                                      pow(ekf_filter.state[1] - gps_local_y, 2));
            double dist_err_3d = sqrt(pow(ekf_filter.state[0] - gps_local_x, 2) + 
                                      pow(ekf_filter.state[1] - gps_local_y, 2) +
                                      pow(ekf_filter.state[2] - gps_local_z, 2));

            printf("\n========== t = %d s ==========\n", (int)(step_count * dt));
            printf("Pos: (%.2f, %.2f, %.2f) -> Goal: (%.2f, %.2f) | Cost: %.0f\n", 
                   cx, cy, cz, goal_x, goal_y, map_cost);
            
            printf("[EKF] X=%.3f Y=%.3f Z=%.3f | Roll=%.1f° Pitch=%.1f° Yaw=%.1f°\n",
                   ekf_filter.state[0], ekf_filter.state[1], ekf_filter.state[2],
                   ekf_filter.state[3] * 180/M_PI, ekf_filter.state[4] * 180/M_PI,
                   ekf_filter.state[5] * 180/M_PI);
            
            printf("[Covariance] X=%.6f Y=%.6f Z=%.6f\n",
                   ekf_filter.covariance[0][0], ekf_filter.covariance[1][1],
                   ekf_filter.covariance[2][2]);
            
            printf("[Error] 2D=%.3fm, 3D=%.3fm\n", dist_err_2d, dist_err_3d);
            
            // 3D Mapping stats
            mapping_print_stats();
            
            printf("=====================================\n");
        }
    }
    
    // Cleanup
    printf("\n[Shutdown] Exporting maps...\n");
    mapping_export_pgm("traversability_map.pgm");
    mapping_print_stats();
    mapping_cleanup();
    
    printf("[Shutdown] Done.\n");
    wb_robot_cleanup();
    return 0;
}
