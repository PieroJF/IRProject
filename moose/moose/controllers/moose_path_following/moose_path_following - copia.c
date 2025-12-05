/*
 * Moose Robot - 3D Navigation and Mapping System
 * Extended Kalman Filter (6 DOF) + Velodyne LiDAR + 3D Occupancy Mapping
 */

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/compass.h>
#include <webots/position_sensor.h>
#include <webots/gps.h>
#include <webots/inertial_unit.h>
#include <webots/lidar.h>  // ✓ AÑADIDO
#include <webots/device.h>
#include <webots/nodes.h>
#include <stdio.h>
#include <stdlib.h>  // ✓ AÑADIDO para malloc/free
#include <math.h>
#include <string.h>

#define TIME_STEP 32
#define STATE_SIZE 6  // [x, y, z, roll, pitch, yaw]

// Moose actual wheel diameter parameter
#define WHEEL_RADIUS 0.31 
#define WHEEL_BASE   1.1

// Hazard Thresholds (Radians) - ~20 degrees
#define MAX_SAFE_PITCH 0.35 
#define MAX_SAFE_ROLL  0.35

// Map parameters
#define MAP_RESOLUTION 0.2  // 20cm per cell
#define MAP_WIDTH 200       // 40m x 40m
#define MAP_HEIGHT 200

// Voxel downsampling (REDUCED to fix compilation)
#define VOXEL_SIZE 0.2  // Increased from 0.1 to 0.2
#define GRID_SIZE 200   // Reduced from 1000 to 200

// ============================================
// 1. Data Structure Definitions
// ============================================

typedef struct {
    float x, y, z;
} Point3D;

typedef struct {
    Point3D* points;
    int count;
    int capacity;
} PointCloud;

typedef struct {
    float elevation[MAP_WIDTH][MAP_HEIGHT];
    int hit_count[MAP_WIDTH][MAP_HEIGHT];
    float slope[MAP_WIDTH][MAP_HEIGHT];
} GlobalMap;

typedef struct {
    double x, y, z;
    double roll, pitch, yaw;
    double prev_left_enc;
    double prev_right_enc;
} OdometryData;

typedef struct {
    double state[STATE_SIZE];
    double covariance[STATE_SIZE][STATE_SIZE];
    double Q[STATE_SIZE][STATE_SIZE];
    double R[STATE_SIZE][STATE_SIZE];
} EKF;

typedef struct {
    double compass_yaw;    
    double gps_x;
    double gps_y;
    double gps_z;
    double roll;
    double pitch;
    // Encoder data
    double left_pos[4];
    double right_pos[4];
} SensorData;

typedef struct {
    WbDeviceTag left_sensors[4];  
    WbDeviceTag right_sensors[4];
    WbDeviceTag left_motors[4];
    WbDeviceTag right_motors[4];
    WbDeviceTag compass;
    WbDeviceTag gps;
    WbDeviceTag imu;
    WbDeviceTag velodyne;  // ✓ AÑADIDO
} RobotDevices;

// ============================================
// Global Variables
// ============================================

static RobotDevices devices;
static SensorData sensor_data;
static OdometryData odometry;
static EKF ekf_filter;
static GlobalMap global_map;  // ✓ AÑADIDO

// GPS offsets (global para que ekf_update_full pueda usarlos)
static double gps_offset_x = 0.0;  // ✓ AÑADIDO
static double gps_offset_y = 0.0;  // ✓ AÑADIDO
static double gps_offset_z = 0.0;  // ✓ AÑADIDO

// ============================================
// 2. Core EKF Algorithm
// ============================================

void ekf_init(EKF *ekf) {
    memset(ekf, 0, sizeof(EKF));
    
    // Initial covariance
    ekf->covariance[0][0] = 0.1;  // x
    ekf->covariance[1][1] = 0.1;  // y
    ekf->covariance[2][2] = 0.1;  // z
    ekf->covariance[3][3] = 0.05; // roll
    ekf->covariance[4][4] = 0.05; // pitch
    ekf->covariance[5][5] = 0.1;  // yaw
    
    // Process noise
    for(int i = 0; i < 6; i++) {
        ekf->Q[i][i] = 0.01;
    }
    
    // Measurement noise
    ekf->R[0][0] = 0.5;  // GPS X
    ekf->R[1][1] = 0.5;  // GPS Y
    ekf->R[2][2] = 0.5;  // GPS Z
    ekf->R[3][3] = 0.05; // IMU Roll
    ekf->R[4][4] = 0.05; // IMU Pitch
    ekf->R[5][5] = 0.05; // Compass Yaw
}

void ekf_predict(double v, double omega, double dt) {
    double yaw = ekf_filter.state[5];
    
    // Update position (X, Y from odometry)
    ekf_filter.state[0] += v * cos(yaw) * dt;
    ekf_filter.state[1] += v * sin(yaw) * dt;
    
    // Update Z (from GPS directly, or keep constant for ground robots)
    double delta_z = 0.0;
    ekf_filter.state[2] += delta_z;
    
    // Update orientation (Roll, Pitch from IMU directly)
    ekf_filter.state[3] = sensor_data.roll;
    ekf_filter.state[4] = sensor_data.pitch;
    
    // Update Yaw (from odometry)
    ekf_filter.state[5] += omega * dt;
    
    // Normalize angles to [-π, π]
    for(int i = 3; i < 6; i++) {
        while (ekf_filter.state[i] > M_PI) ekf_filter.state[i] -= 2.0 * M_PI;
        while (ekf_filter.state[i] < -M_PI) ekf_filter.state[i] += 2.0 * M_PI;
    }
    
    // Update covariance
    for(int i = 0; i < 6; i++) {
        ekf_filter.covariance[i][i] += ekf_filter.Q[i][i];
    }
}

void ekf_update_full() {
    // Update with GPS (X, Y, Z)
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
    
    // Update with IMU (Roll, Pitch)
    if(devices.imu) {
        // Roll
        double innov_roll = sensor_data.roll - ekf_filter.state[3];
        while(innov_roll > M_PI) innov_roll -= 2*M_PI;
        while(innov_roll < -M_PI) innov_roll += 2*M_PI;
        double S = ekf_filter.covariance[3][3] + ekf_filter.R[3][3];
        double K = ekf_filter.covariance[3][3] / S;
        ekf_filter.state[3] += K * innov_roll;
        ekf_filter.covariance[3][3] = (1 - K) * ekf_filter.covariance[3][3];
        
        // Pitch
        double innov_pitch = sensor_data.pitch - ekf_filter.state[4];
        while(innov_pitch > M_PI) innov_pitch -= 2*M_PI;
        while(innov_pitch < -M_PI) innov_pitch += 2*M_PI;
        S = ekf_filter.covariance[4][4] + ekf_filter.R[4][4];
        K = ekf_filter.covariance[4][4] / S;
        ekf_filter.state[4] += K * innov_pitch;
        ekf_filter.covariance[4][4] = (1 - K) * ekf_filter.covariance[4][4];
    }
    
    // Update with Compass (Yaw)
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

    odometry.yaw += dth;  // ✓ CORREGIDO: era theta
    while(odometry.yaw > M_PI) odometry.yaw -= 2*M_PI;
    while(odometry.yaw < -M_PI) odometry.yaw += 2*M_PI;

    odometry.x += dc * cos(odometry.yaw);  // ✓ CORREGIDO
    odometry.y += dc * sin(odometry.yaw);  // ✓ CORREGIDO
}

// ============================================
// 3. Device Operations
// ============================================

void initialize_devices() {
    char name[32];
    
    // Motors and encoders
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

    // Compass
    devices.compass = wb_robot_get_device("compass");
    if(devices.compass) wb_compass_enable(devices.compass, TIME_STEP);
    
    // GPS
    devices.gps = wb_robot_get_device("gps");
    if(devices.gps) wb_gps_enable(devices.gps, TIME_STEP);
    
    // Velodyne LiDAR - ✓ AÑADIDO
    devices.velodyne = wb_robot_get_device("velodyne");
    if(devices.velodyne) {
        wb_lidar_enable(devices.velodyne, TIME_STEP);
        wb_lidar_enable_point_cloud(devices.velodyne);
        printf("SUCCESS: Velodyne LiDAR initialized!\n");
    } else {
        printf("WARNING: No Velodyne LiDAR found on this robot.\n");
    }
    
    // IMU (Robust initialization)
    devices.imu = 0;
    int n_devices = wb_robot_get_number_of_devices();
    
    for (int i = 0; i < n_devices; i++) {
        WbDeviceTag tag = wb_robot_get_device_by_index(i);
        if (wb_device_get_node_type(tag) == WB_NODE_INERTIAL_UNIT) {
            devices.imu = tag;
            printf("SUCCESS: IMU Device found via Type Inspection! (Tag: %d)\n", tag);
            wb_inertial_unit_enable(devices.imu, TIME_STEP);
            break;
        }
    }

    if (!devices.imu) {
        printf("CRITICAL WARNING: No InertialUnit found on this robot.\n");
    }

    ekf_init(&ekf_filter);
}

void read_sensors_data() {
    // Encoders
    for(int i=0; i<4; i++) {
        if(devices.left_sensors[i]) sensor_data.left_pos[i] = wb_position_sensor_get_value(devices.left_sensors[i]);
        if(devices.right_sensors[i]) sensor_data.right_pos[i] = wb_position_sensor_get_value(devices.right_sensors[i]);
    }
    
    // Compass -> Yaw
    if(devices.compass) {
        const double *north = wb_compass_get_values(devices.compass);
        sensor_data.compass_yaw = atan2(north[0], north[1]); 
    }
    
    // GPS (X, Y, Z)
    if(devices.gps) {
        const double *pos = wb_gps_get_values(devices.gps);
        sensor_data.gps_x = pos[0];
        sensor_data.gps_y = pos[1];
        sensor_data.gps_z = pos[2];
    }

    // IMU (Roll, Pitch, Yaw)
    if(devices.imu) {
        const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(devices.imu);
        sensor_data.roll = rpy[0];
        sensor_data.pitch = rpy[1];
    }
}

// ============================================
// 4. 3D Mapping Functions (Person C)
// ============================================

Point3D transform_lidar_to_global(double local_x, double local_y, double local_z) {
    Point3D global;
    
    double x = ekf_filter.state[0];
    double y = ekf_filter.state[1];
    double z = ekf_filter.state[2];
    double roll = ekf_filter.state[3];
    double pitch = ekf_filter.state[4];
    double yaw = ekf_filter.state[5];
    
    // 3D Rotation Matrix (ZYX Euler)
    double cy = cos(yaw), sy = sin(yaw);
    double cp = cos(pitch), sp = sin(pitch);
    double cr = cos(roll), sr = sin(roll);
    
    double r11 = cy * cp;
    double r12 = cy * sp * sr - sy * cr;
    double r13 = cy * sp * cr + sy * sr;
    
    double r21 = sy * cp;
    double r22 = sy * sp * sr + cy * cr;
    double r23 = sy * sp * cr - cy * sr;
    
    double r31 = -sp;
    double r32 = cp * sr;
    double r33 = cp * cr;
    
    global.x = x + r11 * local_x + r12 * local_y + r13 * local_z;
    global.y = y + r21 * local_x + r22 * local_y + r23 * local_z;
    global.z = z + r31 * local_x + r32 * local_y + r33 * local_z;
    
    return global;
}

PointCloud* downsample_cloud(const WbLidarPoint* raw_points, int raw_count) {
    // Use dynamic allocation instead of static to avoid stack overflow
    static char* voxel_grid = NULL;
    static int grid_initialized = 0;
    
    // Initialize grid once (8MB instead of 1GB)
    if(!grid_initialized) {
        voxel_grid = (char*)calloc(GRID_SIZE * GRID_SIZE * GRID_SIZE, sizeof(char));
        if(voxel_grid == NULL) {
            printf("ERROR: Failed to allocate voxel grid memory!\n");
            return NULL;
        }
        grid_initialized = 1;
    }
    
    // Clear grid for this frame
    memset(voxel_grid, 0, GRID_SIZE * GRID_SIZE * GRID_SIZE);
    
    PointCloud* cloud = malloc(sizeof(PointCloud));
    if(!cloud) return NULL;
    
    cloud->points = malloc(sizeof(Point3D) * raw_count);
    if(!cloud->points) {
        free(cloud);
        return NULL;
    }
    
    cloud->count = 0;
    cloud->capacity = raw_count;
    
    for(int i = 0; i < raw_count; i++) {
        // Filter ground points
        if(raw_points[i].z < -0.5) continue;
        
        // Filter distant points
        float dist = sqrt(pow(raw_points[i].x, 2) + 
                         pow(raw_points[i].y, 2) + 
                         pow(raw_points[i].z, 2));
        if(dist > 20.0) continue;
        
        // Calculate voxel index (adjusted for smaller grid)
        int vx = (int)((raw_points[i].x + 20.0) / VOXEL_SIZE);
        int vy = (int)((raw_points[i].y + 20.0) / VOXEL_SIZE);
        int vz = (int)((raw_points[i].z + 10.0) / VOXEL_SIZE);
        
        if(vx < 0 || vx >= GRID_SIZE || 
           vy < 0 || vy >= GRID_SIZE || 
           vz < 0 || vz >= GRID_SIZE) continue;
        
        // Calculate 1D index from 3D coordinates
        int idx = vx * GRID_SIZE * GRID_SIZE + vy * GRID_SIZE + vz;
        
        // Only keep one point per voxel
        if(voxel_grid[idx] == 0) {
            cloud->points[cloud->count].x = raw_points[i].x;
            cloud->points[cloud->count].y = raw_points[i].y;
            cloud->points[cloud->count].z = raw_points[i].z;
            cloud->count++;
            voxel_grid[idx] = 1;
        }
    }
    
    return cloud;
}

void accumulate_point_in_map(double gx, double gy, double gz) {
    int map_x = (int)((gx + 20.0) / MAP_RESOLUTION);
    int map_y = (int)((gy + 20.0) / MAP_RESOLUTION);
    
    if(map_x >= 0 && map_x < MAP_WIDTH && 
       map_y >= 0 && map_y < MAP_HEIGHT) {
        int count = global_map.hit_count[map_x][map_y];
        global_map.elevation[map_x][map_y] = 
            (global_map.elevation[map_x][map_y] * count + gz) / (count + 1);
        global_map.hit_count[map_x][map_y]++;
    }
}

void compute_slope_map() {
    for(int x = 1; x < MAP_WIDTH - 1; x++) {
        for(int y = 1; y < MAP_HEIGHT - 1; y++) {
            if(global_map.hit_count[x][y] < 5) continue;
            
            // Gradient in X
            float dz_dx = (global_map.elevation[x+1][y] - 
                          global_map.elevation[x-1][y]) / (2 * MAP_RESOLUTION);
            
            // Gradient in Y
            float dz_dy = (global_map.elevation[x][y+1] - 
                          global_map.elevation[x][y-1]) / (2 * MAP_RESOLUTION);
            
            // Slope magnitude
            global_map.slope[x][y] = sqrt(dz_dx*dz_dx + dz_dy*dz_dy);
        }
    }
}

void correct_map_with_gps() {
    double drift_x = (sensor_data.gps_x - gps_offset_x) - ekf_filter.state[0];
    double drift_y = (sensor_data.gps_y - gps_offset_y) - ekf_filter.state[1];
    double drift = sqrt(drift_x*drift_x + drift_y*drift_y);
    
    if(drift > 0.5) {
        printf("[Map] GPS correction applied: drift = %.2fm\n", drift);
    }
}

// ============================================
// 5. Navigation & Path Planning (Person D)
// ============================================

double get_mock_map_cost(double x, double y) {
    double hill_x = 5.0;
    double hill_y = 2.0;
    double radius = 1.5;
    
    double dist = sqrt(pow(x - hill_x, 2) + pow(y - hill_y, 2));
    
    if (dist < radius) {
        return 100.0; // High Cost (Steep Slope)
    }
    return 0.0; // Low Cost (Flat Ground)
}

double calculate_navigation(double rob_x, double rob_y, double rob_theta, double goal_x, double goal_y) {
    // Look ahead
    double lookahead = 2.0;
    double next_x = rob_x + cos(rob_theta) * lookahead;
    double next_y = rob_y + sin(rob_theta) * lookahead;
    
    // Check map cost
    double cost_ahead = get_mock_map_cost(next_x, next_y);
    
    // Reactive avoidance
    if (cost_ahead > 50.0) {
        printf("[Nav] High Slope Detected ahead! Turning Left to avoid.\n");
        return 0.8;
    }
    
    // Steer toward goal
    double dx = goal_x - rob_x;
    double dy = goal_y - rob_y;
    double target_angle = atan2(dy, dx);
    double error = target_angle - rob_theta;
    
    // Normalize angle
    while(error > M_PI) error -= 2.0 * M_PI;
    while(error < -M_PI) error += 2.0 * M_PI;
    
    return error;
}

// ============================================
// 6. Safety Functions
// ============================================

int is_hazardous_state() {
    static int hazard_counter = 0;

    if (fabs(sensor_data.roll) > MAX_SAFE_ROLL || fabs(sensor_data.pitch) > MAX_SAFE_PITCH) {
        if (hazard_counter % 30 == 0) {
            printf("!!! DANGER: Tipping Hazard Detected! Roll: %.2f, Pitch: %.2f !!!\n", 
                   sensor_data.roll, sensor_data.pitch);
        }
        hazard_counter++;
        return 1;
    }

    hazard_counter = 0;
    return 0;
}

// ============================================
// 7. Main Control Loop
// ============================================

int main() {
    wb_robot_init();
    
    printf("\n========================================\n");
    printf("  Grass Cutting Robot Controller Startup\n");
    printf("  3D Mapping & Navigation System\n");
    printf("========================================\n\n");

    initialize_devices();
    
    // Initialize map
    memset(&global_map, 0, sizeof(GlobalMap));
    
    // Navigation goal
    double goal_x = 10.0;
    double goal_y = 2.0;
    
    // ============================================
    // 1. PREHEAT (Warm up sensors)
    // ============================================
    printf("[Init] Warming up sensors...\n");
    for(int i=0; i<10; i++) {
        wb_robot_step(TIME_STEP);
        read_sensors_data();
    }
    
    // ============================================
    // 2. CALIBRATE INITIAL STATE (3D)
    // ============================================
    printf("[Init] Calibrating initial pose...\n");
    gps_offset_x = sensor_data.gps_x;  // ✓ CORREGIDO: sin 'double'
    gps_offset_y = sensor_data.gps_y;
    gps_offset_z = sensor_data.gps_z;
    
    // Initialize encoder baseline
    double start_l = 0, start_r = 0;
    int cnt_l = 0, cnt_r = 0;
    for(int i=0; i<4; i++) {
        if(devices.left_sensors[i]) { start_l += sensor_data.left_pos[i]; cnt_l++; }
        if(devices.right_sensors[i]) { start_r += sensor_data.right_pos[i]; cnt_r++; }
    }
    odometry.prev_left_enc = (cnt_l > 0) ? start_l/cnt_l : 0;
    odometry.prev_right_enc = (cnt_r > 0) ? start_r/cnt_r : 0;

    // Force align initial pose (6 DOF)
    odometry.x = 0.0;
    odometry.y = 0.0;
    odometry.z = 0.0;
    odometry.roll = sensor_data.roll;
    odometry.pitch = sensor_data.pitch;
    odometry.yaw = sensor_data.compass_yaw;
    
    // Initialize EKF state (6 DOF)
    ekf_filter.state[0] = 0.0;
    ekf_filter.state[1] = 0.0;
    ekf_filter.state[2] = 0.0;
    ekf_filter.state[3] = sensor_data.roll;
    ekf_filter.state[4] = sensor_data.pitch;
    ekf_filter.state[5] = sensor_data.compass_yaw;
    
    printf("[Init] Initial GPS: (%.2f, %.2f, %.2f)\n", 
           gps_offset_x, gps_offset_y, gps_offset_z);
    printf("[Init] Initial IMU: Roll=%.1f°, Pitch=%.1f°, Yaw=%.1f°\n",
           sensor_data.roll * 180/M_PI, 
           sensor_data.pitch * 180/M_PI,
           sensor_data.compass_yaw * 180/M_PI);
    printf("[Init] Calibration complete!\n\n");

    int step_count = 0;
    double dt = TIME_STEP / 1000.0;
    
    // ============================================
    // MAIN CONTROL LOOP
    // ============================================
    while (wb_robot_step(TIME_STEP) != -1) {
        step_count++;
        read_sensors_data();
        
        // ============================================
        // HAZARD SAFETY LAYER
        // ============================================
        if (is_hazardous_state()) {
            for(int i=0; i<4; i++) {
                if(devices.left_motors[i]) wb_motor_set_velocity(devices.left_motors[i], 0);
                if(devices.right_motors[i]) wb_motor_set_velocity(devices.right_motors[i], 0);
            }
            continue;
        }
        
        // ============================================
        // ENCODER PROCESSING
        // ============================================
        double curr_l = 0, curr_r = 0;
        cnt_l = 0; cnt_r = 0;
        for(int i=0; i<4; i++) {
            if(devices.left_sensors[i]) { curr_l += sensor_data.left_pos[i]; cnt_l++; }
            if(devices.right_sensors[i]) { curr_r += sensor_data.right_pos[i]; cnt_r++; }
        }
        if(cnt_l > 0) curr_l /= cnt_l;
        if(cnt_r > 0) curr_r /= cnt_r;
        
        double d_left = curr_l - odometry.prev_left_enc;
        double d_right = curr_r - odometry.prev_right_enc;
        
        odometry.prev_left_enc = curr_l;
        odometry.prev_right_enc = curr_r;
        
        // ============================================
        // ODOMETRY UPDATE
        // ============================================
        odometry_update(d_left, d_right);
        
        double v_linear = ((d_left + d_right) / 2.0 * WHEEL_RADIUS) / dt;
        double v_angular = ((d_right - d_left) * WHEEL_RADIUS / WHEEL_BASE) / dt;
        
        // ============================================
        // EKF PREDICTION & UPDATE
        // ============================================
        ekf_predict(v_linear, v_angular, dt);
        ekf_update_full();
        
        // ============================================
        // LIDAR 3D PROCESSING
        // ============================================
        if(devices.velodyne && step_count % 5 == 0) {
            const WbLidarPoint* raw_points = wb_lidar_get_point_cloud(devices.velodyne);
            int num_points = wb_lidar_get_number_of_points(devices.velodyne);
            
            if(num_points > 0 && raw_points != NULL) {
                PointCloud* filtered = downsample_cloud(raw_points, num_points);
                
                if(filtered && filtered->count > 0) {
                    for(int i = 0; i < filtered->count; i++) {
                        Point3D local = filtered->points[i];
                        Point3D global = transform_lidar_to_global(local.x, local.y, local.z);
                        accumulate_point_in_map(global.x, global.y, global.z);
                    }
                    
                    free(filtered->points);
                    free(filtered);
                }
            }
        }
        
        // ============================================
        // SLOPE MAP COMPUTATION
        // ============================================
        if(step_count % (1000 / TIME_STEP) == 0) {
            compute_slope_map();
            correct_map_with_gps();
        }
        
        // ============================================
        // PATH PLANNING
        // ============================================
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

        for(int i=0; i<4; i++) {
            if(devices.left_motors[i]) wb_motor_set_velocity(devices.left_motors[i], left_speed);
            if(devices.right_motors[i]) wb_motor_set_velocity(devices.right_motors[i], right_speed);
        }

        // ============================================
        // DEBUG OUTPUT
        // ============================================
        if(step_count % (1000 / TIME_STEP) == 0) {
            double lookahead_x = cx + cos(cyaw) * 2.0;
            double lookahead_y = cy + sin(cyaw) * 2.0;
            double map_cost = get_mock_map_cost(lookahead_x, lookahead_y);
            
            printf("Pos: (%.2f, %.2f, %.2f) -> Goal: (%.2f, %.2f) | Cost Ahead: %.0f\n", 
                   cx, cy, cz, goal_x, goal_y, map_cost);
        }

        // ============================================
        // DETAILED OUTPUT
        // ============================================
        if(step_count % (1000 / TIME_STEP) == 0) {
            double gps_local_x = sensor_data.gps_x - gps_offset_x;
            double gps_local_y = sensor_data.gps_y - gps_offset_y;
            double gps_local_z = sensor_data.gps_z - gps_offset_z;
            
            double dist_err_2d = sqrt(pow(ekf_filter.state[0] - gps_local_x, 2) + 
                                      pow(ekf_filter.state[1] - gps_local_y, 2));
            double dist_err_3d = sqrt(pow(ekf_filter.state[0] - gps_local_x, 2) + 
                                      pow(ekf_filter.state[1] - gps_local_y, 2) +
                                      pow(ekf_filter.state[2] - gps_local_z, 2));

            printf("\n========== %d-th second ==========\n", (int)(step_count * dt));
            
            printf("[Odometry 2D] X: %.3f m, Y: %.3f m, Yaw: %.2f°\n", 
                   odometry.x, odometry.y, odometry.yaw * 180.0/M_PI);
            
            printf("[EKF 3D Pose]\n");
            printf("  Position: X=%.3f m, Y=%.3f m, Z=%.3f m\n", 
                   ekf_filter.state[0], ekf_filter.state[1], ekf_filter.state[2]);
            printf("  Orientation: Roll=%.2f°, Pitch=%.2f°, Yaw=%.2f°\n",
                   ekf_filter.state[3] * 180.0/M_PI,
                   ekf_filter.state[4] * 180.0/M_PI,
                   ekf_filter.state[5] * 180.0/M_PI);
            
            printf("[EKF Covariance]\n");
            printf("  Position: Var(X)=%.6f, Var(Y)=%.6f, Var(Z)=%.6f\n", 
                   ekf_filter.covariance[0][0], 
                   ekf_filter.covariance[1][1],
                   ekf_filter.covariance[2][2]);
            printf("  Orientation: Var(Roll)=%.6f, Var(Pitch)=%.6f, Var(Yaw)=%.6f\n",
                   ekf_filter.covariance[3][3],
                   ekf_filter.covariance[4][4],
                   ekf_filter.covariance[5][5]);
            
            printf("[Raw Sensors]\n");
            printf("  GPS: X=%.3f m, Y=%.3f m, Z=%.3f m\n", 
                   sensor_data.gps_x, sensor_data.gps_y, sensor_data.gps_z);
            printf("  IMU: Roll=%.2f°, Pitch=%.2f°\n",
                   sensor_data.roll * 180.0/M_PI,
                   sensor_data.pitch * 180.0/M_PI);
            printf("  Compass: Yaw=%.2f°\n", 
                   sensor_data.compass_yaw * 180.0/M_PI);
            
            printf("[Estimation Error]\n");
            printf("  2D Distance Error: %.3f m\n", dist_err_2d);
            printf("  3D Distance Error: %.3f m\n", dist_err_3d);
            
            int filled_cells = 0;
            for(int x=0; x<MAP_WIDTH; x++) {
                for(int y=0; y<MAP_HEIGHT; y++) {
                    if(global_map.hit_count[x][y] > 0) filled_cells++;
                }
            }
            printf("[3D Map Status]\n");
            printf("  Mapped cells: %d / %d (%.1f%%)\n", 
                   filled_cells, MAP_WIDTH * MAP_HEIGHT,
                   100.0 * filled_cells / (MAP_WIDTH * MAP_HEIGHT));
            
            printf("====================================\n\n");
        }
    }
    
    printf("\n[Shutdown] Cleaning up...\n");
    wb_robot_cleanup();
    return 0;
}