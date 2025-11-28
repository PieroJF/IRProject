/*

 */

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/compass.h>
#include <webots/position_sensor.h>
#include <webots/gps.h>
#include <webots/inertial_unit.h>
#include <webots/device.h>
#include <webots/nodes.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define TIME_STEP 32
#define STATE_SIZE 3

// Moose actual wheel diameter parameter
#define WHEEL_RADIUS 0.31 
#define WHEEL_BASE   1.1
// [IK] Hazard Thresholds (Radians) - ~20 degrees
#define MAX_SAFE_PITCH 0.35 
#define MAX_SAFE_ROLL  0.35

// ============================================
// 1. Data Structure Definition
// ============================================

typedef struct {
    double x, y, theta;
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
    // [IK] Added Roll/Pitch for Hazard Detection
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
    WbDeviceTag imu; // [IK] Added IMU device tag
} RobotDevices;

// ============================================
// Global Variable
// ============================================

static RobotDevices devices;
static SensorData sensor_data;
static OdometryData odometry;
static EKF ekf_filter;

// ============================================
// Core Algorithm
// ============================================

// Initial EKF
void ekf_init(EKF *ekf) {
    memset(ekf, 0, sizeof(EKF));
    // Initial covariance
    ekf->covariance[0][0] = 0.1; 
    ekf->covariance[1][1] = 0.1; 
    ekf->covariance[2][2] = 0.1;
    // Noise parameter
    ekf->Q[0][0] = 0.01; ekf->Q[1][1] = 0.01; ekf->Q[2][2] = 0.01;
    ekf->R[2][2] = 0.05; 
}

// EKF prediction
void ekf_predict(double v, double omega, double dt) {
    double theta = ekf_filter.state[2];
    ekf_filter.state[0] += v * cos(theta) * dt;
    ekf_filter.state[1] += v * sin(theta) * dt;
    ekf_filter.state[2] += omega * dt;
    
    // Angle normalization
    while (ekf_filter.state[2] > M_PI) ekf_filter.state[2] -= 2.0 * M_PI;
    while (ekf_filter.state[2] < -M_PI) ekf_filter.state[2] += 2.0 * M_PI;
    
    // Simple covariance update
    for(int i=0; i<3; i++) ekf_filter.covariance[i][i] += ekf_filter.Q[i][i];
}

// EKF update
void ekf_update(double z_yaw) {
    double theta = ekf_filter.state[2];
    double y = z_yaw - theta;
    
    while(y > M_PI) y -= 2*M_PI;
    while(y < -M_PI) y += 2*M_PI;

    double S = ekf_filter.covariance[2][2] + ekf_filter.R[2][2];
    double K = ekf_filter.covariance[2][2] / S;

    ekf_filter.state[2] += K * y;
    ekf_filter.covariance[2][2] = (1 - K) * ekf_filter.covariance[2][2];
}

// Odometry update
void odometry_update(double d_left_rad, double d_right_rad) {
    double dl = d_left_rad * WHEEL_RADIUS;
    double dr = d_right_rad * WHEEL_RADIUS;
    double dc = (dl + dr) / 2.0;
    double dth = (dr - dl) / WHEEL_BASE;

    odometry.theta += dth;
    while(odometry.theta > M_PI) odometry.theta -= 2*M_PI;
    while(odometry.theta < -M_PI) odometry.theta += 2*M_PI;

    odometry.x += dc * cos(odometry.theta);
    odometry.y += dc * sin(odometry.theta);
}

// ============================================
// Device operation
// ============================================

void initialize_devices() {
    char name[32];
    
    // Obtain motors and sensors
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

    // Compass and GPS
    devices.compass = wb_robot_get_device("compass");
    if(devices.compass) wb_compass_enable(devices.compass, TIME_STEP);
    
    devices.gps = wb_robot_get_device("gps");
    if(devices.gps) wb_gps_enable(devices.gps, TIME_STEP);
    
    // 3. [IK] Robust IMU Initialization (Find by Type)
    devices.imu = 0; // Initialize to NULL
    int n_devices = wb_robot_get_number_of_devices();
    
    for (int i = 0; i < n_devices; i++) {
        WbDeviceTag tag = wb_robot_get_device_by_index(i);
        // Look for the specific hardware type "InertialUnit"
        if (wb_device_get_node_type(tag) == WB_NODE_INERTIAL_UNIT) {
            devices.imu = tag;
            printf("SUCCESS: IMU Device found via Type Inspection! (Tag: %d)\n", tag);
            wb_inertial_unit_enable(devices.imu, TIME_STEP);
            break; // Stop looking once found
        }
    }

    if (!devices.imu) {
        printf("CRITICAL WARNING: No InertialUnit found on this robot.\n");
    }

    ekf_init(&ekf_filter);
}

void read_sensors_data() {
    // 1. Encoder
    for(int i=0; i<4; i++) {
        if(devices.left_sensors[i]) sensor_data.left_pos[i] = wb_position_sensor_get_value(devices.left_sensors[i]);
        if(devices.right_sensors[i]) sensor_data.right_pos[i] = wb_position_sensor_get_value(devices.right_sensors[i]);
    }
    
    // 2. Compass -> Yaw
    if(devices.compass) {
        const double *north = wb_compass_get_values(devices.compass);
        sensor_data.compass_yaw = atan2(north[0], north[1]); 
    }
    
    // 3. GPS
    if(devices.gps) {
        const double *pos = wb_gps_get_values(devices.gps);
        sensor_data.gps_x = pos[0];
        sensor_data.gps_y = pos[1];
    }

    // [IK] 4. Read IMU (Roll/Pitch)
    if(devices.imu) {
        const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(devices.imu);
        // rpy[0] is roll (x-axis), rpy[1] is pitch (z-axis in Webots sometimes, but usually y)
        sensor_data.roll = rpy[0];
        sensor_data.pitch = rpy[1];
    }
}

// ============================================
// [IK] Person D: Navigation & Mapping Logic
// ============================================

// MMock Cost Map (Simulates Piero's Output)
// Place a "Virtual Hill" at (x=5.0, y=2.0) relative to the start.
// If the robot tries to go there, the cost is HIGH.
  double get_mock_map_cost(double x, double y) {
      double hill_x = 5.0;
      double hill_y = 2.0;
      double radius = 1.5; // The hill is 3 meters wide
      
      // Calculate distance to the virtual hill
      double dist = sqrt(pow(x - hill_x, 2) + pow(y - hill_y, 2));
      
      if (dist < radius) {
          return 100.0; // High Cost (Steep Slope / Rock)
      }
      return 0.0; // Low Cost (Flat Ground)
  }
  
  // Navigation Controller
  // Returns a "turn" value: Positive = Turn Left, Negative = Turn Right
  double calculate_navigation(double rob_x, double rob_y, double rob_theta, double goal_x, double goal_y) {
      // 1. Look Ahead (Project position 2 meters forward)
      double lookahead = 2.0;
      double next_x = rob_x + cos(rob_theta) * lookahead;
      double next_y = rob_y + sin(rob_theta) * lookahead;
      
      // 2. Check the Map Cost at that spot
      double cost_ahead = get_mock_map_cost(next_x, next_y);
      
      // 3. Reactive Avoidance
      if (cost_ahead > 50.0) {
          printf("[Nav] High Slope Detected ahead! Turning Left to avoid.\n");
          return 0.8; // Force a left turn
      }
      
      // 4. If clear, steer toward goal
      double dx = goal_x - rob_x;
      double dy = goal_y - rob_y;
      double target_angle = atan2(dy, dx);
      double error = target_angle - rob_theta;
      
      // Normalize angle to -PI to +PI
      while(error > M_PI) error -= 2.0 * M_PI;
      while(error < -M_PI) error += 2.0 * M_PI;
      
      return error; // P-Controller (Proportional to error)
  }

// ============================================
// [IK] Helper Functions
// ============================================

// Hazard Analysis
// Returns 1 if unsafe, 0 if safe
  int is_hazardous_state() {
      static int hazard_counter = 0; // 'static' keeps the value between function calls
  
      // Check if robot is tipping over
      if (fabs(sensor_data.roll) > MAX_SAFE_ROLL || fabs(sensor_data.pitch) > MAX_SAFE_PITCH) {
          
          // Only print every 30 steps (roughly once per second)
          if (hazard_counter % 30 == 0) {
              printf("!!! DANGER: Tipping Hazard Detected! Roll: %.2f, Pitch: %.2f !!!\n", 
                     sensor_data.roll, sensor_data.pitch);
          }
          
          hazard_counter++;
          return 1; // Always return 1 to cut power, even if we didn't print this time
      }
  
      hazard_counter = 0; // Reset counter when we are safe
      return 0;
  }

// ============================================
// Main program
// ============================================

int main() {
    wb_robot_init();
    
    printf("\n========================================\n");
    printf("  Grass Cutting Robot Controller Startup\n");
    printf("========================================\n\n");

    initialize_devices();
    
    // [IK] Navigation Goal: Go to X=10, Y=2
    // Note: The "Virtual Hill" is at (5,2), so it is directly in the way!
    double goal_x = 10.0;
    double goal_y = 2.0;
    
    // 1. preheat
    for(int i=0; i<10; i++) {
        wb_robot_step(TIME_STEP);
        read_sensors_data();
    }
    
    // 2. Calibrate initial state
    double gps_offset_x = sensor_data.gps_x;
    double gps_offset_y = sensor_data.gps_y;
    
    // Initialize encoder old value
    double start_l = 0, start_r = 0;
    int cnt_l = 0, cnt_r = 0;
    for(int i=0; i<4; i++) {
        if(devices.left_sensors[i]) { start_l += sensor_data.left_pos[i]; cnt_l++; }
        if(devices.right_sensors[i]) { start_r += sensor_data.right_pos[i]; cnt_r++; }
    }
    odometry.prev_left_enc = (cnt_l > 0) ? start_l/cnt_l : 0;
    odometry.prev_right_enc = (cnt_r > 0) ? start_r/cnt_r : 0;

    // Force align odometry heading
    odometry.theta = sensor_data.compass_yaw;
    ekf_filter.state[2] = sensor_data.compass_yaw;

    int step_count = 0;
    double dt = TIME_STEP / 1000.0;
    
    while (wb_robot_step(TIME_STEP) != -1) {
        step_count++;
        read_sensors_data(); // Gets IMU, GPS, Encoders
        
        // [IK] Hazard Safety Layer
        // If the IMU detects we are tipping over, CUT POWER immediately.
        if (is_hazardous_state()) {
            wb_motor_set_velocity(devices.left_motors[0], 0);
            wb_motor_set_velocity(devices.right_motors[0], 0);
            // (Simulate cutting all motors)
            continue; 
        }
        
        // --- Calculate average wheel position ---
        double curr_l = 0, curr_r = 0;
        cnt_l = 0; cnt_r = 0;
        for(int i=0; i<4; i++) {
            if(devices.left_sensors[i]) { curr_l += sensor_data.left_pos[i]; cnt_l++; }
            if(devices.right_sensors[i]) { curr_r += sensor_data.right_pos[i]; cnt_r++; }
        }
        if(cnt_l > 0) curr_l /= cnt_l;
        if(cnt_r > 0) curr_r /= cnt_r;
        
        // --- Algorithm update ---
        double d_left = curr_l - odometry.prev_left_enc;
        double d_right = curr_r - odometry.prev_right_enc;
        
        odometry.prev_left_enc = curr_l;
        odometry.prev_right_enc = curr_r;
        
        odometry_update(d_left, d_right);
        
        double v_linear = ((d_left + d_right) / 2.0 * WHEEL_RADIUS) / dt;
        double v_angular = ((d_right - d_left) * WHEEL_RADIUS / WHEEL_BASE) / dt;
        
        ekf_predict(v_linear, v_angular, dt);
        if(devices.compass) ekf_update(sensor_data.compass_yaw);
        
        // --- [IK] Path Planning Layer ---
        // 1. Get current EKF Pose
        double cx = ekf_filter.state[0];
        double cy = ekf_filter.state[1];
        double ctheta = ekf_filter.state[2];

        // 2. Calculate Steering
        double turn_cmd = calculate_navigation(cx, cy, ctheta, goal_x, goal_y);
        double fwd_cmd = 2.0; // Base speed

        // 3. Apply to Motors (Differential Drive)
        double left_speed = fwd_cmd - turn_cmd;
        double right_speed = fwd_cmd + turn_cmd;

        // Clamp speeds
        if(left_speed > 5.0) left_speed = 5.0;
        if(right_speed > 5.0) right_speed = 5.0;

        for(int i=0; i<4; i++) {
            if(devices.left_motors[i]) wb_motor_set_velocity(devices.left_motors[i], left_speed);
            if(devices.right_motors[i]) wb_motor_set_velocity(devices.right_motors[i], right_speed);
        }

        // --- Debug Output (Every 1 second) ---
        if(step_count % (1000 / TIME_STEP) == 0) {
             printf("Pos: (%.2f, %.2f) -> Goal: (%.2f, %.2f) | Map Cost Ahead: %.0f\n", 
                    cx, cy, goal_x, goal_y, get_mock_map_cost(cx + cos(ctheta)*2, cy + sin(ctheta)*2));
        }

        // --- Output according to specified format (per second) ---
        if(step_count % (1000 / TIME_STEP) == 0) {
            // Calculate true distance error (relative to EKF's local coordinate system)
            double gps_local_x = sensor_data.gps_x - gps_offset_x;
            double gps_local_y = sensor_data.gps_y - gps_offset_y;
            double dist_err = sqrt(pow(ekf_filter.state[0] - gps_local_x, 2) + 
                                   pow(ekf_filter.state[1] - gps_local_y, 2));

            printf("========== the %d-th second ==========\n", (int)(step_count * dt));
            printf("[Odometry] X: %.3f m, Y: %.3f m, Theta: %.2f°\n", 
                   odometry.x, odometry.y, odometry.theta * 180.0/M_PI);
            printf("[EKF Estimate] X: %.3f m, Y: %.3f m, Theta: %.2f°\n", 
                   ekf_filter.state[0], ekf_filter.state[1], ekf_filter.state[2] * 180.0/M_PI);
            printf("[EKF Covariance] Var(X): %.6f, Var(Y): %.6f, Var(Theta): %.6f\n", 
                   ekf_filter.covariance[0][0], ekf_filter.covariance[1][1], ekf_filter.covariance[2][2]);
          
            printf("[Raw IMU] Yaw: %.2f°\n", sensor_data.compass_yaw * 180.0/M_PI);
            printf("[GPS True Value] X: %.3f m, Y: %.3f m\n", sensor_data.gps_x, sensor_data.gps_y);
            printf("[Error] Distance Error: %.3f m\n", dist_err);
        }

    }
    
    wb_robot_cleanup();
    return 0;
}