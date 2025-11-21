/*
 * Moose Robot Controller
 * Responsible for: Robot model interface, motion control, sensor acquisition, odometer, EKF prediction
 */

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/compass.h>
#include <webots/position_sensor.h>
#include <webots/gps.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define TIME_STEP 32
#define STATE_SIZE 3

// Moose Real physical parameters
#define WHEEL_RADIUS 0.31 
#define WHEEL_BASE   1.1

// ============================================
// Data structure definition
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
} RobotDevices;

// ============================================
// global variable
// ============================================

static RobotDevices devices;
static SensorData sensor_data;
static OdometryData odometry;
static EKF ekf_filter;

// ============================================
// Core Algorithm (Odometer & Prediction))
// ============================================

// Initialize EKF (Base Matrix Settings)
void ekf_init(EKF *ekf) {
    memset(ekf, 0, sizeof(EKF));
    // Initial covariance
    ekf->covariance[0][0] = 0.1; 
    ekf->covariance[1][1] = 0.1; 
    ekf->covariance[2][2] = 0.1;
    // Noise parameter setting
    ekf->Q[0][0] = 0.01; ekf->Q[1][1] = 0.01; ekf->Q[2][2] = 0.01;
    ekf->R[2][2] = 0.05; 
}

// EKF Prediction steps (Prediction Step)
// The next moment state is estimated based on the kinematic model
void ekf_predict(double v, double omega, double dt) {
    double theta = ekf_filter.state[2];
    
    // State transition: X_k = f(X_{k-1}, u)
    ekf_filter.state[0] += v * cos(theta) * dt;
    ekf_filter.state[1] += v * sin(theta) * dt;
    ekf_filter.state[2] += omega * dt;
    
    // Angle normalization
    while (ekf_filter.state[2] > M_PI) ekf_filter.state[2] -= 2.0 * M_PI;
    while (ekf_filter.state[2] < -M_PI) ekf_filter.state[2] += 2.0 * M_PI;
    
    // Covariance Forecasting : P = P + Q
    for(int i=0; i<3; i++) ekf_filter.covariance[i][i] += ekf_filter.Q[i][i];
}

// Pure odometer update(Dead Reckoning)
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
// Hardware Interface & Acquisition 
// ============================================

void initialize_devices() {
    char name[32];
    
    // 1. Initialize the motor and encoder
    for (int i = 0; i < 4; i++) {
        // starboard
        sprintf(name, "right motor %d", i+1);
        devices.right_motors[i] = wb_robot_get_device(name);
        if(devices.right_motors[i]) {
            wb_motor_set_position(devices.right_motors[i], INFINITY);
            wb_motor_set_velocity(devices.right_motors[i], 0.0);
        }
        
        sprintf(name, "right wheel sensor %d", i+1);
        devices.right_sensors[i] = wb_robot_get_device(name);
        if(devices.right_sensors[i]) wb_position_sensor_enable(devices.right_sensors[i], TIME_STEP);

        // nearsidey
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

    // 2. Initialize the sensor (Compass & GPS)
    devices.compass = wb_robot_get_device("compass");
    if(devices.compass) wb_compass_enable(devices.compass, TIME_STEP);
    
    devices.gps = wb_robot_get_device("gps");
    if(devices.gps) wb_gps_enable(devices.gps, TIME_STEP);

    // Initialize the algorithm
    ekf_init(&ekf_filter);
}

// Collect the underlying data
void read_sensors_data() {
    // Read the original encoder value
    for(int i=0; i<4; i++) {
        if(devices.left_sensors[i]) sensor_data.left_pos[i] = wb_position_sensor_get_value(devices.left_sensors[i]);
        if(devices.right_sensors[i]) sensor_data.right_pos[i] = wb_position_sensor_get_value(devices.right_sensors[i]);
    }
    
    // Read the compass -> to convert to the Yaw angle
    if(devices.compass) {
        const double *north = wb_compass_get_values(devices.compass);
        sensor_data.compass_yaw = atan2(north[0], north[1]); 
    }
    
    // Read GPS raw coordinates
    if(devices.gps) {
        const double *pos = wb_gps_get_values(devices.gps);
        sensor_data.gps_x = pos[0];
        sensor_data.gps_y = pos[1];
    }
}

// ============================================
// main program
// ============================================

int main() {
    wb_robot_init();
    
    printf("---Infrastructure & Kinematics Started ---\n");

    initialize_devices();
    
    // warm-up
    for(int i=0; i<10; i++) {
        wb_robot_step(TIME_STEP);
        read_sensors_data();
    }
    
    // Initialize the old encoder value
    double start_l = 0, start_r = 0;
    int cnt_l = 0, cnt_r = 0;
    for(int i=0; i<4; i++) {
        if(devices.left_sensors[i]) { start_l += sensor_data.left_pos[i]; cnt_l++; }
        if(devices.right_sensors[i]) { start_r += sensor_data.right_pos[i]; cnt_r++; }
    }
    odometry.prev_left_enc = (cnt_l > 0) ? start_l/cnt_l : 0;
    odometry.prev_right_enc = (cnt_r > 0) ? start_r/cnt_r : 0;

    int step_count = 0;
    double dt = TIME_STEP / 1000.0;
    
    while (wb_robot_step(TIME_STEP) != -1) {
        step_count++;
        
        // 1. data acquisition
        read_sensors_data();
        
        // 2. Calculate the average wheel position increment
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
        
        // 3. Implement odometer measurements
        odometry_update(d_left, d_right);
        
        // 4. EKF prediction steps
        double v_linear = ((d_left + d_right) / 2.0 * WHEEL_RADIUS) / dt;
        double v_angular = ((d_right - d_left) * WHEEL_RADIUS / WHEEL_BASE) / dt;
        ekf_predict(v_linear, v_angular, dt);
        

        // 5. Base status print
        if(step_count % (1000 / TIME_STEP) == 0) {
            printf("========== Sec %d  ==========\n", (int)(step_count * dt));
            printf("[Odometry] X: %.3f, Y: %.3f, Th: %.2f\n", 
                   odometry.x, odometry.y, odometry.theta);
            printf("[EKF Predict] X: %.3f, Y: %.3f, Th: %.2f\n", 
                   ekf_filter.state[0], ekf_filter.state[1], ekf_filter.state[2]);
            printf("[Sensors] GPS Raw: (%.2f, %.2f), Compass: %.2f\n",
                   sensor_data.gps_x, sensor_data.gps_y, sensor_data.compass_yaw);
        }

        // 6. motor control
        double speed = 2.0;
        for(int i=0; i<4; i++) {
            if(devices.left_motors[i]) wb_motor_set_velocity(devices.left_motors[i], speed);
            if(devices.right_motors[i]) wb_motor_set_velocity(devices.right_motors[i], speed);
        }
    }
    
    wb_robot_cleanup();
    return 0;
}