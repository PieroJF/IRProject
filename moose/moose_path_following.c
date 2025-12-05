/*
 * 草地割草机器人控制器 - 集成里程计和卡尔曼滤波
 * 作者: Yichang Chao & Ran Zhang
 * 功能: 传感器融合、位置估计、自主导航
 */

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/inertial_unit.h>
#include <webots/position_sensor.h>
#include <webots/gps.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define TIME_STEP 32  // 控制循环时间步长（毫秒）
#define STATE_SIZE 3  // 状态向量大小 [x, y, theta]

// ============================================
// 数据结构定义
// ============================================

// 里程计数据结构
typedef struct {
    double wheel_radius;       // 轮子半径（米）
    double wheel_base;         // 两轮间距（米）
    double x, y, theta;        // 当前位置和方向
    double prev_left_encoder;  // 上一次左编码器值
    double prev_right_encoder; // 上一次右编码器值
} OdometryData;

// 扩展卡尔曼滤波器数据结构
typedef struct {
    double state[STATE_SIZE];           // 状态向量 [x, y, theta]
    double covariance[STATE_SIZE][STATE_SIZE];  // 协方差矩阵 P
    double Q[STATE_SIZE][STATE_SIZE];   // 过程噪声协方差
    double R[STATE_SIZE][STATE_SIZE];   // 测量噪声协方差
    double wheel_radius;
    double wheel_base;
} EKF;

// 传感器数据结构
typedef struct {
    double left_encoder;
    double right_encoder;
    double imu_roll;
    double imu_pitch;
    double imu_yaw;
    double prev_left_encoder;
    double prev_right_encoder;
} SensorData;

// 机器人硬件接口
typedef struct {
    WbDeviceTag left motor;
    WbDeviceTag right motor;
    WbDeviceTag left_encoder;
    WbDeviceTag right_encoder;
    WbDeviceTag imu;
    WbDeviceTag gps;  // 可选，用于验证
} RobotDevices;

// ============================================
// 全局变量
// ============================================

static RobotDevices devices;
static SensorData sensor_data;
static OdometryData odometry;
static EKF ekf_filter;

// ============================================
// 里程计函数实现
// ============================================

void odometry_init(OdometryData *odom, double wheel_radius, double wheel_base) {
    odom->wheel_radius = wheel_radius;
    odom->wheel_base = wheel_base;
    odom->x = 0.0;
    odom->y = 0.0;
    odom->theta = 0.0;
    odom->prev_left_encoder = 0.0;
    odom->prev_right_encoder = 0.0;
    printf("[里程计] 初始化完成 - 轮半径: %.3fm, 轮距: %.3fm\n", 
           wheel_radius, wheel_base);
}

void odometry_update(OdometryData *odom, double left_encoder, double right_encoder) {
    // 计算两轮转过的角度差（弧度）
    double d_left = left_encoder - odom->prev_left_encoder;
    double d_right = right_encoder - odom->prev_right_encoder;
    
    // 转换为线性距离（米）
    double dist_left = d_left * odom->wheel_radius;
    double dist_right = d_right * odom->wheel_radius;
    
    // 计算机器人中心走过的距离和转过的角度
    double dist = (dist_left + dist_right) / 2.0;
    double d_theta = (dist_right - dist_left) / odom->wheel_base;
    
    // 更新方向角
    odom->theta += d_theta;
    
    // 归一化角度到 [-π, π]
    while (odom->theta > M_PI) odom->theta -= 2.0 * M_PI;
    while (odom->theta < -M_PI) odom->theta += 2.0 * M_PI;
    
    // 更新位置（使用当前角度的平均值）
    double theta_avg = odom->theta - d_theta / 2.0;
    odom->x += dist * cos(theta_avg);
    odom->y += dist * sin(theta_avg);
    
    // 保存当前编码器值
    odom->prev_left_encoder = left_encoder;
    odom->prev_right_encoder = right_encoder;
}

void odometry_get_pose(OdometryData *odom, double *x, double *y, double *theta) {
    *x = odom->x;
    *y = odom->y;
    *theta = odom->theta;
}

// ============================================
// 矩阵运算辅助函数
// ============================================

void matrix_multiply_3x3(double A[3][3], double B[3][3], double result[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_add_3x3(double A[3][3], double B[3][3], double result[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
}

void matrix_transpose_3x3(double A[3][3], double result[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = A[j][i];
        }
    }
}

// ============================================
// 扩展卡尔曼滤波器实现
// ============================================

void ekf_init(EKF *ekf, double wheel_radius, double wheel_base) {
    // 初始化状态向量
    ekf->state[0] = 0.0;  // x
    ekf->state[1] = 0.0;  // y
    ekf->state[2] = 0.0;  // theta
    
    ekf->wheel_radius = wheel_radius;
    ekf->wheel_base = wheel_base;
    
    // 初始化协方差矩阵 P（初始不确定性）
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            ekf->covariance[i][j] = (i == j) ? 0.1 : 0.0;
        }
    }
    
    // 过程噪声协方差 Q（模型不确定性）
    // 这些值需要根据实际情况调整
    ekf->Q[0][0] = 0.01;  // x 位置噪声
    ekf->Q[0][1] = 0.0;
    ekf->Q[0][2] = 0.0;
    ekf->Q[1][0] = 0.0;
    ekf->Q[1][1] = 0.01;  // y 位置噪声
    ekf->Q[1][2] = 0.0;
    ekf->Q[2][0] = 0.0;
    ekf->Q[2][1] = 0.0;
    ekf->Q[2][2] = 0.02;  // 角度噪声
    
    // 测量噪声协方差 R（传感器不确定性）
    ekf->R[0][0] = 0.1;   // x 测量噪声
    ekf->R[0][1] = 0.0;
    ekf->R[0][2] = 0.0;
    ekf->R[1][0] = 0.0;
    ekf->R[1][1] = 0.1;   // y 测量噪声
    ekf->R[1][2] = 0.0;
    ekf->R[2][0] = 0.0;
    ekf->R[2][1] = 0.0;
    ekf->R[2][2] = 0.05;  // 角度测量噪声（IMU 比较准确）
    
    printf("[EKF] 初始化完成\n");
}

void ekf_predict(EKF *ekf, double v_left, double v_right, double dt) {
    // 计算线速度和角速度
    double v = (v_left + v_right) / 2.0 * ekf->wheel_radius;
    double omega = (v_right - v_left) / ekf->wheel_base * ekf->wheel_radius;
    
    double theta = ekf->state[2];
    
    // 状态预测方程
    ekf->state[0] += v * cos(theta) * dt;
    ekf->state[1] += v * sin(theta) * dt;
    ekf->state[2] += omega * dt;
    
    // 归一化角度
    while (ekf->state[2] > M_PI) ekf->state[2] -= 2.0 * M_PI;
    while (ekf->state[2] < -M_PI) ekf->state[2] += 2.0 * M_PI;
    
    // 计算雅可比矩阵 F（状态转移矩阵的线性化）
    double F[3][3] = {
        {1.0, 0.0, -v * sin(theta) * dt},
        {0.0, 1.0,  v * cos(theta) * dt},
        {0.0, 0.0,  1.0}
    };
    
    // 协方差预测: P = F*P*F' + Q
    double temp[3][3];
    matrix_multiply_3x3(F, ekf->covariance, temp);
    
    double F_transpose[3][3];
    matrix_transpose_3x3(F, F_transpose);
    
    double FPF_transpose[3][3];
    matrix_multiply_3x3(temp, F_transpose, FPF_transpose);
    
    matrix_add_3x3(FPF_transpose, ekf->Q, ekf->covariance);
}

void ekf_update(EKF *ekf, double imu_yaw) {
    // 测量向量 z（来自 IMU 的 yaw 角）
    double z_theta = imu_yaw;
    
    // 观测矩阵 H（我们只观测角度）
    double H[3][3] = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0}  // 只观测 theta
    };
    
    // 计算卡尔曼增益 K = P*H' * (H*P*H' + R)^-1
    // 简化版本：只更新 theta
    double S = ekf->covariance[2][2] + ekf->R[2][2];  // 创新协方差
    double K[3];  // 卡尔曼增益向量
    K[0] = ekf->covariance[0][2] / S;
    K[1] = ekf->covariance[1][2] / S;
    K[2] = ekf->covariance[2][2] / S;
    
    // 计算测量残差（创新）
    double innovation = z_theta - ekf->state[2];
    
    // 归一化角度差
    while (innovation > M_PI) innovation -= 2.0 * M_PI;
    while (innovation < -M_PI) innovation += 2.0 * M_PI;
    
    // 状态更新
    ekf->state[0] += K[0] * innovation;
    ekf->state[1] += K[1] * innovation;
    ekf->state[2] += K[2] * innovation;
    
    // 协方差更新 P = (I - K*H) * P
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ekf->covariance[i][j] -= K[i] * H[2][j] * ekf->covariance[2][j];
        }
    }
}

void ekf_get_state(EKF *ekf, double *x, double *y, double *theta) {
    *x = ekf->state[0];
    *y = ekf->state[1];
    *theta = ekf->state[2];
}

// ============================================
// 传感器初始化与读取
// ============================================

void initialize_devices() {
    // 初始化电机
    devices.left motor = wb_robot_get_device("left wheel motor");
    devices.right motor = wb_robot_get_device("right wheel motor");
    
    if (devices.left_motor == 0 || devices.right_motor == 0) {
        printf("[错误] 无法找到电机设备！\n");
        printf("请检查设备名称，常见名称：\n");
        printf("  - left wheel motor / right wheel motor\n");
        printf("  - left motor / right motor\n");
        printf("  - motor left / motor right\n");
    }
    
    // 设置电机为速度控制模式
    wb_motor_set_position(devices.left motor, INFINITY);
    wb_motor_set_position(devices.right motor, INFINITY);
    wb_motor_set_velocity(devices.left motor, 0.0);
    wb_motor_set_velocity(devices.right motor, 0.0);
    
    // 初始化编码器
    devices.left_encoder = wb_robot_get_device("left wheel sensor");
    devices.right_encoder = wb_robot_get_device("right wheel sensor");
    
    if (devices.left_encoder == 0 || devices.right_encoder == 0) {
        printf("[警告] 无法找到编码器设备！尝试其他名称...\n");
        devices.left_encoder = wb_robot_get_device("left_encoder");
        devices.right_encoder = wb_robot_get_device("right_encoder");
    }
    
    if (devices.left_encoder != 0 && devices.right_encoder != 0) {
        wb_position_sensor_enable(devices.left_encoder, TIME_STEP);
        wb_position_sensor_enable(devices.right_encoder, TIME_STEP);
        printf("[传感器] 编码器已启用\n");
    }
    
    // 初始化 IMU
    devices.imu = wb_robot_get_device("imu");
    if (devices.imu == 0) {
        devices.imu = wb_robot_get_device("inertial unit");
    }
    
    if (devices.imu != 0) {
        wb_inertial_unit_enable(devices.imu, TIME_STEP);
        printf("[传感器] IMU 已启用\n");
    } else {
        printf("[警告] 无法找到 IMU 设备\n");
    }
    
    // 初始化 GPS（可选，用于验证）
    devices.gps = wb_robot_get_device("gps");
    if (devices.gps != 0) {
        wb_gps_enable(devices.gps, TIME_STEP);
        printf("[传感器] GPS 已启用（用于验证）\n");
    }
    
    printf("[初始化] 所有设备初始化完成\n");
}

void read_sensors() {
    // 读取编码器
    if (devices.left_encoder != 0) {
        sensor_data.left_encoder = wb_position_sensor_get_value(devices.left_encoder);
        sensor_data.right_encoder = wb_position_sensor_get_value(devices.right_encoder);
    }
    
    // 读取 IMU
    if (devices.imu != 0) {
        const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(devices.imu);
        sensor_data.imu_roll = rpy[0];
        sensor_data.imu_pitch = rpy[1];
        sensor_data.imu_yaw = rpy[2];
    }
}

// ============================================
// 电机控制函数
// ============================================

void set_motor_speeds(double left_speed, double right_speed) {
    wb_motor_set_velocity(devices.left motor, left_speed);
    wb_motor_set_velocity(devices.right motor, right_speed);
}

// ============================================
// 主程序
// ============================================

int main(int argc, char **argv) {
    // 初始化 Webots API
    wb_robot_init();
    
    printf("\n========================================\n");
    printf("  草地割草机器人控制器启动\n");
    printf("  里程计 + 卡尔曼滤波传感器融合\n");
    printf("========================================\n\n");
    
    // 初始化硬件设备
    initialize_devices();
    
    // 初始化里程计
    // 注意：这些参数需要根据您的 MOONS 机器人实际尺寸调整
    double WHEEL_RADIUS = 0.025;  // 轮子半径 25mm = 0.025m
    double WHEEL_BASE = 0.088;    // 轮距 88mm = 0.088m
    
    odometry_init(&odometry, WHEEL_RADIUS, WHEEL_BASE);
    
    // 初始化 EKF
    ekf_init(&ekf_filter, WHEEL_RADIUS, WHEEL_BASE);
    
    // 等待第一帧传感器数据
    wb_robot_step(TIME_STEP);
    read_sensors();
    
    // 初始化编码器起始值
    sensor_data.prev_left_encoder = sensor_data.left_encoder;
    sensor_data.prev_right_encoder = sensor_data.right_encoder;
    odometry.prev_left_encoder = sensor_data.left_encoder;
    odometry.prev_right_encoder = sensor_data.right_encoder;
    
    printf("\n[启动] 进入主控制循环...\n\n");
    
    int step_count = 0;
    double dt = TIME_STEP / 1000.0;  // 转换为秒
    
    // 主控制循环
    while (wb_robot_step(TIME_STEP) != -1) {
        step_count++;
        
        // 读取所有传感器数据
        read_sensors();
        
        // ========== 里程计更新 ==========
        odometry_update(&odometry, sensor_data.left_encoder, sensor_data.right_encoder);
        
        // ========== EKF 预测步骤 ==========
        // 计算轮速（rad/s）
        double v_left = (sensor_data.left_encoder - sensor_data.prev_left_encoder) / dt;
        double v_right = (sensor_data.right_encoder - sensor_data.prev_right_encoder) / dt;
        
        ekf_predict(&ekf_filter, v_left, v_right, dt);
        
        // ========== EKF 更新步骤（使用 IMU） ==========
        if (devices.imu != 0) {
            ekf_update(&ekf_filter, sensor_data.imu_yaw);
        }
        
        // 保存当前编码器值供下次使用
        sensor_data.prev_left_encoder = sensor_data.left_encoder;
        sensor_data.prev_right_encoder = sensor_data.right_encoder;
        
        // ========== 获取位置估计 ==========
        double odom_x, odom_y, odom_theta;
        odometry_get_pose(&odometry, &odom_x, &odom_y, &odom_theta);
        
        double ekf_x, ekf_y, ekf_theta;
        ekf_get_state(&ekf_filter, &ekf_x, &ekf_y, &ekf_theta);
        
        // ========== 输出调试信息（每 1 秒） ==========
        if (step_count % (1000 / TIME_STEP) == 0) {
            printf("========== 第 %d 秒 ==========\n", step_count / (1000 / TIME_STEP));
            printf("[里程计] X: %.3f m, Y: %.3f m, Theta: %.2f°\n", 
                   odom_x, odom_y, odom_theta * 180.0 / M_PI);
            printf("[EKF估计] X: %.3f m, Y: %.3f m, Theta: %.2f°\n", 
                   ekf_x, ekf_y, ekf_theta * 180.0 / M_PI);
            printf("[IMU原始] Yaw: %.2f°\n", sensor_data.imu_yaw * 180.0 / M_PI);
            
            // 如果有 GPS，显示真实值对比
            if (devices.gps != 0) {
                const double *gps_values = wb_gps_get_values(devices.gps);
                printf("[GPS真值] X: %.3f m, Y: %.3f m\n", gps_values[0], gps_values[1]);
                double error_x = ekf_x - gps_values[0];
                double error_y = ekf_y - gps_values[1];
                double error_distance = sqrt(error_x * error_x + error_y * error_y);
                printf("[误差] 距离误差: %.3f m\n", error_distance);
            }
            printf("\n");
        }
        
        // ========== 电机控制（示例：直线行驶） ==========
        // 您可以在这里添加路径规划和控制逻辑
        double speed = 5.0;  // rad/s
        set_motor_speeds(speed, speed);
        
        // ========== 示例：简单的方形路径 ==========
        /*
        int phase = (step_count / (2000 / TIME_STEP)) % 4;
        if (phase == 0 || phase == 2) {
            // 直行
            set_motor_speeds(5.0, 5.0);
        } else {
            // 转弯
            set_motor_speeds(3.0, -3.0);
        }
        */
    }
    
    // 清理并退出
    wb_robot_cleanup();
    printf("\n[退出] 控制器已停止\n");
    
    return 0;
}