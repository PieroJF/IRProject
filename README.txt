Contributor:YICHANG CHAO

Role:Robot Modeling, Low-level Control & Kinematics

File: protos/Moose.proto

To enable odometry calculation, I modified the original Moose robot PROTO file. The default model lacked position sensors on the wheels, which made closed-loop control impossible.

Modification: Added PositionSensor nodes to all 8 wheel joints.

Right Side: Added sensors named right wheel sensor 1 to right wheel sensor 4.

Left Side: Added sensors named left wheel sensor 1 to left wheel sensor 4.

2. Firmware & Data Acquisition

File: controllers/moose_path_following/moose_path_following.c

I implemented the low-level C driver to interface with the Webots API.

Device Initialization (initialize_devices):

Configured handles for 8 motors, 8 encoders, Compass, and GPS.

Set motors to velocity control mode.

Sensor Reading (read_sensors_data):

Reads raw values from encoders (radians).

Converts Compass vector to Yaw angle.

3. Kinematics & Algorithms

File: controllers/moose_path_following/moose_path_following.c

I developed the core motion algorithms based on the skid-steering kinematic model

Odometry (odometry_update):Calculates position ($x, y$) and heading ($\theta$) using differential drive equations.Uses averaged encoder values from left/right sides to minimize slip error.

EKF Prediction (ekf_predict):Implemented the Prediction Step of the Extended Kalman Filter.Propagates the state estimate and covariance matrix $P$ based on control inputs (velocity).

Motion Control:

Implemented velocity command distribution to all 8 wheels.

How to Run:
Ensure Moose.proto is placed in the protos/ directory.

Compile the controller:

Bash

cd controllers/moose_path_following
make
Start the simulation in Webots. The robot will start moving forward, and the console will output the Odometry and EKF Prediction values.
