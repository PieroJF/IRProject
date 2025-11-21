# IRProject
Role: Sensor Fusion, Calibration & Validation

1. Sensor Fusion Algorithm 
File: controllers/moose_path_following/moose_path_following.c

I was responsible for closing the control loop by implementing the Update Step of the Extended Kalman Filter (EKF). While the odometry (Person A) provides a high-frequency guess, it drifts over time. My algorithm corrects this drift using absolute orientation data. 

EKF Update (ekf_update):

Implemented the measurement update equations: K=PH(T)S(-1).

Fused Compass Yaw data with the prediction state to correct the robot's heading (0).

Updated the Covariance Matrix (P) to reflect the reduced uncertainty after fusion.



 2. System Calibration & Ground Truth
File: controllers/moose_path_following/moose_path_following.c

I developed the validation system to benchmark the localization performance against the GPS ground truth. I also solved a critical coordinate system alignment issue.

Coordinate Alignment :

Issue: The robot's odometer coordinate system was initially 180 degrees opposite to the World coordinate system.

Fix: Implemented an auto-alignment routine in main() that initializes the odometry heading (theta) using the initial Compass reading.



GPS Verification :

Implemented gps_offset logic to zero the coordinate system at the robot's starting position.

Calculated the real-time Euclidean Distance Error between the EKF estimate and GPS truth.

3. Data Analysis & Output 
File: controllers/moose_path_following/moose_path_following.c

I formatted the data output to facilitate performance analysis:
Covariance Visualization: Output the diagonal elements of the P matrix (Var(X), Var(Y), Var(Theta)) to monitor the filter's confidence.

Output Format:

Plaintext

[EKF Covariance] Var(X): 0.410012...
[GPS True Value] X: 12.351 m, Y: 1.077 m
[Error] Distance Error: 0.215 m

 Evaluation Results 
With the fusion algorithm and calibration I implemented, the localization error was reduced from an initial ~19 meters (due to coordinate misalignment) to < 0.5 meters under normal operation.