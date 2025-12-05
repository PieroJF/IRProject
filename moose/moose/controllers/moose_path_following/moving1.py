"""
Copyright 1996-2021 Cyberbotics Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from controller import Robot, Keyboard

# Constants
TIME_STEP = 16
TARGET_POINTS_SIZE = 13
DISTANCE_TOLERANCE = 1.5
MAX_SPEED = 7.0
TURN_COEFFICIENT = 4.0

# Enum equivalents
X, Y, Z, ALPHA = 0, 1, 2, 3
LEFT, RIGHT = 0, 1

# Target positions
targets = [
    {'u': -4.209318, 'v': 9.147717},
    {'u': 0.946812, 'v': 9.404304},
    {'u': 0.175989, 'v': -1.784311},
    {'u': -2.805353, 'v': -8.829694},
    {'u': -3.846730, 'v': -15.602851},
    {'u': -4.394915, 'v': -24.550777},
    {'u': -1.701877, 'v': -33.617226},
    {'u': -4.394915, 'v': -24.550777},
    {'u': -3.846730, 'v': -15.602851},
    {'u': -2.805353, 'v': -8.829694},
    {'u': 0.175989, 'v': -1.784311},
    {'u': 0.946812, 'v': 9.404304},
    {'u': -7.930821, 'v': 6.421292}
]

# Global variables
current_target_index = 0
autopilot = True
old_autopilot = True
old_key = -1

# Initialize robot
robot = Robot()
motors = []
gps = None
compass = None
keyboard = None


def modulus_double(a, m):
    """Calculate modulus for double values."""
    div = int(a / m)
    r = a - div * m
    if r < 0.0:
        r += m
    return r


def robot_set_speed(left, right):
    """Set left and right motor speed [rad/s]."""
    for i in range(4):
        motors[i].setVelocity(left)
        motors[i + 4].setVelocity(right)


def check_keyboard():
    """Check keyboard input and control robot accordingly."""
    global autopilot, old_autopilot, old_key
    
    speeds = [0.0, 0.0]
    
    key = keyboard.getKey()
    
    if key >= 0:
        if key == Keyboard.UP:
            speeds[LEFT] = MAX_SPEED
            speeds[RIGHT] = MAX_SPEED
            autopilot = False
        elif key == Keyboard.DOWN:
            speeds[LEFT] = -MAX_SPEED
            speeds[RIGHT] = -MAX_SPEED
            autopilot = False
        elif key == Keyboard.RIGHT:
            speeds[LEFT] = MAX_SPEED
            speeds[RIGHT] = -MAX_SPEED
            autopilot = False
        elif key == Keyboard.LEFT:
            speeds[LEFT] = -MAX_SPEED
            speeds[RIGHT] = MAX_SPEED
            autopilot = False
        elif key == ord('P'):
            if key != old_key:  # perform this action just once
                position_3d = gps.getValues()
                print(f"position: {{{position_3d[X]}, {position_3d[Y]}}}")
        elif key == ord('A'):
            if key != old_key:  # perform this action just once
                autopilot = not autopilot
    
    if autopilot != old_autopilot:
        old_autopilot = autopilot
        if autopilot:
            print("auto control")
        else:
            print("manual control")
    
    robot_set_speed(speeds[LEFT], speeds[RIGHT])
    old_key = key


def norm(v):
    """Calculate vector norm ||v||."""
    return math.sqrt(v['u'] * v['u'] + v['v'] * v['v'])


def normalize(v):
    """Normalize vector v = v/||v||."""
    n = norm(v)
    v['u'] /= n
    v['v'] /= n


def minus(v1, v2):
    """Subtract vectors: result = v1 - v2."""
    return {'u': v1['u'] - v2['u'], 'v': v1['v'] - v2['v']}


def run_autopilot():
    """Autopilot mode - pass through predefined target positions."""
    global current_target_index
    
    speeds = [0.0, 0.0]
    
    # Read GPS position and compass values
    position_3d = gps.getValues()
    north_3d = compass.getValues()
    
    # Compute the 2D position of the robot
    position = {'u': position_3d[X], 'v': position_3d[Y]}
    
    # Compute the direction and distance to the target
    direction = minus(targets[current_target_index], position)
    distance = norm(direction)
    normalize(direction)
    
    # Compute the error angle
    robot_angle = math.atan2(north_3d[0], north_3d[1])
    target_angle = math.atan2(direction['v'], direction['u'])
    beta = modulus_double(target_angle - robot_angle, 2.0 * math.pi) - math.pi
    
    # Move singularity
    if beta > 0:
        beta = math.pi - beta
    else:
        beta = -beta - math.pi
    
    # A target position has been reached
    if distance < DISTANCE_TOLERANCE:
        index_char = "th"
        if current_target_index == 0:
            index_char = "st"
        elif current_target_index == 1:
            index_char = "nd"
        elif current_target_index == 2:
            index_char = "rd"
        print(f"{current_target_index + 1}{index_char} target reached")
        current_target_index += 1
        current_target_index %= TARGET_POINTS_SIZE
    # Move the robot to the next target
    else:
        speeds[LEFT] = MAX_SPEED - math.pi + TURN_COEFFICIENT * beta
        speeds[RIGHT] = MAX_SPEED - math.pi - TURN_COEFFICIENT * beta
    
    # Set the motor speeds
    robot_set_speed(speeds[LEFT], speeds[RIGHT])


def main():
    """Main function."""
    global motors, gps, compass, keyboard
    
    # Print user instructions
    print("You can drive this robot:")
    print("Select the 3D window and use cursor keys:")
    print("Press 'A' to return to the autopilot mode")
    print("Press 'P' to get the robot position")
    print()
    
    robot.step(1000)
    
    # Motor names
    names = ["left motor 1", "left motor 2", "left motor 3", "left motor 4",
             "right motor 1", "right motor 2", "right motor 3", "right motor 4"]
    
    # Get motor devices
    for i in range(8):
        motor = robot.getDevice(names[i])
        motor.setPosition(float('inf'))
        motors.append(motor)
    
    # Get GPS device and enable
    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)
    
    # Get compass device and enable
    compass = robot.getDevice("compass")
    compass.enable(TIME_STEP)
    
    # Enable keyboard
    keyboard = robot.getKeyboard()
    keyboard.enable(TIME_STEP)
    
    # Start forward motion
    robot_set_speed(MAX_SPEED, MAX_SPEED)
    
    # Main loop
    while robot.step(TIME_STEP) != -1:
        check_keyboard()
        if autopilot:
            run_autopilot()


if __name__ == "__main__":
    main()
