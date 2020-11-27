"""
CSCI 3302: Introduction to Robtics Fall 2020
Jorge Ortiz Venegas, Ryan Than, Sage Garrett, Sarah Schwallier, William Culkin
Maze Search + Rescue 
Robot Controller.
"""

import math
import copy
import time
import numpy as np
from controller import Robot, Motor, DistanceSensor
import FinalProject_supervisor

state = 'start'

# create the Robot instance, given controller code
FinalProject_supervisor.init_supervisor()
robot = FinalProject_supervisor.supervisor

# World Map Variables, based from lab 5
MAP_BOUNDS_X = [-0.75,0.83] # based on the maze and safe zone upper and lower bounds
MAP_BOUNDS_Y = [0.5,1.5] # based on the maze and safe zone upper and lower bounds
CELL_RESOLUTIONS = np.array([0.05, 0.05]) # 26 by 20 cells
NUM_X_CELLS = int((MAP_BOUNDS_X[1]-MAP_BOUNDS_X[0]) / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int((MAP_BOUNDS_Y[1]-MAP_BOUNDS_Y[0]) / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS,NUM_X_CELLS])

# LIDAR Variables, based from lab 4
LIDAR_SENSOR_MAX_RANGE = .4 # Meters
LIDAR_ANGLE_BINS = 3 # 3 Bins to cover the angular range of the lidar, centered at 1
LIDAR_ANGLE_RANGE = 1.5708 # 90 degrees, 1.5708 radians

pose_x = 0
pose_y = 0
pose_theta = 0
pose_x_2 = 0
pose_y_2 = 0
pose_theta_2 = 0

left_wheel_direction = 0
right_wheel_direction = 0
EPUCK_MAX_WHEEL_SPEED = 0.12880519 * 4.0
EPUCK_AXLE_DIAMETER = 0.053 # mm; from lab 4
OBJECT_AVOIDANCE = 0.1 

# get the time step of the current world, given controller code
timestep = int(robot.getBasicTimeStep())
#print("timestep: ", timestep)

# Update the odometry, based from lab 2 and 4
def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    pose_theta += (right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER
    pose_x += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.
    pose_y += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.


# get and enable lidar, from lab 4
lidar = robot.getLidar("LDS-01")
lidar.enable(timestep)
lidar.enablePointCloud()

#Initialize lidar motors, from lab 4
lidar_main_motor = robot.getMotor('LDS-01_main_motor')
lidar_secondary_motor = robot.getMotor('LDS-01_secondary_motor')
lidar_main_motor.setPosition(float('inf'))
lidar_secondary_motor.setPosition(float('inf'))
lidar_main_motor.setVelocity(30.0)
lidar_secondary_motor.setVelocity(60.0)

# instance of a device of the robot, given controller code and based from labs
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Initialize the LIDAR, based from lab 4
lidar_readings = []
lidar_offsets = []
for i in range(0, LIDAR_ANGLE_BINS):
    angle = LIDAR_ANGLE_RANGE / 2.0 - (i * LIDAR_ANGLE_RANGE / (LIDAR_ANGLE_BINS - 1.0))
    lidar_offsets.append(angle)
#print(lidar_offsets)

# Take LIDAR readings and convert to world coords, based from lab 4
def convert_lidar_reading_to_world_coord(lidar_bin, lidar_distance):  
    x = math.cos(lidar_offsets[lidar_bin]) * lidar_distance
    y = math.sin(lidar_offsets[lidar_bin]) * lidar_distance

    rot = np.array([[math.cos(pose_theta_2), -1*math.sin(pose_theta_2), pose_x_2],
                   [math.sin(pose_theta_2), math.cos(pose_theta_2), pose_y_2],
                   [0,0,1]])
    res = np.dot(rot, np.array([x, y, 1]))
    world_x = res[0]
    world_y = res[1]
    return (world_x, world_y)

# Update the map with the LIDAR findings, based from lab 4 and 5
def transform_world_coord_to_map_coord(world_coord):

    if(world_coord[0] > MAP_BOUNDS_X[1] or world_coord[1] > MAP_BOUNDS_Y[1]):
        return None
    if(world_coord[0] < MAP_BOUNDS_X[0] or world_coord[1] < MAP_BOUNDS_Y[0]):
        return None
    row = int((world_coord[1] - MAP_BOUNDS_Y[0]) / CELL_RESOLUTIONS[1])
    col = int((world_coord[0] - MAP_BOUNDS_X[0]) / CELL_RESOLUTIONS[0])
    return (row, col)

# Update the map with the LIDAR findings, based from lab 4 and 5
def transform_map_coord_world_coord(map_coord):
    if(map_coord[0] >= NUM_Y_CELLS or map_coord[1] >= NUM_X_CELLS):
        return None
    if(map_coord[0] < 0 or map_coord[1] < 0):
        return None
    x = map_coord[1] * CELL_RESOLUTIONS[1] + MAP_BOUNDS_Y[0]
    y = map_coord[0] * CELL_RESOLUTIONS[0] + MAP_BOUNDS_X[0]
    return (x, y)
    
# Update the map with the LIDAR findings, based from lab 4 and 5
def update_map(lidar_readings_array):
    for i in range(0, LIDAR_ANGLE_BINS):
        if(lidar_readings_array[i] < LIDAR_SENSOR_MAX_RANGE):
            world_coord = convert_lidar_reading_to_world_coord(i, lidar_readings[i])
            #print("world_coord: ", world_coord)
            map_coord = transform_world_coord_to_map_coord(world_coord)
            #print("map_coord: ", map_coord)
            if(map_coord != None):
                world_map[map_coord[0],map_coord[1]] = 1
                
# Display the map using the LIDAR findings, based from lab 4 and 5
def display_map(m):
    for col in range(NUM_X_CELLS):
        print("---", end = '')
    print("")
    robot_pos = transform_world_coord_to_map_coord([pose_x_2,pose_y_2])
    goal_pos = transform_world_coord_to_map_coord([goal_pose[0],goal_pose[1]])
    world_map[goal_pos[0],goal_pos[1]] = 3
    world_map[robot_pos[0],robot_pos[1]] = 4
    for row in range(NUM_Y_CELLS-1,-1,-1):
        for col in range(NUM_X_CELLS):
            if(world_map[row,col] == 1): # Checking for a wall
                print("[X]", end = '')
            elif(world_map[row,col] == 2): # Start location
                print("[S]", end = '')
            elif(world_map[row,col] == 3): # Goal
                print("[G]", end = '')
            elif(world_map[row,col] == 4): # Robot
                print("[R]", end = '')
            elif(world_map[row,col] == 5): # Previous robot path/track
                print("[o]", end = '')
            else:
                print("[ ]", end = '') # Unknown or empty
        print("")
    for col in range(NUM_X_CELLS):
        print("---", end = '')
    print("")
    world_map[robot_pos[0],robot_pos[1]] = 5
# Update the odometry
last_odometry_update_time = None
# robots start location, needs to return to this spot
start_pose = FinalProject_supervisor.supervisor_get_robot_pose()
pose_x, pose_y, pose_theta = start_pose
#print("start_pose: ", start_pose)
goal_pose = FinalProject_supervisor.supervisor_get_target_pose()

# Main loop:
while robot.step(timestep) != -1:

    # Update odometry, based from labs 2 and 4
    if last_odometry_update_time is None:
        last_odometry_update_time = robot.getTime()
    time_elapsed = robot.getTime() - last_odometry_update_time
    #print("Time: ", time_elapsed)
    last_odometry_update_time += time_elapsed
    update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)
 
    if state == 'start':
        # Should start in a stopped position
        state = 'update_map'
        
        pass
    if state == 'update_map':
        robot_pose = FinalProject_supervisor.supervisor_get_robot_pose()
        pose_x_2 = robot_pose[0]
        pose_y_2 = robot_pose[1]
        pose_theta_2 = robot_pose[2]
        #print("Robot pose: ", robot_pose)
        #print("Odometry Pose:", pose_x, pose_y, pose_theta)
        lidar_readings = lidar.getRangeImage()
        update_map(lidar_readings)
        display_map(world_map)
        #print("lidar_readings: ", lidar_readings)
        #print("timestep: ", timestep)
        #print(" len lidar_readings: ", len(lidar_readings))
        if lidar_readings[1] < OBJECT_AVOIDANCE:
            # This will need to be updated in navigation step
            state = 'stop'
        else:
            state = 'move_forward'
        
        pass
    if state == 'move_forward':
        # wheel directions should be updated, both are currently set to 0, EPUCK_MAX_WHEEL_SPEED may need to be updated
        left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
        right_wheel_direction = EPUCK_MAX_WHEEL_SPEED
        leftMotor.setVelocity(left_wheel_direction)
        rightMotor.setVelocity(right_wheel_direction)
        state = 'update_map'
        
        pass
    if state == 'stop':
        left_wheel_direction = 0.0
        right_wheel_direction = 0.0
        leftMotor.setVelocity(left_wheel_direction)
        rightMotor.setVelocity(right_wheel_direction)
        state = 'update_map'
        
        pass
    pass

# Enter here exit cleanup code.
exit()