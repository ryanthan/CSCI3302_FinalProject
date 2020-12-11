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
MAP_BOUNDS_X = [-0.75, 0.75]  # based on the maze and safe zone upper and lower bounds
MAP_BOUNDS_Y = [0.5, 1.5]  # based on the maze and safe zone upper and lower bounds
CELL_RESOLUTIONS = np.array([0.05, 0.05])  # 26 by 20 cells
NUM_X_CELLS = int((MAP_BOUNDS_X[1] - MAP_BOUNDS_X[0]) / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int((MAP_BOUNDS_Y[1] - MAP_BOUNDS_Y[0]) / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS, NUM_X_CELLS])

# LIDAR Variables, based from lab 4
LIDAR_SENSOR_MAX_RANGE = .3  # Meters
LIDAR_ANGLE_BINS = 3  # 3 Bins to cover the angular range of the lidar, centered at 1
LIDAR_ANGLE_RANGE = 1.0472  # 90 degrees, 1.5708 radians

pose_x = 0
pose_y = 0
pose_theta = 0
pose_x_2 = 0
pose_y_2 = 0
pose_theta_2 = 0

left_wheel_direction = 0
right_wheel_direction = 0
EPUCK_MAX_WHEEL_SPEED = 0.12880519 * 2.0  # was 10
EPUCK_AXLE_DIAMETER = 0.053  # mm; from lab 4
OBJECT_AVOIDANCE = 0.055

# get the time step of the current world, given controller code
timestep = int(robot.getBasicTimeStep())


# timestep = 16 #added for debugging

# Update the odometry, based from lab 2 and 4
def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    pose_theta += (
                          right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER
    pose_x += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (
            left_wheel_direction + right_wheel_direction) / 2.
    pose_y += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (
            left_wheel_direction + right_wheel_direction) / 2.


# get and enable lidar, from lab 4
lidar = robot.getLidar("LDS-01")
lidar.enable(timestep)
lidar.enablePointCloud()

# Initialize lidar motors, from lab 4
lidar_main_motor = robot.getMotor('LDS-01_main_motor')
lidar_secondary_motor = robot.getMotor('LDS-01_secondary_motor')
lidar_main_motor.setPosition(float('inf'))
lidar_secondary_motor.setPosition(float('inf'))
lidar_main_motor.setVelocity(10.0)
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



# Take LIDAR readings and convert to world coords, based from lab 4
def convert_lidar_reading_to_world_coord(lidar_bin, lidar_distance):
    x = math.cos(lidar_offsets[lidar_bin]) * lidar_distance
    y = math.sin(lidar_offsets[lidar_bin]) * lidar_distance
    rot = np.array([[math.cos(pose_theta_2), -1 * math.sin(pose_theta_2), pose_x_2],
                    [math.sin(pose_theta_2), math.cos(pose_theta_2), pose_y_2],
                    [0, 0, 1]])
    res = np.dot(rot, np.array([x, y, 1]))
    world_x = res[0]
    world_y = res[1]
    return (world_x, world_y)



# Update the map with the LIDAR findings, based from lab 4 and 5
def transform_world_coord_to_map_coord(world_coord):
    if (world_coord[0] > MAP_BOUNDS_X[1] or world_coord[1] > MAP_BOUNDS_Y[1]):
        return None
    if (world_coord[0] < MAP_BOUNDS_X[0] or world_coord[1] < MAP_BOUNDS_Y[0]):
        return None
    row = int((world_coord[1] - MAP_BOUNDS_Y[0]) / CELL_RESOLUTIONS[1])
    col = int((world_coord[0] - MAP_BOUNDS_X[0]) / CELL_RESOLUTIONS[0])
    return (row, col)



# Update the map with the LIDAR findings, based from lab 4 and 5
def transform_map_coord_world_coord(map_coord):
    if (map_coord[0] >= NUM_Y_CELLS or map_coord[1] >= NUM_X_CELLS):
        return None
    if (map_coord[0] < 0 or map_coord[1] < 0):
        return None
    x = map_coord[1] * CELL_RESOLUTIONS[1] + MAP_BOUNDS_Y[0]
    y = map_coord[0] * CELL_RESOLUTIONS[0] + MAP_BOUNDS_X[0]
    return (x, y)



# Update the map with the LIDAR findings, based from lab 4 and 5
def update_map(lidar_readings_array):
    for i in range(0, LIDAR_ANGLE_BINS):
        if (lidar_readings_array[i] < LIDAR_SENSOR_MAX_RANGE):
            world_coord = convert_lidar_reading_to_world_coord(i, lidar_readings[i])
            map_coord = transform_world_coord_to_map_coord(world_coord)
            if (map_coord != None):
                world_map[map_coord[0], map_coord[1]] = 1
                
                
                
# Display the map using the LIDAR findings, based from lab 4 and 5
def display_map(m):
    for col in range(NUM_X_CELLS):
        print("---", end='')
    print("")
    robot_pos = transform_world_coord_to_map_coord([pose_x_2, pose_y_2])
    goal_pos = transform_world_coord_to_map_coord([goal_pose[0], goal_pose[1]])

    if state != 'get_path':  # Kept getting an error with these lines when using Dijkstra's so I had to add this if statement
        world_map[goal_pos[0], goal_pos[1]] = 3
        world_map[robot_pos[0], robot_pos[1]] = 4
    for row in range(NUM_Y_CELLS - 1, -1, -1):
        for col in range(NUM_X_CELLS):
            if (world_map[row, col] == 1):  # Checking for a wall
                print("[X]", end='')
            elif (world_map[row, col] == 2):  # Start location
                print("[S]", end='')
            elif (world_map[row, col] == 3):  # Goal
                print("[G]", end='')
            elif (world_map[row, col] == 4):  # Robot
                print("[R]", end='')
            elif (world_map[row, col] == 5):  # Previous robot path/track
                print("[o]", end='')
            elif (world_map[row, col] == 6):  # Path to goal (from Dijkstra's)
                print("[+]", end='')
            else:
                print("[ ]", end='')  # Unknown or empty
        print("")
    for col in range(NUM_X_CELLS):
        print("---", end='')
    print("")

    if state != 'get_path':
        world_map[robot_pos[0], robot_pos[1]] = 5



# Dijkstra's Path Finding (from lab 5)
#########################################################################################################
def get_travel_cost(source_vertex, dest_vertex):
    global world_map

    # Get the i, j coordinates of both cells
    i_source = source_vertex[0]
    j_source = source_vertex[1]

    i_dest = dest_vertex[0]
    j_dest = dest_vertex[1]

    distance = abs(i_source - i_dest) + abs(j_source - j_dest)  # Calculate the distance between cells

    # If the source or destination cell is occupied OR the two cells are NOT neighbors:
    if (world_map[i_source][j_source] == 1 or world_map[i_dest][j_dest] == 1 or distance > 1):
        cost = 1e5  # Return a high cost
    # If the source and destination cell are the same:
    elif (i_source == i_dest and j_source == j_dest):
        cost = 0  # Return 0
    # If the source and destination are neighbors:
    elif (distance == 1):
        cost = 1  # Return 1
    return cost


def dijkstra(source_vertex):
    global world_map

    source_vertex = tuple(source_vertex)  # Make it a tuple so you can index into numpy arrays and dictionaries with it

    # Initialize Dijkstra variables
    dist = np.full_like(world_map, math.inf)  # Create a matrix the same size as world_map with a large value
    dist[source_vertex] = 0  # Set the source vertex cost value to zero
    q_cost = []  # This is how many nodes we have left to visit
    prev = {}  # Dictionary to map tuples to each other

    q_cost.append((source_vertex, 0))  # Initialize q_cost with (source_vertex, 0)

    # While q_cost is not empty:
    while (robot.step(timestep) != -1 and q_cost):
        q_cost.sort(key=lambda x: x[1])  # Sort q_cost by cost
        u = q_cost[0][0]  # Get the vertex with the smallest value
        q_cost = q_cost[1:]  # Remove the smallest value from the array

        neighbors = [(u[0], u[1] + 1), (u[0] + 1, u[1]), (u[0], u[1] - 1),
                     (u[0] - 1, u[1])]  # Set the neighbors of vertex u

        for v in neighbors:  # Loop through each neighbor
            if (v[0] < 0 or v[1] < 0 or v[0] >= len(world_map) or v[1] >= len(world_map)):  # Ignore invalid cells
                pass
            else:
                alt = dist[u] + get_travel_cost(u, v)  # Calculate the alternate cost of getting to node v from node u

                if alt < dist[v]:  # Check if alternate cost is less than dist[v]
                    dist[v] = alt  # Set dist[v] to the new best cost to get there
                    prev[v] = u  # Set prev[v] to the new vertex to use
                    q_cost.append((v, dist[v]))  # Append v onto q_cost with its new cost (dist[v])
    return prev



def reconstruct_path(prev, goal_vertex):
    # Initialize variables
    path = [goal_vertex]
    cur_vertex = goal_vertex

    # While cur_vertex is in prev:
    while (robot.step(timestep) != -1 and cur_vertex in prev):
        path.append(prev[cur_vertex])  # Append whatever's in prev[cur_vertex] to the path
        cur_vertex = prev[cur_vertex]  # Set cur_vertex to prev[cur_vertex]

    path.reverse()  # Reverse the path so it starts with the original source_vertex

    return path  # Return the final path from the source to the goal vertex


def visualize_path(path):
    global world_map

    # For each vertex along the path: in the world_map for rendering: 2 = Path, 3 = Goal, 4 = Start
    for v in path:
        if (path.index(v) == 0):  # Set the start to be 4
            world_map[v] = 4
        elif (path.index(v) == len(path) - 1):  # Set the goal to be 3
            world_map[v] = 3
        else:  # Set the path to be 6
            world_map[v] = 6
    return


#########################################################################################################




# Update the odometry
last_odometry_update_time = None
# robots start location, needs to return to this spot
start_pose = FinalProject_supervisor.supervisor_get_robot_pose()
pose_x, pose_y, pose_theta = start_pose

goal_pose = FinalProject_supervisor.supervisor_get_target_pose()

target = 0
boolean = False
turning = False
turnDirection = None
headingFront = [1, 0]
headingFront2 = [2,0]
headingRight = [0,1]
headingRight2 = [0,2]
direction = 'N'
firstContact = False




# Main loop:
def update_heading(direction, turnType):
    west = 'W', [0, -1], [0,-2],[1,0], [2, 0]
    east = 'E', [0, 1],[0,2], [-1, 0], [-2,0]
    south = 'S', [-1, 0],[-2,0],[0, -1],[0,-2]
    north = 'N', [1, 0],[2,0], [0, 1],[0,2]
    if direction == 'N' and turnType == 'left':
        return west
    if direction == 'N' and turnType == 'right':
        return east

    if direction == 'W' and turnType == 'left':
        return south
    if direction == 'W' and turnType == 'right':
        return north

    if direction == 'S' and turnType == 'left':
        return east
    if direction == 'S' and turnType == 'right':
        return west

    if direction == 'E' and turnType == 'left':
        return north
    if direction == 'E' and turnType == 'right':
        return south

def turnPoseIntoDirection(pose):
    two_pi = math.pi * 2
    pose = abs(pose%two_pi)
    if 1.5699 < pose < 1.579:
        return 'N'
    if -.004 < pose < .004:
        return 'E'
    if 3.12 < pose < 3.14:
        return 'W'
    if 4.6999 < pose < 4.71:
        return 'S'


while robot.step(timestep) != -1:

    # Update odometry, based from labs 2 and 4
    if last_odometry_update_time is None:
        last_odometry_update_time = robot.getTime()
    time_elapsed = robot.getTime() - last_odometry_update_time

    if state == 'start':
        # Should start in a stopped position
        state = 'update_map'

        pass
    if state == 'update_map':
        robot_pose = FinalProject_supervisor.supervisor_get_robot_pose()
        pose_x_2 = robot_pose[0]
        pose_y_2 = robot_pose[1]
        pose_theta_2 = robot_pose[2]
        lidar_readings = lidar.getRangeImage()
        update_map(lidar_readings)
        if lidar_readings[1] < OBJECT_AVOIDANCE:
            state = 'stop'
        else:
            state = 'find_goal'

        pass

    state = "find_goal"
    if state == "find_goal":
        robot_pose = FinalProject_supervisor.supervisor_get_robot_pose()
        pose_x_2 = robot_pose[0]
        pose_y_2 = robot_pose[1]
        pose_theta_2 = robot_pose[2]
        robot_pos = transform_world_coord_to_map_coord([pose_x_2, pose_y_2])
        # no x to right go righjt
        # display_map(world_map)
        # if world_map[robot_pos[0] + 1, robot_pos[1]] == 0 and world_map[robot_pos[0], robot_pos[1] + 1] == 0:
        # x in front of robot go lleft

        checkForInfront = [robot_pos[0] + headingFront[0], robot_pos[1] + headingFront[1]]
        checkForRightWall = [robot_pos[0] + headingRight[0], robot_pos[1] + headingRight[1]]
        checkForInfront2 = [robot_pos[0] + headingFront2[0], robot_pos[1] + headingFront2[1]]
        checkForRightWall2 = [robot_pos[0] + headingRight2[0], robot_pos[1] + headingRight2[1]]

        # check for in front

        if world_map[checkForRightWall[0], checkForRightWall[1]] == 1 or world_map[
            checkForRightWall2[0], checkForRightWall2[1]] == 1:
            firstContact = True
        #print(robot_pose[2])
        #print(turnPoseIntoDirection(robot_pose[2]))
        #print(direction)

        if turning:
            if turnPoseIntoDirection(robot_pose[2]) == direction:
                left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
                right_wheel_direction = EPUCK_MAX_WHEEL_SPEED
                leftMotor.setVelocity(left_wheel_direction)
                rightMotor.setVelocity(right_wheel_direction)
                turning = False

                if turnDirection == 'right':
                    print(direction)
                    if world_map[checkForRightWall[0], checkForRightWall[1]] == 0 or world_map[checkForRightWall2[0], checkForRightWall2[1]] == 0:
                        turning = True
                        turnDirection = 'right'


            elif turnDirection == 'left':
                left_wheel_direction = 0 * EPUCK_MAX_WHEEL_SPEED
                right_wheel_direction = EPUCK_MAX_WHEEL_SPEED
                leftMotor.setVelocity(0 * left_wheel_direction)
                rightMotor.setVelocity(right_wheel_direction)
            else:
                left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
                right_wheel_direction = 0 * EPUCK_MAX_WHEEL_SPEED
                leftMotor.setVelocity(left_wheel_direction)
                rightMotor.setVelocity(0 * right_wheel_direction)


         elif world_map[checkForRightWall2[0], checkForRightWall2[1]] == 0 and firstContact:
            #print(checkForRightWall, checkForRightWall2)
            #print(robot_pos)
            #print(world_map[checkForRightWall[0], checkForRightWall[1]])
            #print(world_map[checkForRightWall2[0], checkForRightWall2[1]])
            #print(firstContact)
            turning = True
            turnDirection = 'right'
            direction,headingFront,headingFront2,headingRight,headingRight2 = update_heading(direction, 'right')

        elif world_map[checkForInfront2[0], checkForInfront2[1]] == 1:
            #print(checkForInfront,checkForInfront2)
            #print(robot_pos)
            firstContact = True
            turning = True
            turnDirection = 'left'
            direction,headingFront,headingFront2,headingRight,headingRight2 = update_heading(direction, 'left')

            turningTo = direction


       



        else:

            left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            right_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            leftMotor.setVelocity(left_wheel_direction)
            rightMotor.setVelocity(right_wheel_direction)
        state = 'update_map'

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

    if state == 'get_path':
        source_vertex = transform_world_coord_to_map_coord([pose_x, pose_y])  # Set the source vertex
        goal_vertex = transform_world_coord_to_map_coord([goal_pose[0], goal_pose[1]])  # Set the overall goal vertex

        prev = dijkstra(source_vertex)  # Get prev using Dijkstra's
        path = reconstruct_path(prev, goal_vertex)  # Reconstruct the path using prev
        visualize_path(path)  # Visualize the path on the world map
        display_map(world_map)

        pass

    robot_pose = FinalProject_supervisor.supervisor_get_robot_pose()
    pose_x_2 = robot_pose[0]
    pose_y_2 = robot_pose[1]
    pose_theta_2 = robot_pose[2]

    lidar_readings = lidar.getRangeImage()
    update_map(lidar_readings)
    display_map(world_map)
    pass

# Enter here exit cleanup code.
exit()
