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
from queue import PriorityQueue

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
# Robot Pose Values
pose_x = 0
pose_y = 0
pose_theta = 0
left_wheel_direction = 0
right_wheel_direction = 0

# Constants to help with the Odometry update
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# GAIN Values
theta_gain = 1.0
distance_gain = 0.3
MAX_VEL_REDUCTION = 0.25

EPUCK_MAX_WHEEL_SPEED = 0.12880519 # m/s
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
EPUCK_WHEEL_RADIUS = 0.0205 # ePuck's wheels are 0.041m in diameter.

left_wheel_direction = 0
right_wheel_direction = 0
EPUCK_MAX_WHEEL_SPEED = 0.12880519 * 2.0  # was 10
EPUCK_AXLE_DIAMETER = 0.053  # mm; from lab 4
OBJECT_AVOIDANCE = 0.055

# get the time step of the current world, given controller code
timestep = int(robot.getBasicTimeStep())


# timestep = 16 #added for debugging
# print("timestep: ", timestep)

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

def get_wheel_speeds(target_pose):
    '''
    @param target_pose: Array of (x,y,theta) for the destination robot pose
    @return motor speed as percentage of maximum for left and right wheel motors
    '''

    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction

    pose_x, pose_y, pose_theta = FinalProject_supervisor.supervisor_get_robot_pose()

    bearing_error = math.atan2((target_pose[1] - pose_y), (target_pose[0] - pose_x)) - pose_theta
    distance_error = np.linalg.norm(target_pose[:2] - np.array([pose_x, pose_y]))
    heading_error = target_pose[2] - pose_theta

    BEAR_THRESHOLD = 0.06
    DIST_THRESHOLD = 0.03
    dT_gain = theta_gain
    dX_gain = distance_gain
    if distance_error > DIST_THRESHOLD:
        dTheta = bearing_error
        if abs(bearing_error) > BEAR_THRESHOLD:
            dX_gain = 0
    else:
        dTheta = heading_error
        dX_gain = 0

    dTheta *= dT_gain
    dX = dX_gain * min(3.14159, distance_error)

    phi_l = (dX - (dTheta * EPUCK_AXLE_DIAMETER / 2.)) / EPUCK_WHEEL_RADIUS
    phi_r = (dX + (dTheta * EPUCK_AXLE_DIAMETER / 2.)) / EPUCK_WHEEL_RADIUS

    left_speed_pct = 0
    right_speed_pct = 0

    wheel_rotation_normalizer = max(abs(phi_l), abs(phi_r))
    left_speed_pct = (phi_l) / wheel_rotation_normalizer
    right_speed_pct = (phi_r) / wheel_rotation_normalizer

    if distance_error < 0.05 and abs(heading_error) < 0.05:
        left_speed_pct = 0
        right_speed_pct = 0

    left_wheel_direction = left_speed_pct * MAX_VEL_REDUCTION
    phi_l_pct = left_speed_pct * MAX_VEL_REDUCTION * leftMotor.getMaxVelocity()

    right_wheel_direction = right_speed_pct * MAX_VEL_REDUCTION
    phi_r_pct = right_speed_pct * MAX_VEL_REDUCTION * rightMotor.getMaxVelocity()

    return phi_l_pct, phi_r_pct

# print(lidar_offsets)

# Take LIDAR readings and convert to world coords, based from lab 4
def convert_lidar_reading_to_world_coord(lidar_bin, lidar_distance):
    x = math.cos(lidar_offsets[lidar_bin]) * lidar_distance
    y = math.sin(lidar_offsets[lidar_bin]) * lidar_distance
    # print("pose",pose_x_2, pose_y_2, pose_theta_2) #added for debugging
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
            # print("world_coord: ", world_coord)
            map_coord = transform_world_coord_to_map_coord(world_coord)
            # print("map_coord: ", map_coord)
            if (map_coord != None):
                world_map[map_coord[0], map_coord[1]] = 1
                # eprint("i LIDAR", i, lidar_readings_array[i])
                # print("world_coord: ", world_coord)
                # print("map_coord: ", map_coord)
                # if (map_coord[1] == 1):
                    # print("MAP", map_coord[0], map_coord[1])
                    # print("WORLD", world_coord[0], world_coord[1])


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
    # for row in range(NUM_Y_CELLS-1,-1,-1): #added for debugging
    # for col in range(NUM_X_CELLS):#added for debugging
    # if(world_map[row,col] == 1): # Checking for a wall#added for debugging
    # print("row,col", row,col)#added for debugging
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


def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    '''
    Given the amount of time passed and the direction each wheel was rotating,
    update the robot's pose information accordingly
    '''
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    pose_theta += (
                              right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER;
    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (
                left_wheel_direction + right_wheel_direction) / 2.;
    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (
                left_wheel_direction + right_wheel_direction) / 2.;
    pose_theta = get_bounded_theta(pose_theta)


def get_bounded_theta(theta):
    '''
    Returns theta bounded in [-PI, PI]
    '''
    while theta > math.pi: theta -= 2. * math.pi
    while theta < -math.pi: theta += 2. * math.pi
    return theta


def get_wheel_speeds(target_pose):
    '''
    @param target_pose: Array of (x,y,theta) for the destination robot pose
    @return motor speed as percentage of maximum for left and right wheel motors
    '''

    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction

    pose_x, pose_y, pose_theta = csci3302_lab5_supervisor.supervisor_get_robot_pose()

    bearing_error = math.atan2((target_pose[1] - pose_y), (target_pose[0] - pose_x)) - pose_theta
    distance_error = np.linalg.norm(target_pose[:2] - np.array([pose_x, pose_y]))
    heading_error = target_pose[2] - pose_theta

    BEAR_THRESHOLD = 0.06
    DIST_THRESHOLD = 0.03
    dT_gain = theta_gain
    dX_gain = distance_gain
    if distance_error > DIST_THRESHOLD:
        dTheta = bearing_error
        if abs(bearing_error) > BEAR_THRESHOLD:
            dX_gain = 0
    else:
        dTheta = heading_error
        dX_gain = 0

    dTheta *= dT_gain
    dX = dX_gain * min(3.14159, distance_error)

    phi_l = (dX - (dTheta * EPUCK_AXLE_DIAMETER / 2.)) / EPUCK_WHEEL_RADIUS
    phi_r = (dX + (dTheta * EPUCK_AXLE_DIAMETER / 2.)) / EPUCK_WHEEL_RADIUS

    left_speed_pct = 0
    right_speed_pct = 0

    wheel_rotation_normalizer = max(abs(phi_l), abs(phi_r))
    left_speed_pct = (phi_l) / wheel_rotation_normalizer
    right_speed_pct = (phi_r) / wheel_rotation_normalizer

    if distance_error < 0.05 and abs(heading_error) < 0.05:
        left_speed_pct = 0
        right_speed_pct = 0

    left_wheel_direction = left_speed_pct * MAX_VEL_REDUCTION
    phi_l_pct = left_speed_pct * MAX_VEL_REDUCTION * leftMotor.getMaxVelocity()

    right_wheel_direction = right_speed_pct * MAX_VEL_REDUCTION
    phi_r_pct = right_speed_pct * MAX_VEL_REDUCTION * rightMotor.getMaxVelocity()

    return phi_l_pct, phi_r_pct


def transform_world_coord_to_map_coord(world_coord):
    """
    @param world_coord: Tuple of (x,y) position in world coordinates
    @return grid_coord: Tuple of (i,j) coordinates corresponding to grid row (y-coord) and column (x-coord) in our map
    """
    col, row = np.array(world_coord) / CELL_RESOLUTIONS
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return tuple(np.array([row, col]).astype(int))


def transform_map_coord_world_coord(map_coord):
    """
    @param map_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    @return world_coord: Tuple of (x,y) position corresponding to the center of map_coord, in world coordinates
    """
    row, col = map_coord
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return np.array([(col + 0.5) * CELL_RESOLUTIONS[1], (row + 0.5) * CELL_RESOLUTIONS[0]])


def display_map(m):
    """
    @param m: The world map matrix to visualize
    """
    m2 = copy.copy(m)
    robot_pos = transform_world_coord_to_map_coord([pose_x, pose_y])
    m2[robot_pos] = 8
    map_str = ""
    for row in range(m.shape[0] - 1, -1, -1):
        for col in range(m.shape[1]):
            if m2[row, col] == 0:
                map_str += '[ ]'
            elif m2[row, col] == 1:
                map_str += '[X]'
            elif m2[row, col] == 2:
                map_str += '[+]'
            elif m2[row, col] == 3:
                map_str += '[G]'
            elif m2[row, col] == 4:
                map_str += '[S]'
            elif m2[row, col] == 8:
                map_str += '[r]'
            else:
                map_str += '[E]'

        map_str += '\n'

    print(map_str)
    print(' ')


# ^ Starter code: You shouldn't have to modify any of this ^
############################################################


###################
# Part 1.1
###################
def get_travel_cost(source_vertex, dest_vertex):
    """
    @param source_vertex: world_map coordinates for the starting vertex
    @param dest_vertex: world_map coordinates for the destination vertex
    @return cost: Cost to travel from source to dest vertex.
    """
    # The next three lines to the return were given

    global world_map
    if (world_map[dest_vertex[1]][dest_vertex[0]] == 1):
        cost = 1e5
    else:
        cost = abs(dest_vertex[0] - source_vertex[0]) + abs(dest_vertex[1] - source_vertex[1])
    # print("Cost: ", cost)
    return cost


###################
# Part 1.2
###################
# added
def get_index(v):
    return (v[0] * NUM_Y_CELLS) + v[1]


# added
def get_vertex(vi):
    row = int(vi / NUM_Y_CELLS)
    col = vi % NUM_Y_CELLS
    return (row, col)


# added
def get_neighbors(u):
    f = []
    if (u[0] != 0):
        s = (u[0] - 1, u[1])
        f.append(s)
    if (u[0] != (NUM_X_CELLS - 1)):
        n = (u[0] + 1, u[1])
        f.append(n)
    if (u[1] != 0):
        w = (u[0], u[1] - 1)
        f.append(w)
    if (u[1] != (NUM_Y_CELLS - 1)):
        e = (u[0], u[1] + 1)
        f.append(e)
    return f


def dijkstra(source_vertex):
    """
    @param source_vertex: Starting vertex for the search algorithm.
    @return prev: Data structure that maps every vertex to the coordinates of the previous vertex (along the shortest path back to source)
    """
    global world_map

    # TODO: Initialize these variables
    dist = np.zeros([NUM_Y_CELLS * NUM_X_CELLS])
    prev = np.zeros([NUM_Y_CELLS * NUM_X_CELLS])

    # TODO: Your code here
    # Based from the lecture pseudocode
    Q = PriorityQueue()
    for row in range(world_map.shape[0]):
        for col in range(world_map.shape[1]):
            v = (row, col)
            vi = get_index((row, col))
            if source_vertex is None:
                continue
            if row != source_vertex[0] or col != source_vertex[1]:
                dist[vi] = math.inf
            prev[vi] = -1
            Q.put((vi, dist[vi]))
            # print(vi, dist[vi])
    # print("DP0", prev)
    # print(dist)

    while (not Q.empty()):
        ui = Q.get()[0]
        if dist[ui] != math.inf:
            # print(prev)
            # print("uidist", dist[ui])
            # print("Q", ui)
            u = get_vertex(ui)
            # print("u", u)
            neighbors = get_neighbors(u)
            # print("n", neighbors)
            for v in neighbors:
                if (world_map[v[0], v[1]]) != 1:
                    vi = get_index(v)
                    q_cost = get_travel_cost(u, v)
                    # print("qcost", q_cost)
                    alt = dist[ui] + q_cost
                    # print("a", alt)
                    if alt < dist[vi]:
                        dist[vi] = alt
                        prev[vi] = ui
                        Q.put((vi, alt))
        # return prev
    # print("Dprev", prev)
    # print(dist)

    return prev


###################
# Part 1.3
###################
def reconstruct_path(prev, goal_vertex):
    """
    @param prev: Data structure mapping each vertex to the next vertex along the path back to "source" (from Dijkstra)
    @param goal_vertex: Map coordinates of the goal_vertex
    @return path: List of vertices where path[0] = source_vertex_ij_coords and path[-1] = goal_vertex_ij_coords
    """

    # Hint: Start at the goal_vertex and work your way backwards using prev until there's no "prev" left to follow.
    #       Then, reverse the list and return it!

    path = []
    u = goal_vertex
    ui = get_index(u)
    # print(prev)
    # print("u", u)
    # print("ui", ui)
    # print("pui", prev[ui])
    while prev[ui] != -1:
        path.insert(0, u)
        ui = int(prev[ui])
        u = get_vertex(ui)
    path.insert(0, u)

    return path


###################
# Part 1.4
###################
def visualize_path(path):
    """
    @param path: List of graph vertices along the robot's desired path
    """
    global world_map

    # TODO: Set a value for each vertex along path in the world_map for rendering: 2 = Path, 3 = Goal, 4 = Start
    count = 1
    path_len = len(path)
    for v in path:
        if (count == 1):
            world_map[v[1], v[0]] = 3
        elif (count == path_len):
            world_map[v[1], v[0]] = 4
        else:
            world_map[v[1], v[0]] = 2
        count += 1

    return


# For 2.1b
def calc_theta(from_v, to_v):
    if (from_v[0] == to_v[0]):
        if (from_v[1] > to_v[1]):
            return math.pi * 3 / 2
        else:
            return math.pi / 2
    if (from_v[1] == to_v[1]):
        if (from_v[0] > to_v[0]):
            return math.pi
    return 0
#########################################################################################################


# Update the odometry
last_odometry_update_time = None
# robots start location, needs to return to this spot
start_pose = FinalProject_supervisor.supervisor_get_robot_pose()
pose_x, pose_y, pose_theta = start_pose
# print("start_pose: ", start_pose)
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

path_progress = 1
status = 'get_waypoint'
while robot.step(timestep) != -1:

    # Update odometry, based from labs 2 and 4
    if last_odometry_update_time is None:
        last_odometry_update_time = robot.getTime()
    time_elapsed = robot.getTime() - last_odometry_update_time
    # print("Time: ", time_elapsed)
    # last_odometry_update_time += time_elapsed
    # update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)

    if state == 'start':
        # Should start in a stopped position
        state = 'update_map'
        # state = "find_goal"
        # state = 'get_path' #Testing Dijkstra's Algorithm

        pass
    if state == 'update_map':
        robot_pose = FinalProject_supervisor.supervisor_get_robot_pose()
        pose_x_2 = robot_pose[0]
        pose_y_2 = robot_pose[1]
        pose_theta_2 = robot_pose[2]
        # print("Robot pose: ", robot_pose)
        # print("Odometry Pose:", pose_x, pose_y, pose_theta)
        lidar_readings = lidar.getRangeImage()
        update_map(lidar_readings)
        # display_map(world_map)
        # print("lidar_readings: ", lidar_readings)
        # print("timestep: ", timestep)
        # print(" len lidar_readings: ", len(lidar_readings))
        if lidar_readings[1] < OBJECT_AVOIDANCE:
            # This will need to be updated in navigation step
            state = 'stop'
        else:
            state = 'find_goal'

        pass
    print("Help")
    state = "find_goal"
    state = 'get_path'
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

        # print(checkForInfront)
        # print(headingFront)
        # print(world_map[checkForInfront[0],checkForInfront[1]])
        # print(index)
        # check for in front

        if world_map[checkForRightWall[0], checkForRightWall[1]] == 1 or world_map[
            checkForRightWall2[0], checkForRightWall2[1]] == 1:
            firstContact = True
        print(robot_pose[2])
        print(turnPoseIntoDirection(robot_pose[2]))
        print(direction)

        if turning:
            if turnPoseIntoDirection(robot_pose[2]) == direction:
                left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
                right_wheel_direction = EPUCK_MAX_WHEEL_SPEED
                leftMotor.setVelocity(left_wheel_direction)
                rightMotor.setVelocity(right_wheel_direction)
                turning = False
                print("pleasaase", turnDirection)
                if turnDirection == 'right':
                    print(direction)
                    if world_map[checkForRightWall2[0], checkForRightWall2[1]] == 0:
                        turning = True




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
            print("SHIT")
            print(checkForRightWall, checkForRightWall2)
            print(robot_pos)
            print(world_map[checkForRightWall[0], checkForRightWall[1]])
            print(world_map[checkForRightWall2[0], checkForRightWall2[1]])
            print(firstContact)
            turning = True
            turnDirection = 'right'
            direction,headingFront,headingFront2,headingRight,headingRight2 = update_heading(direction, 'right')


        elif world_map[checkForInfront2[0], checkForInfront2[1]] == 1:
            print("FUCK")
            print(checkForInfront,checkForInfront2)
            print(robot_pos)
            print(world_map[checkForRightWall[0], checkForRightWall[1]])
            print(world_map[checkForRightWall2[0], checkForRightWall2[1]])
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

        if 'get_waypoint' == status:
            print("part one seems okay")
            ###################
            # Part 2.1b
            ###################
            # Get the next waypoint from the path
            if path_progress < len(path):
                goal_waypoint = path[path_progress]
                if len(path) - path_progress != 2:
                    goal_waypoint_theta = calc_theta(path[len(path) - path_progress], path[len(path) - 1 - path_progress])

                waypoint_pose = transform_map_coord_world_coord(goal_waypoint)
                goal_pose[0] = waypoint_pose[0]
                goal_pose[1] = waypoint_pose[1]
                goal_pose[2] = goal_waypoint_theta
                status = 'move_to_waypoint'
            else:
                goal_pose[0] = source_vertex[0]
                goal_pose[1] = source_vertex[1]
                goal_pose[2] = 4.6
                state = 'get_path'

        elif status == 'move_to_waypoint':
            ###################
            # Part 2.1c
            ###################
            print("fdjkgsh")
            # Hint: Use the IK control function to travel to the current waypoint
            # Syntax/Hint for using the IK Controller:
            # lspeed, rspeed = get_wheel_speeds(target_wp)
            # leftMotor.setVelocity(lspeed)
            # rightMotor.setVelocity(rspeed)

            if abs(goal_pose[0] - pose_x) < 0.05 and abs(goal_pose[1] - pose_y) < 0.05 and abs(goal_pose[2] - pose_theta) < 0.05:
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(0)
                left_wheel_direction, right_wheel_direction = 0, 0
                state = 'get_waypoint'
                path_progress += 1
            else:
                lspeed, rspeed = get_wheel_speeds(goal_pose)
                leftMotor.setVelocity(lspeed)
                rightMotor.setVelocity(rspeed)

    robot_pose = FinalProject_supervisor.supervisor_get_robot_pose()
    pose_x_2 = robot_pose[0]
    pose_y_2 = robot_pose[1]
    pose_theta_2 = robot_pose[2]
    # print("Robot pose: ", robot_pose)
    # print("Odometry Pose:", pose_x, pose_y, pose_theta)
    lidar_readings = lidar.getRangeImage()
    update_map(lidar_readings)
    display_map(world_map)


# Enter here exit cleanup code.
exit()
