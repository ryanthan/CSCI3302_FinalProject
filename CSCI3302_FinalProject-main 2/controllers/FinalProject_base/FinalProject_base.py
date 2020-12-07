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
MAP_BOUNDS_X = [-0.7,0.85] # based on the maze and safe zone upper and lower bounds
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
EPUCK_MAX_WHEEL_SPEED = 0.12880519 * 10.0
EPUCK_AXLE_DIAMETER = 0.053 # mm; from lab 4
OBJECT_AVOIDANCE = 0.055

# get the time step of the current world, given controller code
timestep = int(robot.getBasicTimeStep())
#timestep = 16 #added for debugging
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
                #eprint("i LIDAR", i, lidar_readings_array[i])
                #print("world_coord: ", world_coord)
                #print("map_coord: ", map_coord)
                
# Display the map using the LIDAR findings, based from lab 4 and 5
def display_map(m):
    for col in range(NUM_X_CELLS):
        print("---", end = '')
    print("")
    robot_pos = transform_world_coord_to_map_coord([pose_x_2,pose_y_2])
    goal_pos = transform_world_coord_to_map_coord([goal_pose[0],goal_pose[1]])
    
    if state != 'get_path': #Kept getting an error with these lines when using Dijkstra's so I had to add this if statement
        world_map[goal_pos[0],goal_pos[1]] = 3
        world_map[robot_pos[0],robot_pos[1]] = 4
    #for row in range(NUM_Y_CELLS-1,-1,-1): #added for debugging
        #for col in range(NUM_X_CELLS):#added for debugging
            #if(world_map[row,col] == 1): # Checking for a wall#added for debugging
               # print("row,col", row,col)#added for debugging
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
            elif(world_map[row,col] == 6): # Path to goal (from Dijkstra's)
                print("[+]", end = '')
            else:
                print("[ ]", end = '') # Unknown or empty
        print("")
    for col in range(NUM_X_CELLS):
        print("---", end = '')
    print("")
    
    if state != 'get_path':
        world_map[robot_pos[0],robot_pos[1]] = 5
    

# Dijkstra's Path Finding (from lab 5)
#########################################################################################################
def get_travel_cost(source_vertex, dest_vertex):
    global world_map
    
    #Get the i, j coordinates of both cells
    i_source = source_vertex[0]
    j_source = source_vertex[1]
    
    i_dest = dest_vertex[0]
    j_dest = dest_vertex[1]
    
    distance = abs(i_source - i_dest) + abs(j_source - j_dest) #Calculate the distance between cells
    
    #If the source or destination cell is occupied OR the two cells are NOT neighbors:
    if (world_map[i_source][j_source] == 1 or world_map[i_dest][j_dest] == 1 or distance > 1):
        cost = 1e5 #Return a high cost
    #If the source and destination cell are the same:
    elif (i_source == i_dest and j_source == j_dest):
        cost = 0 #Return 0
    #If the source and destination are neighbors:
    elif (distance == 1):
        cost = 1 #Return 1
    return cost 
    
    
def dijkstra(source_vertex):
    global world_map
    
    source_vertex = tuple(source_vertex) # Make it a tuple so you can index into numpy arrays and dictionaries with it

    #Initialize Dijkstra variables
    dist = np.full_like(world_map, math.inf) #Create a matrix the same size as world_map with a large value
    dist[source_vertex] = 0 #Set the source vertex cost value to zero
    q_cost = [] #This is how many nodes we have left to visit
    prev = {} #Dictionary to map tuples to each other

    q_cost.append((source_vertex, 0)) #Initialize q_cost with (source_vertex, 0)

    #While q_cost is not empty:
    while(robot.step(timestep) != -1 and q_cost):
        q_cost.sort(key=lambda x:x[1]) #Sort q_cost by cost
        u = q_cost[0][0] #Get the vertex with the smallest value
        q_cost = q_cost[1:] #Remove the smallest value from the array
        
        neighbors = [(u[0], u[1] + 1), (u[0] + 1, u[1]), (u[0], u[1] - 1), (u[0] - 1, u[1])] #Set the neighbors of vertex u
        
        for v in neighbors: #Loop through each neighbor
            if (v[0] < 0 or v[1] < 0 or v[0] >= len(world_map) or v[1] >= len(world_map)): #Ignore invalid cells
                pass
            else:
                alt = dist[u] + get_travel_cost(u, v) #Calculate the alternate cost of getting to node v from node u
                
                if alt < dist[v]: #Check if alternate cost is less than dist[v]
                    dist[v] = alt #Set dist[v] to the new best cost to get there
                    prev[v] = u #Set prev[v] to the new vertex to use
                    q_cost.append((v, dist[v])) #Append v onto q_cost with its new cost (dist[v])
    return prev

def reconstruct_path(prev, goal_vertex):  
    #Initialize variables
    path = [goal_vertex]
    cur_vertex = goal_vertex
    
    #While cur_vertex is in prev:
    while (robot.step(timestep) != -1 and cur_vertex in prev):
        path.append(prev[cur_vertex]) #Append whatever's in prev[cur_vertex] to the path
        cur_vertex = prev[cur_vertex] #Set cur_vertex to prev[cur_vertex]

    path.reverse() #Reverse the path so it starts with the original source_vertex
    
    return path #Return the final path from the source to the goal vertex


def visualize_path(path):
    global world_map
    
    #For each vertex along the path: in the world_map for rendering: 2 = Path, 3 = Goal, 4 = Start
    for v in path:
        if (path.index(v) == 0): #Set the start to be 4
            world_map[v] = 4
        elif (path.index(v) == len(path)-1): #Set the goal to be 3
            world_map[v] = 3
        else: #Set the path to be 6
            world_map[v] = 6
    return

#########################################################################################################
  

# Update the odometry
last_odometry_update_time = None
# robots start location, needs to return to this spot
start_pose = FinalProject_supervisor.supervisor_get_robot_pose()
pose_x, pose_y, pose_theta = start_pose
#print("start_pose: ", start_pose)
goal_pose = FinalProject_supervisor.supervisor_get_target_pose()

target = 0
boolean = False
headingFront = [1,0]
headingRight = [0,1]
index = 0
states = [[1,0],[0,-1],[-1,0],[0,1]]
firstContact = False
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
        # state = 'get_path' #Testing Dijkstra's Algorithm
        
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
        #display_map(world_map)
        #print("lidar_readings: ", lidar_readings)
        #print("timestep: ", timestep)
        #print(" len lidar_readings: ", len(lidar_readings))
        if lidar_readings[1] < OBJECT_AVOIDANCE:
            # This will need to be updated in navigation step
            state = 'stop'
        else:
            state = 'move_forward'
        
        pass
    
    state = "find_goal"
    if state == "find_goal":
        robot_pos = transform_world_coord_to_map_coord([pose_x_2,pose_y_2])
        
        #no x to right go righjt
        display_map(world_map)
        #if world_map[robot_pos[0] + 1, robot_pos[1]] == 0 and world_map[robot_pos[0], robot_pos[1] + 1] == 0:
        # x in front of robot go lleft
       
        checkForInfront =  [robot_pos[0] + headingFront[0],robot_pos[1] + headingFront[1]]
        checkForRightWall = [robot_pos[0] + headingRight[0],robot_pos[1] + headingRight[1]]
        #print(checkForInfront)
        #print(headingFront)
        #print(world_map[checkForInfront[0],checkForInfront[1]])
        #print(index)
        #check for in front
        two_pi = math.pi *2
        if world_map[checkForRightWall[0],checkForRightWall[1]] == 1:
            firstContact = True
        if world_map[checkForInfront[0],checkForInfront[1]] == 1:
            firstContact = True
            robot_pos = transform_world_coord_to_map_coord([pose_x_2,pose_y_2])
            new = robot_pose[2]
            if boolean == False: 
                target = new + (math.pi/2)
                boolean = True
            
            if abs(target%two_pi-new%two_pi) < .01 : 
                
                boolean = False
                headingFront = states[(index+1)%(4)]
                headingRight = states[(index)%(4)]
                index +=1
                
            left_wheel_direction = 0*EPUCK_MAX_WHEEL_SPEED
            right_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            leftMotor.setVelocity(0*left_wheel_direction)
            rightMotor.setVelocity(right_wheel_direction)
        
        elif world_map[checkForRightWall[0],checkForRightWall[1]] == 0 and firstContact == True:
            robot_pos = transform_world_coord_to_map_coord([pose_x_2,pose_y_2])
            new = robot_pose[2]
            if boolean == False: 
                target = new + (math.pi/2)
                boolean = True
            
            if abs(target%two_pi-new%two_pi) < .01 : 
                
                boolean = False
                headingFront = states[(index-1)%(4)]
                headingRight = states[(index-2)%(4)]
                index -=1
                
            left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            right_wheel_direction = 0*EPUCK_MAX_WHEEL_SPEED
            leftMotor.setVelocity(left_wheel_direction)
            rightMotor.setVelocity(0*right_wheel_direction)
        
        else:
         
            left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            right_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            leftMotor.setVelocity(left_wheel_direction)
            rightMotor.setVelocity(right_wheel_direction)
             
            
        #right of robot is x continue going straight
        

        
        #no x to right go righjt
        
        #
        
        
        
        #display_map(world_map)
        #if wall on the right go forward
        
        #if wall in front turn left
        # if lidar_readings[1] < .1:
            # left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            # right_wheel_direction = -EPUCK_MAX_WHEEL_SPEED
            # leftMotor.setVelocity(left_wheel_direction)
            # rightMotor.setVelocity(-right_wheel_direction)
        
        # if lidar_readings[2] > .1:
            # print("huh")
            # left_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            # right_wheel_direction = EPUCK_MAX_WHEEL_SPEED
            # leftMotor.setVelocity(left_wheel_direction)
            # rightMotor.setVelocity(right_wheel_direction)
        
        #if no wall in front or on right turn right
        
        
        
        
        
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
        source_vertex = transform_world_coord_to_map_coord([pose_x,pose_y]) #Set the source vertex
        goal_vertex = transform_world_coord_to_map_coord([goal_pose[0], goal_pose[1]]) #Set the overall goal vertex
        
        prev = dijkstra(source_vertex) #Get prev using Dijkstra's
        path = reconstruct_path(prev, goal_vertex) #Reconstruct the path using prev
        visualize_path(path) #Visualize the path on the world map
        display_map(world_map)
        
        pass
        
    pass

# Enter here exit cleanup code.
exit()