from controllers import *
import cv2

import random

if __name__ == '__main__':  
    entities = [
    #{'id': 'car1', 'length': 0.2, 'width': 0.15, 'x': 1, 'y': 0.75, 'theta': np.radians(180), 'encoders': [random.randint(0, 10000), random.randint(0, 10000)]},
    #{'id': 'car1', 'length': 0.2, 'width': 0.15, 'x': 0, 'y': 0, 'theta': 0, 'encoders': [random.randint(0, 10000), random.randint(0, 10000)]}, # for testing

    {'id': 'car1', 'length': 0.2, 'width': 0.15, 'x': 0.25, 'y': 0.25, 'theta': np.radians(0)},
    {'id': 'car2', 'length': 0.2, 'width': 0.15, 'x': 1.75, 'y': 1.75, 'theta': np.radians(180)},
    # {'id': 'car3', 'length': 0.2, 'width': 0.15, 'x': 0.25, 'y': 1.75, 'theta': np.radians(270)},
    # {'id': 'car4', 'length': 0.2, 'width': 0.15, 'x': 1.75, 'y': 0.25, 'theta': np.radians(90)}

    ] # max 3 cars, very slow otherwise
    
    # beacons = [
    #     {'id': 'A', 'x': 1, 'y': 0.25, 'code': '011011'},
    #     {'id': 'B', 'x': 0.25, 'y': 0.3, 'code': '011110'},
    #     {'id': 'C', 'x': 1.7, 'y': 0.25 , 'code': '100111'},
    #     {'id': 'D', 'x': 1.5, 'y': 1.85 , 'code': '101101'},
    #     {'id': 'E', 'x': 1.8, 'y': 1.5 , 'code': '111001'},
    # ]

    # print("encoders: ", entities[0]['encoders'])

    # LINE TRAJECTORY
    A,B,C = (1, -1, 0) # change these values to whatever
    m = -A/B # gradient for calculating target angle
    T = 0.05
    x_values0 = np.arange(0, 2, T)
    y_values0 = (-A * x_values0 - C) / B
    line_traj = np.array([[x_values0], [y_values0]])
    line_name = "Line Trajectory" 
    line = PursuitLine(x_values0, y_values0)

    # PURE PURSUIT CIRCLE
    radius, center = 0.7, (1,1)
    theta = np.linspace(0, 2 * np.pi, 300)
    x_values1 = radius * np.cos(theta) + center[0]
    y_values1 = radius * np.sin(theta) + center[1]  
    circle_traj = np.array([[x_values1], [y_values1]])
    circle_name = "Follow Circle"
    circle = PursuitLine(x_points=x_values1, y_points=y_values1)
    
    # MOVE TO POINT
    goal_x = 1.7
    goal_y = 1
    goal = np.array([[goal_x], [goal_y]])
    goal_name = "Move to Point"

    trajectories = [
        #(line_name, line_traj),
        (circle_name, circle_traj),
        (goal_name, goal),
    ]

    
    pure_pursuit(entities, trajectories, circle, 0.1, 25, KL=5, goal=[goal_x, goal_y])
    # no_localiser_template(entities, beacons)
    # move_to_singular_point(entities, goal, target_v=25, trajectories=trajectories, KL=1.5)
    # movetopoint_noarena(entities, goal, target_v=25, trajectories=trajectories, KL=1.5)
    plt.show()
    
