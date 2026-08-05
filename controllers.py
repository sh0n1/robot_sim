from plot_arena import *
import cv2

import time
from copy import deepcopy

running = True

def on_key(event):
    """Callback function to stop the loop when 'q' is pressed."""
    global running
    if event.key == "q":
        running = False
        print("Exit key detected. Closing...")


def _copy_entities(entities):
    return [entity.copy() for entity in entities]


def _step_pure_pursuit_entity(entity, follow_traj, follow_RAD, target_v, KL=10, goal=None):
    next_entity = entity.copy()
    robot_pose = np.array([[next_entity['x']], [next_entity['y']], [next_entity['theta']]])

    current_x, current_y, current_theta = next_entity['x'], next_entity['y'], next_entity['theta']

    next_entity['current_index'] = follow_traj.get_next_closest_point(
        next_entity['current_index'],
        (next_entity['x'], next_entity['y']),
        follow_RAD,
    )
    target = follow_traj.get_point(next_entity['current_index'])

    turn = angle_to_point(current_x, current_y, np.degrees(current_theta), target)
    yl = KL * turn
    turn = lock_angle(180, yl)

    if not is_close_wall(current_x, current_y):
        robot_pose = turnwheels_sim(robot_pose, target_v, turn, next_entity['width'])

    reached_goal = False
    if goal is not None and is_close_point(current_x, current_y, goal[0], goal[1], tol=0.1):
        reached_goal = True

    next_entity['x'] = robot_pose[0, 0]
    next_entity['y'] = robot_pose[1, 0]
    next_entity['theta'] = robot_pose[2, 0]

    return next_entity, reached_goal


def simulate_pure_pursuit(entities, trajectories, follow_traj, follow_RAD, target_v, KL=10, goal=None, max_steps=2400, dt=0.05, progress_interval=None):
    """Run pure pursuit without rendering and return a frame log."""
    current_entities = deepcopy(entities)
    start_time = time.perf_counter()

    for entity in current_entities:
        entity['current_index'] = follow_traj.find_closest_index((entity['x'], entity['y']))

    history = []
    for step in range(max_steps):
        if progress_interval and step % progress_interval == 0:
            elapsed = time.perf_counter() - start_time
            print(f'Simulation progress: step {step}, sim t={step * dt:.2f}s, elapsed={elapsed:.4f}s')

        history.append({
            'step': step,
            'time': step * dt,
            'entities': _copy_entities(current_entities),
            'trajectories': trajectories,
        })

        next_entities = []
        goal_reached = False
        for index, entity in enumerate(current_entities):
            next_entity, reached_goal = _step_pure_pursuit_entity(
                entity,
                follow_traj,
                follow_RAD,
                target_v,
                KL=KL,
                goal=goal if index == 0 else None,
            )
            next_entities.append(next_entity)
            goal_reached = goal_reached or reached_goal

        current_entities = next_entities
        if goal_reached:
            history.append({
                'step': step + 1,
                'time': (step + 1) * dt,
                'entities': _copy_entities(current_entities),
                'trajectories': trajectories,
            })
            break

    return history, current_entities


def move_to_singular_point(entities, point, target_v, trajectories, KL=10):
    ax = plt.ion()
    plt.figure()

    plt.gcf().canvas.mpl_connect("key_press_event", on_key)

    start_time = time.perf_counter()
    t = 0

    while running:
        # print(f"\ntime: {t:.2f}")
        current_time = time.perf_counter()
        t = current_time - start_time
        
        axes = plt.gca()

         
        for entity in entities:
            plot_arena(axes, entities=entities, arena_size=[2, 2], trajectories=trajectories)
            #plot_noarena(axes, entities=entities, trajectories=trajectories)
            robot_pose = np.array([[entity['x']], [entity['y']], [entity['theta']]])

            current_x, current_y, current_theta = entity['x'], entity['y'], entity['theta'] #getLocalizerPose()

            turn = angle_to_point(current_x, current_y, np.degrees(current_theta), point)
            yl = KL * turn
            turn = lock_angle(180, yl)
            #print(f"turn: {turn:.2f}")

            if not is_close_wall(current_x, current_y) and not is_close_point(current_x, current_y, point[0, 0], point[1, 0], tol = 0.05):
                robot_pose = turnwheels_sim(robot_pose, target_v, (turn), entity['width'])
                
            print(f"Plotting {entity['id']} at ({current_x}, {current_y}) with angle {wrap_angle(np.degrees(current_theta))} in degrees")

            entity['x'] = robot_pose[0, 0]
            entity['y'] = robot_pose[1, 0]        
            entity['theta'] = robot_pose[2, 0]

        if t > 120:
            break

    plt.ioff() 


def movetopoint_noarena(entities, point, target_v, trajectories, KL=10):
    ax = plt.ion()
    plt.figure()

    plt.gcf().canvas.mpl_connect("key_press_event", on_key)

    start_time = time.perf_counter()
    t = 0

    goal_reached = False
    while running and not goal_reached:
        print(f"\ntime: {t:.2f}")
        current_time = time.perf_counter()
        t = current_time - start_time
        
        axes = plt.gca()

         
        for entity in entities:
            if not goal_reached:
                plot_noarena(axes, entities=entities, trajectories=trajectories, current_time=t, interval=0.5)
                print(f"Plotting triangle at ({entity['x']}, {entity['y']}) with angle {wrap_angle(np.degrees(entity['theta']))} in degrees")

            robot_pose = np.array([[entity['x']], [entity['y']], [entity['theta']]])
            current_x, current_y, current_theta = entity['x'], entity['y'], entity['theta'] #getLocalizerPose()

            turn = angle_to_point(current_x, current_y, np.degrees(current_theta), point)
            yl = KL * turn
            turn = lock_angle(180, yl)
            #print(f"turn: {turn:.2f}")

            if not is_close_point(current_x, current_y, point[0, 0], point[1, 0], tol = 0.05):
                robot_pose = turnwheels_sim(robot_pose, target_v, (turn), entity['width'])
            else:
                goal_reached = True
                print("Goal reached!")
                break
                

            entity['x'] = robot_pose[0, 0]
            entity['y'] = robot_pose[1, 0]        
            entity['theta'] = robot_pose[2, 0]

        if t > 120:
            break

    plt.ioff() 

def pure_pursuit(entities, trajectories, follow_traj, follow_RAD, target_v, KL=10, goal=None):
    ax = plt.ion()
    plt.figure()

    plt.gcf().canvas.mpl_connect("key_press_event", on_key)

    start_time = time.perf_counter()
    t = 0
    end = False

    for entity in entities:
        entity['current_index'] = follow_traj.find_closest_index((entity['x'], entity['y']))

    while running:
        # print(f"\ntime: {t:.2f}")
        current_time = time.perf_counter()
        t = current_time - start_time
        
        axes = plt.gca()

        for entity in entities:
            plot_arena(axes, entities=entities, arena_size=[2, 2], trajectories=trajectories)
            robot_pose = np.array([[entity['x']], [entity['y']], [entity['theta']]])

            current_x, current_y, current_theta = entity['x'], entity['y'], entity['theta'] #getLocalizerPose()

            entity['current_index'] = follow_traj.get_next_closest_point(entity['current_index'], (entity['x'], entity['y']), follow_RAD)
            target = follow_traj.get_point(entity['current_index'])

            turn = angle_to_point(current_x, current_y, np.degrees(current_theta), target)
            yl = KL * turn
            turn = lock_angle(180, yl)
            # print(f"turn: {turn:.2f}")

            if not is_close_wall(current_x, current_y):
                robot_pose = turnwheels_sim(robot_pose, target_v, (turn), entity['width'])

            if entity == entities[0] and goal is not None:
                if is_close_point(current_x, current_y, goal[0], goal[1], tol = 0.1):
                    print(f"Entity {entity['id']} reached the goal!")
                    end = True
                    break
                
            print(f"Plotting {entity['id']} at ({current_x}, {current_y}) with angle {wrap_angle(np.degrees(current_theta))} in degrees")

            entity['x'] = robot_pose[0, 0]
            entity['y'] = robot_pose[1, 0]        
            entity['theta'] = robot_pose[2, 0]

        if t > 120 or end == True:
            print("Sim ended at a time of: ", t)
            break
    
    plt.ioff() 


def no_localiser_template(entities, beacons):
    ax = plt.ion()
    plt.figure()

    plt.gcf().canvas.mpl_connect("key_press_event", on_key)

    start_time = time.perf_counter()
    t = 0

    while running:
        print(f"\ntime: {t:.2f}")
        current_time = time.perf_counter()
        t = current_time - start_time
        
        axes = plt.gca()

        for entity in entities:
            plot_arena(axes, entities=entities, arena_size=[2, 2], beacons=beacons)
            robot_pose = np.array([[entity['x']], [entity['y']], [entity['theta']]])

            current_x, current_y, current_theta = entity['x'], entity['y'], entity['theta'] #getLocalizerPose()
            encoders = entity['encoders']
            
            # yucky code that moves the robot in a square (clean up b4 real testing)
            if t < 1:
                robot_pose = setVelocity(entity['width'], .25, .25, current_pose=robot_pose)
            if t > 1 and t < 2.5:
                robot_pose = setVelocity(entity['width'], 0,0.165, current_pose=robot_pose)

            if t > 2.5 and t < 3.5:
                robot_pose = setVelocity(entity['width'], 0.25,0.25, current_pose=robot_pose)
            if t > 3.5 and t < 5:
                robot_pose = setVelocity(entity['width'], 0,0.165, current_pose=robot_pose)
            
            if t > 5 and t < 6:
                robot_pose = setVelocity(entity['width'], 0.25,0.25, current_pose=robot_pose)
            if t > 6 and t < 7.5:
                robot_pose = setVelocity(entity['width'], 0,0.165, current_pose=robot_pose)
            
            if t > 7.5 and t < 8.5:
                robot_pose = setVelocity(entity['width'], 0.25,0.25, current_pose=robot_pose)
            if t > 8.5 and t < 10:
                robot_pose = setVelocity(entity['width'], 0,0.165, current_pose=robot_pose)

            if t > 10 and t < 11:
                            robot_pose = setVelocity(entity['width'], 0.25,0.25, current_pose=robot_pose)

            print(f"Plotting triangle at ({current_x}, {current_y}) with angle {wrap_angle(np.degrees(current_theta))} in degrees")

            entity['x'] = robot_pose[0, 0]
            entity['y'] = robot_pose[1, 0]        
            entity['theta'] = robot_pose[2, 0]

        if t > 120:
            break

    plt.ioff() 



