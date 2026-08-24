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


def _normalize_follow_trajs(entities, follow_trajs):
    if hasattr(follow_trajs, 'find_closest_index'):
        return [follow_trajs for _ in entities]

    follow_trajs = list(follow_trajs)
    if len(follow_trajs) != len(entities):
        raise ValueError('Expected one follow trajectory per entity')

    return follow_trajs


def _step_pure_pursuit_entity(entity, follow_traj, follow_RAD, target_v, KL=10, goal=None):
    next_entity = entity.copy()

    if next_entity.get('reached_goal', False):
        return next_entity, True

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
        next_entity['reached_goal'] = True

    next_entity['x'] = robot_pose[0, 0]
    next_entity['y'] = robot_pose[1, 0]
    next_entity['theta'] = robot_pose[2, 0]

    return next_entity, reached_goal


def simulate_pure_pursuit(entities, trajectories, follow_trajs, follow_RAD, target_v, KL=10, goal=None, max_steps=2400, dt=0.05, progress_interval=None):
    """Run pure pursuit without rendering and return a frame log."""
    current_entities = deepcopy(entities)
    start_time = time.perf_counter()

    follow_trajs = _normalize_follow_trajs(current_entities, follow_trajs)

    for entity, follow_traj in zip(current_entities, follow_trajs):
        entity['current_index'] = follow_traj.find_closest_index((entity['x'], entity['y']))
        entity['reached_goal'] = False

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
        goal_reached = True if goal is not None else False
        for entity, follow_traj in zip(current_entities, follow_trajs):
            next_entity, reached_goal = _step_pure_pursuit_entity(
                entity,
                follow_traj,
                follow_RAD,
                target_v,
                KL=KL,
                goal=goal,
            )
            next_entities.append(next_entity)
            goal_reached = goal_reached and reached_goal if goal is not None else False

        current_entities = next_entities
        if goal is not None and goal_reached:
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

def create_obstacle_grid(obstacles, N, BUFFER):
    # Create a 101x101 obstacle grid
    obstacle_grid = np.zeros((N, N), dtype=np.float64)
    # Loop through obstacles and mark cells as occupied
    obs_buffered = []
    for obstacle in obstacles:
        xmin, xmax, ymin, ymax = obstacle

        # Add buffer in metres
        xmin_m = max(0, xmin - BUFFER*0.02)  # Clamp xmin to 0
        xmax_m = min(2, xmax + BUFFER*0.02)  # Clamp xmax to 2
        ymin_m = max(0, ymin - BUFFER*0.02)  # Clamp ymin to 0
        ymax_m = min(2, ymax + BUFFER*0.02)  # Clamp ymax to 2

        # Convert to cells
        # Adding Buffer around obstacle
        xmin = max(0,(int(xmin*100/2) - BUFFER))
        xmax = min(N-1,(int(xmax*100/2) + BUFFER))
        ymin = max(0,(int(ymin*100/2) - BUFFER))
        ymax = min(N-1,(int(ymax*100/2) + BUFFER))

        # Mark the corresponding cells as occupied
        # Add 1 to max to include max in splicing
        obstacle_grid[ymin:ymax+1, xmin:xmax+1] = 1
        # Append buffered obstacles:
        obs_buffered.append((xmin_m, xmax_m, ymin_m, ymax_m))

    # Add border around the grid
    obstacle_grid[:BUFFER, :] = 1  # Top border
    obstacle_grid[-BUFFER:, :] = 1  # Bottom border
    obstacle_grid[:, :BUFFER] = 1  # Left border
    obstacle_grid[:, -BUFFER:] = 1  # Right border
    return obstacle_grid, obs_buffered, obstacles

def distance_transform(map:np.ndarray,goal:np.ndarray, N:int):
    """
    Compute the distance transform from the provided map and goal location.

    parameters
    ----------
    map
        An np.ndarray of shape (N,N) representing an occupancy grid. Values of 1
        indicate occupied space (i.e. a wall), and values of 0 represent empty
        space.
    goal
        An np.ndarray of shape (2,) giving the x-y coordinates of the goal.

    Outputs
    -------
    distance_map
        An np.ndarray the same shape as map where each value represents the
        distance to the goal. Any walls should have the value np.nan.
    """
    distance_map = np.full_like(map, np.inf)  # Initialize distance map with NaNs
    print(goal)
    goal_x = int(goal[0]*100/2)
    goal_y = int(goal[1]*100/2)
    #? Swap x and y coz rows and collumns matrix
    distance_map[goal_y, goal_x] = 0  # Distance to goal is 0
    # Directions in surrounding 8 cells
    directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)])


    queue = Queue()
    queue.put((goal_x,goal_y))

    while not queue.empty():
        current_x, current_y = queue.get()

        for dx, dy in directions:
            new_x, new_y = current_x + dx, current_y + dy

            # Check if the new position is within the map boundaries and not a wall
            if 0 <= new_x < N and 0 <= new_y < N and map[new_y, new_x] == 0:
                # Calculate the distance based on Euclidean measure
                distance = np.sqrt(dx ** 2 + dy ** 2) if dx != 0 and dy != 0 else 1
                new_distance = distance_map[current_y, current_x] + distance

                # Update the distance if it's shorter
                if np.isnan(distance_map[new_y, new_x]) or new_distance < distance_map[new_y, new_x]:
                    distance_map[new_y, new_x] = new_distance
                    queue.put((new_x, new_y))

    return distance_map

def init_path(distance_map, start, N):
    path = search_path(distance_map, start, N)
    path_x = path[:, 0]
    path_y = path[:, 1]
    # For both path_x and path_y, need to undo int(start[0]*100/2) convertion
    xpath = path_x / 100 * 2
    ypath = path_y / 100 * 2
    return path, path_x, path_y, xpath, ypath

def search_path(distance_map: np.ndarray, start: np.ndarray, N: int):
    """
    Search for a path from the start position to the goal position using the distance map.

    Parameters
    ----------
    distance_map : np.ndarray
        An array of shape (N,N) representing the distance map.
    start : np.ndarray
        An array of shape (2,) giving the x-y coordinates of the start position.

    Returns
    -------
    path : np.ndarray
        An array of shape (P, 2) representing the path from start to goal, where P is the number of points in the path.
    """
    path = np.array([], dtype=int).reshape(0, 2)
    # Directions in surrounding 8 cells
    directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)])

    # Convert start position to integer coordinates
    current_x = int(start[0]*100/2)
    current_y = int(start[1]*100/2)

    while True:
        # Add the current position to the path
        path = np.vstack([path, [current_x, current_y]])

        min_distance = np.inf
        next_x, next_y = current_x, current_y

        # Search for the neighboring cell with the minimum distance
        for dx, dy in directions:
            new_x, new_y = current_x + dx, current_y + dy

            # Check if the new position is within the map boundaries
            if 0 <= new_x < N and 0 <= new_y < N:
                distance = distance_map[new_y, new_x]

                # Update the minimum distance and next position
                if distance < min_distance:
                    min_distance = distance
                    next_x, next_y = new_x, new_y

        # If the minimum distance is still infinity, no valid path is found
        if min_distance == np.inf:
            print("No valid path found")
            break

        # Move to the next position
        current_x, current_y = next_x, next_y

        # If the current position is the goal, break out of the loop
        if min_distance == 0:
            print("Path to goal done")
            break

    return path