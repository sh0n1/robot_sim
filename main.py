from controllers import *
from plot_arena import render_history

import argparse
import time

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Robot simulation demo')
    parser.add_argument('--render-stride', type=int, default=1, help='Render every K simulated frames')
    parser.add_argument('--playback-speed', type=float, default=10.0, help='Replay speed relative to simulated time')
    parser.add_argument('--perf', action='store_true', help='Print simulation and playback timing information')
    parser.add_argument('--perf-interval', type=int, default=100, help='Print a progress line every N simulated frames')
    args = parser.parse_args()

    entities = [
    {'id': 'car1', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.25, 'theta': np.radians(0)},
    {'id': 'car2', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.40, 'theta': np.radians(0)},
    {'id': 'car3', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.55, 'theta': np.radians(0)},
    {'id': 'car4', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.70, 'theta': np.radians(0)},
    {'id': 'car5', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.85, 'theta': np.radians(0)},
    {'id': 'car6', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1., 'theta': np.radians(0)},
    {'id': 'car7', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.15, 'theta': np.radians(0)},
    {'id': 'car8', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.3, 'theta': np.radians(0)},
    {'id': 'car9', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.45, 'theta': np.radians(0)},
    {'id': 'car10', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.6, 'theta': np.radians(0)},
    {'id': 'car11', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.75, 'theta': np.radians(0)},



    ] # max 3 cars, very slow otherwise

    # LINE TRAJECTORY
    A,B,C = (1, -1, 0) # change these values to whatever
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
    goal_x = 0.3
    goal_y = 1
    goal = np.array([[goal_x], [goal_y]])
    goal_name = "Move to Point"

    trajectories = [
        (circle_name, circle_traj),
        (goal_name, goal),
    ]

    if args.perf:
        print('Performance test: simulation starting')

    sim_start = time.perf_counter()
    history, final_entities = simulate_pure_pursuit(
        entities,
        trajectories,
        circle,
        0.1,
        25,
        KL=5,
        goal=[goal_x, goal_y],
        progress_interval=args.perf_interval if args.perf else None,
    )
    sim_end = time.perf_counter()

    if args.perf:
        sim_duration = sim_end - sim_start
        frame_count = len(history)
        simulated_time = history[-1]['time'] if history else 0.0
        print(f'Simulation complete: {frame_count} frames in {sim_duration:.4f}s')
        print(f'Simulation throughput: {frame_count / sim_duration:.2f} frames/s')
        print(f'Simulated timeline: {simulated_time:.2f}s')

    if args.perf:
        print('Performance test: playback starting')

    render_start = time.perf_counter()
    render_history(
        history,
        trajectories=trajectories,
        arena_size=[2, 2],
        playback_speed=args.playback_speed,
        stride=max(1, args.render_stride),
    )
    render_end = time.perf_counter()

    if args.perf:
        render_duration = render_end - render_start
        render_frames = max(1, len(history[::max(1, args.render_stride)]))
        print(f'Playback complete: {render_frames} frames in {render_duration:.4f}s')
        print(f'Playback throughput: {render_frames / render_duration:.2f} frames/s')

    plt.show()
    
