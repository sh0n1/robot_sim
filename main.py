from controllers import *
from plot_arena import render_history

import argparse
import time


def build_demo_problem():
    entities = [
        {'id': 'car1', 'id_num': 1, 'show_id': True, 'show_arrows': False, 'color': 'r', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.25, 'theta': np.radians(0)},
        {'id': 'car2', 'id_num': 2, 'show_id': True, 'show_arrows': False, 'color': 'g', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.40, 'theta': np.radians(0)},
        {'id': 'car3', 'id_num': 3, 'show_id': True, 'show_arrows': False, 'color': 'm', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.55, 'theta': np.radians(0)},
        {'id': 'car4', 'id_num': 4, 'show_id': True, 'show_arrows': False, 'color': 'c', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.70, 'theta': np.radians(0)},
        {'id': 'car5', 'id_num': 5, 'show_id': True, 'show_arrows': False, 'color': 'y', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.85, 'theta': np.radians(0)},
        {'id': 'car6', 'id_num': 6, 'show_id': True, 'show_arrows': False, 'color': 'tab:orange', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1., 'theta': np.radians(0)},
        {'id': 'car7', 'id_num': 7, 'show_id': True, 'show_arrows': False, 'color': 'tab:purple', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.15, 'theta': np.radians(0)},
        {'id': 'car8', 'id_num': 8, 'show_id': True, 'show_arrows': False, 'color': 'tab:brown', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.3, 'theta': np.radians(0)},
        {'id': 'car9', 'id_num': 9, 'show_id': True, 'show_arrows': False, 'color': 'tab:pink', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.45, 'theta': np.radians(0)},
        {'id': 'car10', 'id_num': 10, 'show_id': True, 'show_arrows': False, 'color': 'tab:gray', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.6, 'theta': np.radians(0)},
        {'id': 'car11', 'id_num': 11, 'show_id': True, 'show_arrows': False, 'color': 'tab:olive', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.75, 'theta': np.radians(0)},
    ]

    a = 0.7
    theta = np.linspace(0, 2 * np.pi, 300)
    x_values = a * np.cos(theta) / (np.sin(theta)**2 + 1) + 1
    y_values = a * np.sin(theta) * np.cos(theta) / (np.sin(theta)**2 + 1) + 1
    lemniscate_traj = np.array([[x_values], [y_values]])
    lemniscate = PursuitLine(x_points=x_values, y_points=y_values)

    goal_x = 1.25
    goal_y = 1.2
    goal = np.array([[goal_x], [goal_y]])

    trajectories = [
        ("Follow Lemniscate", lemniscate_traj),
        ("Move to Point", goal),
    ]

    return entities, trajectories, lemniscate, goal


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Robot simulation demo')
    parser.add_argument('--render-stride', type=int, default=1, help='Render every K simulated frames')
    parser.add_argument('--playback-speed', type=float, default=10.0, help='Replay speed relative to simulated time')
    parser.add_argument('--no-render', action='store_true', help='Run the simulation without opening the renderer')
    parser.add_argument('--perf', action='store_true', help='Print simulation and playback timing information')
    parser.add_argument('--perf-interval', type=int, default=100, help='Print a progress line every N simulated frames')
    parser.add_argument('--perf-runs', type=int, default=1, help='Run the simulation N times and average the performance')
    args = parser.parse_args()

    entities, trajectories, lemniscate, goal = build_demo_problem()

    if args.perf:
        print('Performance test: simulation starting')

    sim_durations = []
    history = []
    final_entities = []
    for run_index in range(max(1, args.perf_runs)):
        if args.perf and args.perf_runs > 1:
            print(f'Performance test: simulation run {run_index + 1}/{args.perf_runs}')

        sim_start = time.perf_counter()
        history, final_entities = simulate_pure_pursuit(
            entities,
            trajectories,
            lemniscate,
            0.2,
            25,
            KL=5,
            goal=(goal[0, 0], goal[1,0]),
            progress_interval=args.perf_interval if args.perf else None,
        )
        sim_end = time.perf_counter()
        sim_durations.append(sim_end - sim_start)

    if args.perf:
        sim_duration = sum(sim_durations) / len(sim_durations)
        sim_total = sum(sim_durations)
        frame_count = len(history)
        simulated_time = history[-1]['time'] if history else 0.0
        print(f'Simulation complete: {frame_count} frames per run')
        print(f'Average simulation time over {len(sim_durations)} run(s): {sim_duration:.4f}s')
        print(f'Total simulation time across runs: {sim_total:.4f}s')
        print(f'Simulation throughput: {frame_count / sim_duration:.2f} frames/s')
        print(f'Simulated timeline: {simulated_time:.2f}s')

    if args.perf:
        if args.no_render:
            print('Performance test: playback skipped')
        else:
            print('Performance test: playback starting')

    if not args.no_render:
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
    
