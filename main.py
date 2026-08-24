from controllers import *
from plot_arena import render_history

import argparse
import time
import random

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Robot simulation demo')
    parser.add_argument('--render-stride', type=int, default=1, help='Render every K simulated frames')
    parser.add_argument('--playback-speed', type=float, default=10.0, help='Replay speed relative to simulated time')
    parser.add_argument('--no-render', action='store_true', help='Run the simulation without opening the renderer')
    parser.add_argument('--perf', action='store_true', help='Print simulation and playback timing information')
    parser.add_argument('--perf-interval', type=int, default=100, help='Print a progress line every N simulated frames')
    parser.add_argument('--perf-runs', type=int, default=1, help='Run the simulation N times and average the performance')
    args = parser.parse_args()

    # PURE PURSUIT LEMNISCATE WITH GOAL
    # a = 0.7
    # theta = np.linspace(0, 2 * np.pi, 300)
    # x_values = a * np.cos(theta) / (np.sin(theta)**2 + 1) + 1
    # y_values = a * np.sin(theta) * np.cos(theta) / (np.sin(theta)**2 + 1) + 1
    # lemniscate_traj = np.array([[x_values], [y_values]])
    # path = PursuitLine(x_points=x_values, y_points=y_values)

    goal_x = 1.5
    goal_y = 1.5
    goal = np.array([goal_x, goal_y])

    # PATH PLANNING
    N = 101 #for bench mark: 101, 251, 501, 1001, 2001
    buf = (N-1)//50  # The obstacle detection buffer. 5 at N=101

    arena_size = 3.0
    obstacles = [(0.5, 0.9, 0.7, 1),
                (1.3, 1.7, 1.1, 1.4),
                (0.7, 1.0, 1.3, 1.6),
                (2, 2.7, 1.5, 2.5),
                (1.7, 2.5, 0.7, 1.3),
                (1.9, 2.3, 0.4, 0.9),
                (0.3, 0.7, 1.9, 2.3),
                (1.4, 1.8, 2.2, 2.6),
                (0.1, 0.5, 2.5, 2.9),
                (1.0, 1.3, 0.1, 0.5),
                ]

    ob_grid, obs_buffered, obs_og = create_obstacle_grid(obstacles, N, buf, arena_size=arena_size)
    rng = random.Random(42)

    def cell_to_coord(cell_index):
        return cell_index * arena_size / (N - 1)

    def sample_spawn_position(free_cells, used_cells):
        available_indices = [index for index in range(len(free_cells)) if index not in used_cells]
        if not available_indices:
            raise ValueError('No free spawn cells left')

        chosen_index = rng.choice(available_indices)
        row, col = free_cells[chosen_index]
        used_cells.add(chosen_index)
        return cell_to_coord(col), cell_to_coord(row)

    # # FIXED SPAWNS
    # entities = [
    #         {'id': 'car1', 'id_num': 1, 'show_id': True, 'show_arrows': False, 'color': 'r', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.25, 'theta': np.radians(0)},
    #         {'id': 'car2', 'id_num': 2, 'show_id': True, 'show_arrows': False, 'color': 'g', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.40, 'theta': np.radians(0)},
    #         {'id': 'car3', 'id_num': 3, 'show_id': True, 'show_arrows': False, 'color': 'm', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.55, 'theta': np.radians(0)},
    #         {'id': 'car4', 'id_num': 4, 'show_id': True, 'show_arrows': False, 'color': 'c', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.70, 'theta': np.radians(0)},
    #         {'id': 'car5', 'id_num': 5, 'show_id': True, 'show_arrows': False, 'color': 'y', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 0.85, 'theta': np.radians(0)},
    #         {'id': 'car6', 'id_num': 6, 'show_id': True, 'show_arrows': False, 'color': 'tab:orange', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1., 'theta': np.radians(0)},
    #         {'id': 'car7', 'id_num': 7, 'show_id': True, 'show_arrows': False, 'color': 'tab:purple', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.15, 'theta': np.radians(0)},
    #         {'id': 'car8', 'id_num': 8, 'show_id': True, 'show_arrows': False, 'color': 'tab:brown', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.3, 'theta': np.radians(0)},
    #         {'id': 'car9', 'id_num': 9, 'show_id': True, 'show_arrows': False, 'color': 'tab:pink', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.45, 'theta': np.radians(0)},
    #         {'id': 'car10', 'id_num': 10, 'show_id': True, 'show_arrows': False, 'color': 'tab:gray', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.6, 'theta': np.radians(0)},
    #         {'id': 'car11', 'id_num': 11, 'show_id': True, 'show_arrows': False, 'color': 'tab:olive', 'length': 0.1, 'width': 0.1, 'x': 0.25, 'y': 1.75, 'theta': np.radians(0)},
    #     ]
    ####################################
    if args.perf:
        print('Performance test: simulation starting')

    sim_durations = []
    transform_durations = []
    history = []
    final_entities = []
    for run_index in range(max(1, args.perf_runs)):
        if args.perf and args.perf_runs > 1:
            print(f'Performance test: simulation run {run_index + 1}/{args.perf_runs}')

        distance_transform_start = time.perf_counter()
        distance_map = distance_transform(ob_grid, goal, N, arena_size=arena_size)
        distance_transform_end = time.perf_counter()
        transform_durations.append(distance_transform_end - distance_transform_start)

        free_cells = np.argwhere(np.isfinite(distance_map))
        entity_num = 10
        if len(free_cells) < entity_num:
            print(f"Not enough reachable free cells: {len(free_cells)} < {entity_num}")
            entity_num = max(1, len(free_cells))

        used_spawn_cells = set()
        entities = []
        for i in range(entity_num):
            spawn_x, spawn_y = sample_spawn_position(free_cells, used_spawn_cells)
            entities.append({
                'id': f'car{i+1}',
                'id_num': i+1,
                'show_id': True,
                'show_arrows': False,
                'color': random.choice(['r', 'g', 'b', 'm', 'c', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']),
                'length': 0.1,
                'width': 0.1,
                'x': spawn_x,
                'y': spawn_y,
                'theta': np.radians(random.uniform(0, 360)),
            })

        follow_trajs = []
        path_trajectories = []
        for entity in entities:
            _, _, _, xpath, ypath = init_path(distance_map, np.array([entity['x'], entity['y']]), N, arena_size=arena_size)
            follow_trajs.append(PursuitLine(x_points=xpath, y_points=ypath))
            path_trajectories.append((f"Path Finding {entity['id']}", np.array([[xpath], [ypath]])))

        trajectories = [
            ("GOAL", np.array([[goal[0]], [goal[1]]])),
            *path_trajectories,
        ]

        sim_start = time.perf_counter()
        history, final_entities = simulate_pure_pursuit(
            entities,
            trajectories,
            follow_trajs,
            0.2,
            15,
            KL=1000,
            goal=goal,
            arena_size=arena_size,
            progress_interval=args.perf_interval if args.perf else None,
        )
        sim_end = time.perf_counter()
        sim_durations.append(sim_end - sim_start)

    if args.perf:
        transform_total = sum(transform_durations)
        transform_avg = transform_total / len(transform_durations)
        sim_duration = sum(sim_durations) / len(sim_durations)
        sim_total = sum(sim_durations)
        frame_count = len(history)
        simulated_time = history[-1]['time'] if history else 0.0
        print(f'Distance transform time: {transform_durations[-1]:.4f}s')
        print(f'Total distance transform time across {len(transform_durations)} run(s): {transform_total:.4f}s')
        print(f'Average distance transform time: {transform_avg:.4f}s')
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
            arena_size=[3, 3],
            obstacles=obstacles,
            buf_obstacles=obs_buffered,
            obstacle_positions=obs_og,
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
    
