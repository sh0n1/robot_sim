from robot_movement import *
from typing import List,Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import time
import math
from queue import Queue, PriorityQueue

'''
def plot_triangle(axes:Axes, x, y, theta, length, width, color='b'):
    tip = (length / 2, 0)
    rear_left = (-length / 2, -width / 2)
    rear_right = (-length / 2, width / 2)
    triangle = Polygon([tip, rear_right, rear_left])

    centroid = triangle.centroid
    triangle = affinity.translate(triangle, xoff=-centroid.x, yoff=-centroid.y)
    rotated = affinity.rotate(triangle, theta, origin='centroid', use_radians=True)

    
    transformed = affinity.translate(rotated, xoff=x, yoff=y)

    x, y = transformed.exterior.xy
    axes.plot(x, y, color)
    
'''
def plot_triangle(x, y, theta, base_length: float, height: float, color:str, axes=None, coordinate_frame:bool=False, triangle:bool=True):
    """
    Plot an isosceles triangle indicating a provided base length, height, position,
    rotation and color of plot on an existing axis (if provided).

    An isosceles triangle has two sides that are the same length, with the third
    side being referred to as the base.

    Parameters
    ----------
    theta
        Rotation of the triangle in radians.
    translation
        A 2x1 matrix describing a translation in a 2D plane.
    ax
        The axes object to plot the triangle on. If None, a new plot will be created.
    """
    vertices = np.vstack([[0, 0, 1], [-height, base_length/2, 1], [-height, -base_length/2, 1]])

    centroid = np.mean(vertices[:, :2], axis=0)
    vertices[:, :2] -= centroid # make centroid the origin
    translation = np.array([[x], [y]])

    origin = np.array([[0, 0, 1]]).T   # Origin of the triangle
    x_axis = np.array([[0.1, 0, 1]]).T  # Local x-axis 
    y_axis = np.array([[0, 0.1, 1]]).T  # Local y-axis 

    T = SE2(theta, translation)

    Tvertices = np.array([T @ v for v in vertices])
    Tvertices = np.vstack((Tvertices, Tvertices[0]))

    T_origin = (T @ origin).flatten()
    T_x_axis = (T @ x_axis).flatten()
    T_y_axis = (T @ y_axis).flatten()

    if axes is None:
        axes = plt.gca()  # Get the current axes if none are provided

    if triangle:
        axes.plot(Tvertices[:, 0], Tvertices[:, 1], color)
        axes.plot(Tvertices[:-1, 0], Tvertices[:-1, 1], 'k.', markersize=6)

    if coordinate_frame:
        axes.quiver(
            T_origin[0], T_origin[1],  # Start at triangle's origin
            T_x_axis[0] - T_origin[0], T_x_axis[1] - T_origin[1],  # X-axis vector
            color='r', angles='xy', scale_units='xy', scale=1, width=0.005, label='X-axis'
        )
        axes.quiver(
            T_origin[0], T_origin[1],  # Start at triangle's origin
            T_y_axis[0] - T_origin[0], T_y_axis[1] - T_origin[1],  # Y-axis vector
            color='b', angles='xy', scale_units='xy', scale=1, width=0.005, label='Y-axis'
        )
    
    robot_pose = np.array([[translation[0,0]], [translation[1,0]], [(theta)]])
    # print("tri pose wrt global: \n", robot_pose)
    return robot_pose

def plot_arena(axes:Axes, 
               entities:List[dict]=[], 
               arena_size:List=[], 
               trajectories:List[Tuple[str,np.ndarray]]=[],
               beacons:List[dict]=[],
               obstacles:List[Tuple]=[],
               buf_obstacles:List[Tuple]=[],
               obstacle_positions:List[Tuple[str,np.ndarray]]=[]):
    """
    Plot the arena with entities and obstacles.
    
    This function acts as a sort of localiser view of the arena
    """
    seperate = arena_size[0]*1.05 - arena_size[0]
    # Clear the AXES then set the limits and aspect ratio
    #plt.clf()
    axes.clear()
    axes.set_xlim(0-seperate, arena_size[0]+seperate)
    axes.set_ylim(0-seperate, arena_size[1]+seperate)
    axes.set_aspect('equal')
    axes.set_title("Arena")
    axes.set_xlabel("X-axis")
    axes.set_ylabel("Y-axis")
    axes.grid(True)

    # Draw the arena boundary
    axes.plot([0, arena_size[0], arena_size[0], 0, 0], 
              [0, 0, arena_size[1], arena_size[1], 0], 'r-')
    
    # Plot the trajectories
    for trajectory in trajectories:
        if len(np.shape(trajectory[1])) > 2:
            title = trajectory[0]
            points = trajectory[1]

            axes.plot(points[0,0], points[1,0], linestyle=':', color='green')
            axes.set_title(title)
        else:
            title = trajectory[0]
            point = trajectory[1]
            axes.plot(point[0,0], point[1,0], 'or', markersize=5)
            axes.set_title(title)

    # Plot ground-truth position of beacons
    for beacon in beacons:
        axes.plot(beacon['x'], beacon['y'], 'kx', markersize=5)
        axes.text(beacon['x'], beacon['y'], beacon['id'], fontsize=7, ha='right')
    
    # Plot the entities starting positions
    for entity in entities:
        x, y, theta = entity['x'], entity['y'], entity['theta'] 
        length, width = entity['length'], entity['width']
        plot_triangle(x, y, theta, base_length=width, height=length, color='b', axes=axes, coordinate_frame=True)
        
    plt.grid()
    plt.draw()  
    plt.pause(0.0001)

def plot_noarena(axes:Axes,
                 entities:List[dict]=[], 
                 trajectories:List[Tuple[str,np.ndarray]]=[],
                 current_time:float=0.0,
                 interval:float=1):
    #axes.clear()
    axes.set_title("Arena")
    axes.set_xlabel("X-axis")
    axes.set_ylabel("Y-axis")

    # Plot the trajectories
    for trajectory in trajectories:
        if len(np.shape(trajectory[1])) > 2:
            title = trajectory[0]
            points = trajectory[1]

            axes.plot(points[0,0], points[1,0], linestyle=':', color='green')
            axes.set_title(title)
        else:
            title = trajectory[0]
            point = trajectory[1]
            axes.plot(point[0,0], point[1,0], 'or', markersize=5)
            axes.set_title(title)

    # Plot the entities starting positions
    if np.isclose(current_time%interval, 0, atol=0.035):
        for entity in entities:
            print(f"Entity: {entity['id']}")
            x, y, theta = entity['x'], entity['y'], entity['theta'] 
            length, width = entity['length'], entity['width']
            plot_triangle(x, y, theta, base_length=width, height=length, color='b', axes=axes, coordinate_frame=True, triangle=False)
            
    plt.draw()  
    plt.pause(0.0001)


def render_history(history, trajectories=None, arena_size=None, beacons=None, playback_speed=10.0, stride=1, min_pause=0.02):
    """Render a previously simulated frame log after the simulation completes."""
    if trajectories is None:
        trajectories = []
    if arena_size is None:
        arena_size = [2, 2]
    if beacons is None:
        beacons = []

    plt.ion()
    plt.figure()
    axes = plt.gca()

    previous_time = None
    for frame in history[::stride]:
        plot_arena(
            axes,
            entities=frame['entities'],
            arena_size=arena_size,
            trajectories=trajectories,
            beacons=beacons,
        )
        axes.set_title(f"Arena - t={frame['time']:.2f}s")

        if previous_time is None:
            pause = min_pause
        else:
            simulated_delta = max(frame['time'] - previous_time, 0.0)
            pause = max(simulated_delta / playback_speed, min_pause)
        previous_time = frame['time']

        plt.pause(pause)

    plt.ioff()
