import numpy as np
import math

# MATH FUNCTIONS
def SE2(theta:float,translation:np.ndarray) -> np.ndarray:
    """
    Create an SE(2) matrix for a given angle and translation.

    Parameters
    ----------
    theta
        The angle of rotation in radians

    translation
        A 2x1 matrix describing a translation in a 2D plane.

    Returns
    -------
    np.ndarray
        The SE(2) matrix.
    """

    return np.array([[np.cos(theta), -np.sin(theta), translation[0, 0]],
                     [np.sin(theta), np.cos(theta), translation[1,0]],
                    [0, 0, 1]], dtype=float)

def using_SE2(T_AB:np.ndarray,P_A:np.ndarray) -> np.ndarray:
    """
    Compute the position of point P in frame B.

    Parameters
    ----------
    T_AB
        An SE2 matrix representing the rotation from frame A to frame B.
    P_A
        The position of point P in coordinate frame A.

    Returns
    -------
    np.ndarray
        A 2x1 matrix descibing the point P in frame B.
    """
    T = np.linalg.inv(T_AB).dot(np.vstack((P_A[0], P_A[1], 1)))
    return T[0:2]

def wrap_angle(angle):
    # wraps angle between -180 to 180 deg, use for robot turn calculations
    return (angle + 180) % 360 - 180

def wrap_angle_360(angle):
    # wraps angle between 0 to 360 deg, use for printing robot angle
    return (angle) % 360

def setVelocity(W, motor_left=0, motor_right=0, dt=0.05, current_pose=np.array([[0], [0], [0]])):
    """
    Update the robot's pose based on skid-steering kinematics.

    Parameters
    ----------
    motor_left : float
        Speed of the left motor.
    motor_right : float
        Speed of the right motor.
    wheel_base : float
        Distance between the left and right wheels.
    dt : float
        Time step for updating position.
    current_pose : np.ndarray
        Current pose of the robot (x, y, theta in radians).

    Returns
    -------
    new_pose : np.ndarray
        Updated pose after applying velocities.
    """

    v = (motor_right + motor_left) / 2  # Linear velocity
    omega = (motor_right - motor_left) / W  # Angular velocity

    x, y, theta = current_pose.flatten()

    # Update position using simple integration
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt

    new_pose = np.array([[x_new], [y_new], [theta_new]])

    return new_pose

def lock_angle(max_angle, angle):
    # Ensure the angle is within the range of -max_angle to max_angle
    if angle > max_angle:
        return max_angle
    elif angle < -max_angle:
        return -max_angle
    else:
        return angle
    
def turnwheels_sim(robot_pose, velocity,theta, W):
    # the wheel speeds, to turn at a certain speed, a certain degree
    left = velocity - (W/2)*theta
    right = velocity + (W/2)*theta
    # print("left wheel speed: ", left, "right wheel speed: ", right)
    return setVelocity(W, float(left/100),float(right/100), current_pose=robot_pose) # Assume sim speed:real speed ratio is 100:1

def angle_to_point(x,y,theta,point):
        delta_x = point[0] - x
        delta_y = point[1] - y
        angle_to_point = np.degrees(np.arctan2(delta_y, delta_x))
        angle_difference = wrap_angle(angle_to_point - theta)
        return angle_difference # in degrees

def is_close_wall(x, y, tol=0.05, arena_size=2.0):
    # When localiser not detected
    if (x==0 or y==0):
        return True
    # Check if x is close to the arena edges within the specified tolerance
    if (math.isclose(x, 0, abs_tol=tol) or math.isclose(x, arena_size, abs_tol=tol)):
        return True
    # Check if y is close to the arena edges within the specified tolerance
    if (math.isclose(y, 0, abs_tol=tol) or math.isclose(y, arena_size, abs_tol=tol)):
        return True
    # Else not close to wall
    return False

def is_close_point(x, y, px, py, tol=0.21):
    # Check if x is close goal
    if (math.isclose(x, px, abs_tol=tol)):
        # Check if y is close to goal
        if (math.isclose(y, py, abs_tol=tol)):
            return True
    # Else not close to goal
    return False

class PursuitLine:
    def __init__(self, x_points, y_points) -> None:
        self.x = np.array(x_points)
        self.y = np.array(y_points)
        self.length = len(self.x)
    
    def show(self, ax4) -> None:
        ax4.plot(self.x, self.y, linestyle='--', color='orange')

    def find_closest_index(self, point: tuple[2]):
        point_X, point_Y = *point,
        distances = np.sqrt((self.x - point_X)**2 + (self.y - point_Y)**2)
        return list(distances).index(min(distances))

    def get_point(self, index):
        index = index % self.length
        return self.x[index], self.y[index]

    def _distance(self, point, index):
        index = index % self.length
        d = np.sqrt((point[1] - self.y[index])**2 + (point[0] - self.x[index])**2)
        return d

    def get_next_closest_point(self, index, current_position, radius):
        if self._distance(current_position, index + 1) <= radius:
            # print(f"To the next point - {index + 1}")
            return index + 1
        return index

    def get_distance_to_path(self, index, current_position):
        return self._distance(current_position, index + 1)

    def get_trajectory(self):
        return self.x, self.y
