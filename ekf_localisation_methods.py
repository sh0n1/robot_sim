import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# Provided helper functions for this question

def wrapToPi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def sense(k, num_landmarks, sensor):
    idx = (k) * num_landmarks
    z = sensor[idx:idx + num_landmarks, :]
    return z


def get_odom(odom):
    """
    odom is the odometry_d function
    """
    d = odom[0]
    dth = odom[1]
    return d, dth


def ask_the_oracle(timestep, true_pose):
    # column
    current_pose = true_pose[:, timestep]
    current_pose = current_pose.reshape((3, 1))
    return current_pose


# Question 1
def predict_step(
        x: np.array,
        S: np.array,
        d: float,
        dth: float,
        R: np.array):
    """
    This function performs a prediction step of the EKF localiser

    Parameters
    ----------
    x
        the mean of the robot pose
    S
        the covariance matrix of the robot pose
    d
        distance travelled in timestep
    dth
        change of heading
    R
        the covariance matrix of odometry noise

    Outputs
    -------
    x
        the mean of the robot pose
    S
        the covariance of the robot pose
    """
    theta = x[2, 0]
    x_pred = x + np.array([[d * np.cos(theta)],
                           [d * np.sin(theta)],
                           [dth]])
    x_pred[2, 0] = wrapToPi(x_pred[2, 0])

    # Jacobian of the motion model
    Fx = np.array([[1, 0, -d * np.sin(theta)],
                   [0, 1, d * np.cos(theta)],
                   [0, 0, 1]])

    R_extended = np.zeros((3, 3))
    R_extended[:2, :2] = R
    # Update covariance
    S_pred = Fx @ S @ Fx.T + R_extended

    return x_pred, S_pred

def update_step(
        landmark_map: np.array,
        z: np.array,
        x: np.array,
        S: np.array,
        Q: np.array):
    """
    This function performs an update step of the EKF localiser

    Parameters
    ----------
    landmark_map
        location of all landmarks
    z
        An np.ndarray condaining the range and bearing to all landmarks in
        map at current timestep
    x
        the mean of the robot pose
    S
        the covariance matrix of the robot pose
    Q
        the covariance matrix of sensor noise

    Outputs
    -------
    x
        the mean of the robot pose
    S
        the covariance of the robot pose
    """

    theta = x[2, 0]
    H = np.zeros((2 * landmark_map.shape[0], 3))
    z_pred = []

    for i, (landmark_x, landmark_y) in enumerate(landmark_map):
        delta_x = landmark_x - x[0, 0]
        delta_y = landmark_y - x[1, 0]
        q = delta_x ** 2 + delta_y ** 2
        z_pred_i = [np.sqrt(q), wrapToPi(np.arctan2(delta_y, delta_x) - theta)]
        z_pred.append(z_pred_i)

        H_i = np.array([[-delta_x / np.sqrt(q), -delta_y / np.sqrt(q), 0],
                        [delta_y / q, -delta_x / q, -1]])
        H[2 * i:2 * i + 2, :] = H_i

    z_pred = np.array(z_pred).flatten()
    z = z.flatten()

    # Innovation
    y = z - z_pred
    y[1::2] = wrapToPi(y[1::2])

    Q_extended = np.zeros((2 * landmark_map.shape[0], 2 * landmark_map.shape[0]))
    for i in range(landmark_map.shape[0]):
        Q_extended[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Q

    # Kalman gain

    S_z = H @ S @ H.T + Q_extended
    K = S @ H.T @ np.linalg.inv(S_z)

    # Update state

    x = x + K @ y.reshape((-1, 1))
    x[2, 0] = wrapToPi(x[2, 0])

    # Update covariance

    S = (np.eye(len(S)) - K @ H) @ S

    return x, S


def plot_results(true_trajectory, estimated_trajectory, landmarks):
    plt.figure(figsize=(10, 10))
    plt.plot(true_trajectory[0, :], true_trajectory[1, :], 'g-', label='True Trajectory')
    plt.plot(estimated_trajectory[0, :], estimated_trajectory[1, :], 'r--', label='Estimated Trajectory')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='b', marker='*', label='Landmarks')
    plt.legend()
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('EKF Localization')
    plt.grid()
    plt.show()

