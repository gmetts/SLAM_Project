'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021






    The code for this project very closely follows HW3. For now the only change is updated the
    measurement function as spoken to in our project report. This included updating the landmark
    measurement function as is reflected in bearing_range_estimation(). This update to the
    landmark measurement function does not change the Jacobian that was derived in homework 3.
    The plan was to processing and input data from the UGV run in the same/similar format as
    this homework assignment. These updates were going to be added as data was processed. This 
    will be done in future work as the data is processed.
'''

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def warp2pi(angle_rad):
    """
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor(
        (angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    '''
    Initialize the state vector given odometry and observations.
    '''
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks), dtype=bool) #np.bool

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta) in the shape (2, )
    '''
    # TODO: return odometry estimation

    x0 = x[i]
    y0 = x[i+1]
    x1 = x[i+2]
    y1 = x[i+3]
    odom = np.array([x1-x0, y1-y0])

    return odom


def bearing_range_estimation(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks and angle (added an additional state)
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    '''
    # TODO: return bearing range estimations
    #Get positions from states
    rx, ry = x[i], x[i+1]
    lx, ly = x[j], x[j+1]

    #Adding in angle contraint
    r_theta = x[i+2]

    theta = warp2pi(np.arctan2(ly-ry, lx-rx)) + r_theta
    d = np.sqrt((lx-rx)**2 + (ly-ry)**2)

    obs = np.array([theta, d])

    return obs


def compute_meas_obs_jacobian(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    '''
    # TODO: return jacobian matrix
    #Get positions from states

    #this Jacobian Matrix in the same as HW3 as the angle terms in the 
    #bearing angle in the landmark function goes to zero

    rx, ry = x[i], x[i+1]
    lx, ly = x[j], x[j+1]

    jacobian = np.zeros((2, 4))

    d = (lx-rx)**2+(ly-ry)**2

    jacobian[0,0] = (ly-ry)/d
    jacobian[0,1] = -(lx-rx)/d
    jacobian[0,2] = -(ly-ry)/d
    jacobian[0,3] = (lx-rx)/d

    jacobian[1,0] = -(lx-rx)/np.sqrt(d)
    jacobian[1,1] = -(ly-ry)/np.sqrt(d)
    jacobian[1,2] = (lx-rx)/np.sqrt(d)
    jacobian[1,3] = (ly-ry)/np.sqrt(d)

    return jacobian


def create_linear_system(x, odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param x State vector x at which we linearize the system.
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    '''

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))
    sigma_odom = sqrt_inv_odom[0,0]
    sigma_obs = sqrt_inv_obs[0,0]

    # TODO:
    A = np.zeros((M, N))
    b = np.zeros((M, ))

    A[0,0] = sigma_odom*1
    A[1,1] = sigma_odom*1

    j = 0
    counter = 0
    num_obs = 0
    row = n_poses*2
    for i in range(2, M-5, 2):
        # TODO: 
        if i <= n_poses*2-2:
            #get odometry estimation
            e_odom = odometry_estimation(x, j)

            #Add to A matrix
            A[i,j] = sigma_odom*-1
            A[i,j+2] = sigma_odom*1
            A[i+1,j+1] = sigma_odom*-1
            A[i+1,j+3] = sigma_odom*1

            #Add to b matrix
            b[i] = sigma_odom*(odom[counter, 0] - e_odom[0])
            b[i+1] = sigma_odom*(odom[counter, 1] - e_odom[1])
            
            #update counters
            counter += 1
            j += 2


        # TODO: Then fill in landmark measurements
        else:
            #get pose and landmark index
            pose_index = observations[num_obs,0].astype(int)
            landmark_index = observations[num_obs,1].astype(int)
            landmark_index_x = landmark_index*2 + (n_poses)*2
            pose_index_x = pose_index*2
            
            #get Jacobian for corresponding states and landmark measurements
            jacobian = compute_meas_obs_jacobian(x, pose_index_x, landmark_index_x, n_poses)
            e_obs = bearing_range_estimation(x, pose_index_x, landmark_index_x, n_poses) #this is whith the robot angle taken into account
            theta = e_obs[0]
            d = e_obs[1]


            #Add to A matrix
            jacobian = sqrt_inv_obs @ jacobian

            #position in the state vector
            landmark_index_y = landmark_index_x + 1
            pose_index_y = pose_index_x + 1

            #fill in landmark measurements
            A[row, pose_index_x] = jacobian[0,0]
            A[row, pose_index_y] = jacobian[0,1]
            A[row, landmark_index_x] = jacobian[0,2]
            A[row, landmark_index_y] = jacobian[0,3]

            A[row+1, pose_index_x] = jacobian[1,0]
            A[row+1, pose_index_y] = jacobian[1,1]
            A[row+1, landmark_index_x] = jacobian[1,2]
            A[row+1, landmark_index_y] = jacobian[1,3]

            #Add to b matrix
            z_x = observations[num_obs,2]
            z_y = observations[num_obs,3]
            b[i] = warp2pi(sigma_obs*(z_x - theta))
            b[i+1] = sigma_obs*(z_y - d)

            #update counter
            num_obs += 1
            row += 2

    return csr_matrix(A), b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/mkchappe/SLAM/16833_HW3_SOLVERS_DATA/data/2d_nonlinear.npz')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['default'],
        help='method')

    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    # plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-')
    # plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='b', marker='+')
    # plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f'Applying {method}')
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)
        print('Before optimization')
        #plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks, 'lu_before') #uncomment this

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in range(1000):
            A, b = create_linear_system(x, odom, observations, sigma_odom,
                                        sigma_landmark, n_poses, n_landmarks)
            dx, _ = solve(A, b, method)
            x = x + dx

        traj, landmarks = devectorize_state(x, n_poses)
        print('After optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks, 'lu_after')