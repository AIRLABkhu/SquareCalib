import numpy as np 
import tf
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

def transform(params, points):
    theta = params[3:]
    translation = params[:3]
  
    quat = tf.transformations.quaternion_from_euler(theta[0], theta[1], theta[2])
    rotation_matrix = R.from_quat(quat).as_matrix()
    transformed_points = (rotation_matrix @ points.T).T  + translation

    return transformed_points.flatten()

def residuals(params):
    transformed = transform(params, stack_l_four_centers)
   
    return transformed - stack_cam_four_centers.flatten()

def sort_centers(pts):
    y_idxs = np.argsort(pts[:, 1]) # y 
    pts = pts[y_idxs]

    x_12_idxs = np.argsort(pts[:2, 0])
    x_12_pts = pts[x_12_idxs]

    x_34_idxs = np.argsort(pts[2:, 0])
    x_34_pts = pts[2:, :][x_34_idxs[::-1]]

    sorted_pts = np.concatenate([x_12_pts, x_34_pts], axis=0)
    return sorted_pts


stack_cam_four_centers = np.empty((0, 3), dtype=np.float32)
stack_l_four_centers = np.empty((0, 3), dtype=np.float32)


# Consider Only Single Calibration Data
for pose in range(1, 4):
    # Read camera 4 centers 
    cam_four_centers = np.load(f'sample/calib_features/rs_centers/{pose}.npy')

    # Read LiDAR 4 centers 
    l_four_centers = np.load(f'sample/calib_features/vlp_centers/{pose}.npy')
    
    # Transform LiDAR system to Camera system
    l_four_centers = l_four_centers.astype(np.float32)
    l_four_centers = (np.array([[0, -1, 0],
                               [0, 0, -1],
                               [1, 0, 0]]) @ l_four_centers.T).T
    cam_four_centers = cam_four_centers.astype(np.float32)

    # Top-left sort 
    cam_four_centers = sort_centers(cam_four_centers)
    l_four_centers = sort_centers(l_four_centers)

    # Accumulate
    stack_l_four_centers = np.append(stack_l_four_centers, l_four_centers, axis=0)
    stack_cam_four_centers = np.append(stack_cam_four_centers, cam_four_centers, axis=0)

    # Optimization
    initial_guess = [0, 0, 0, 0, 0, 0]
    result = least_squares(residuals, initial_guess, method='lm')

    # Result
    extrinsic_parameters = result.x
    print(extrinsic_parameters)

    np.save('sample/extrinisic_parameters/l2c_params.npy', extrinsic_parameters)

  