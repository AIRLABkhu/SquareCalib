import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy import optimize
import open3d as o3d 
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

np.random.seed(42)

def calc_cov_sqrt(cov):
    d, V = np.linalg.eig(cov)
    return V @ np.sqrt(np.diag(d)) @ np.linalg.inv(V)

def data_roi_setting(pcd_np):
    # pose 2
    pcd_np_y_min_mask = pcd_np[:,0] > 1.5
    pcd_np = pcd_np[pcd_np_y_min_mask]
    pcd_np_y_min_mask = pcd_np[:,0] < 5
    pcd_np = pcd_np[pcd_np_y_min_mask]
    pcd_np_y_min_mask = pcd_np[:,1] > -2
    pcd_np = pcd_np[pcd_np_y_min_mask]
    pcd_np_y_min_mask = pcd_np[:,1] < 2
    pcd_np = pcd_np[pcd_np_y_min_mask]

    return pcd_np

def edges_detector_laser_num(pts):
    l_range = set(pts[:, 3])
    board_sort_laser_pts = []
 
    
    for laser in l_range:
        mask = (pts[:, 3] == laser)
        masked = pts[mask]
        board_sort_laser_pts.append(masked)
    
    depth_discontinuous = []
    for laser in board_sort_laser_pts:
        thres = 0.6
        for i in range(len(laser) -2):
            i += 1

            prev = laser[i-1] 
            prev_d = np.sqrt(prev[0]**2 + prev[1]**2 + prev[2]**2)
            cur = laser[i]
            cur_d = np.sqrt(cur[0]**2 + cur[1]**2 + cur[2]**2)
            next = laser[i+1]

            next_d = np.sqrt(next[0]**2 + next[1]**2 + next[2]**2)

            temp = max(max(prev_d - cur_d, next_d -cur_d), 0)
            if temp > thres : 
                
                depth_discontinuous.append(cur)

   
    depth_discontinuous = np.array(depth_discontinuous)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(pts[:, 0], pts[:, 1], pts[:,2 ], c= 'b', alpha= 0.01)

    # ax.scatter(depth_discontinuous[:, 0], depth_discontinuous[:, 1], depth_discontinuous[:,2 ], c= 'r')
    # plt.show()
    edges = depth_discontinuous
    board = pts[:, :3]

    return board, edges

def filter_circle_edges(edges):
    circle_laser = np.empty((0, 4))
    for col_num in range(laser_ch) :
        pts = edges [edges[:, -1] == col_num]
        if len(pts) != 0 :
            filtered_pts = np.delete(pts, [0, -1], axis=0)
            if len(pts) != 0 :
                circle_laser = np.append(circle_laser, filtered_pts, axis=0)


    return circle_laser

def ransac_pca_rot(pts, circle_pts):
    MODE = 'RANSAC'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    labels = np.array(
        pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=True))

    target_point = circle_pts[0, :3] 

    boards_cluster = None
    for label in set(labels):
        if label == -1:
            continue  # -1은 outlier를 나타냅니다.

        cluster_indices = np.where(labels == label)[0]
        cluster = pcd.select_by_index(cluster_indices)
        is_point_present = np.any(np.all(np.isclose(np.asarray(cluster.points), target_point), axis=1))

        if is_point_present: # 만약에 저 세그먼트가 board의 세그먼트라면 
            if MODE == 'PCA':
                board_pts = np.asarray(cluster.points)
                covariance_matrix = np.cov(board_pts, rowvar=False)

                # NumPy의 linalg.eig 함수를 사용하여 고유값(eigenvalues)과 고유벡터(eigenvectors) 계산
                eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

                # 고유값과 고유벡터를 내림차순으로 정렬
                sorted_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
                return board_pts, eigenvectors
            
            elif MODE == 'RANSAC':
                plane_model, inliers = cluster.segment_plane(distance_threshold=0.01 , ransac_n=3, num_iterations=500) # Ransac
                board_pts = np.asarray(cluster.points)

                plane_normal = plane_model[0:3]
                a, b, c = plane_normal[0], plane_normal[1], plane_normal[2]
                v1 = np.array([a, b, c])
                v2 = np.array([0, 0, 1])
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                theta = np.arccos(cos_theta)

                rotation_vector = np.cross(v1, v2)
                rotation_matrix = Rotation.from_rotvec(rotation_vector * theta).as_matrix()
                normal_vec = v1

                return board_pts, rotation_matrix, normal_vec
                
def get_transformation_matrix_scipy(tx, ty, tz, rx, ry, rz):
    translation_matrix = np.array([[tx],
                                   [ty],
                                   [tz]])

    rotation_matrix = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    transformation_matrix = np.concatenate([rotation_matrix, translation_matrix], axis=1)
    margin = np.array([[0, 0, 0, 1]])
    transformation_matrix = np.concatenate([transformation_matrix, margin], axis=0)

    return transformation_matrix
    
def dbscan_clustering(point_cloud_data, epsilon, min_samples):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    labels = np.array(
        pcd.cluster_dbscan(eps=epsilon, min_points=min_samples, print_progress=True))
    
    return labels

def find_intersection(line_vectors, plane_normal, plane_origin):
    inter_pts = []

    for line_vector in line_vectors:
        line_direction = line_vector / norm(line_vector)
        # line_direction = line_vector
        plane_normal = plane_normal / norm(plane_normal)
        
        t = -np.dot(plane_normal, (line_vector - plane_origin)) / np.dot(plane_normal, line_direction)
        
        intersection_point = line_vector + (t)* line_direction

        inter_pts.append(intersection_point)


        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(line_vectors)
        ##
        start_point = np.array([0, 0, 0])
        line_points = np.vstack([start_point, line_vector])

        lines = [[0, 1]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        ##
        start_point = np.array([0, 0, 0])
        line_points = np.vstack([start_point, intersection_point])

        lines = [[0, 1]]

        line_set2 = o3d.geometry.LineSet()
        line_set2.points = o3d.utility.Vector3dVector(line_points)
        line_set2.lines = o3d.utility.Vector2iVector(lines)

        one_point_cloud = o3d.geometry.PointCloud()
        one_point_cloud.points = o3d.utility.Vector3dVector(np.array([intersection_point]))

        o3d.visualization.draw_geometries([point_cloud, line_set, line_set2, one_point_cloud])



    
    return np.array(inter_pts)

def find_intersection_between_line_and_plane(start_point, end_points, plane_origin, plane_normal):
    temp = []
    for end_point in end_points:
        start_point = np.array(start_point)
        end_point = np.array(end_point)[:3]

        # 평면의 원점과 법선 벡터
        plane_origin = np.array(plane_origin)
        plane_normal = np.array(plane_normal)

        # 직선의 방향 벡터
        line_direction = end_point - start_point

        # 직선의 방향 벡터와 평면의 법선 벡터가 수직이 아니면 교점이 없음
        if np.dot(line_direction, plane_normal) == 0:
            return None

        # 직선과 평면의 교점 계산
        t = np.dot(plane_normal, (plane_origin - start_point)) / np.dot(plane_normal, line_direction)
        intersection_point = start_point + t * line_direction
        intersection_point.tolist()
        temp.append(intersection_point)

    return np.array(temp)

def circle_model_known_radius(params, points, known_radius):
    # 주어진 반지름을 가진 원 모델에 대한 residual 함수
    cx, cy = params
    return np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - known_radius

def ransac_circle_fit_known_radius(points, known_radius, threshold=0.02, max_iterations=200):
    best_inliers = None
    best_params = None

    for _ in range(max_iterations):
        # RANSAC 단계: 무작위로 세 점 선택
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]

        # least_squares를 사용하여 모델 파라미터 추정
        initial_params = [np.mean(sample_points[:, 0]), np.mean(sample_points[:, 1])]
        
        result = least_squares(circle_model_known_radius, initial_params, args=(sample_points, known_radius), method='lm')

        # 임계값 내의 인라이어 확인
        inliers = np.abs(circle_model_known_radius(result.x, points, known_radius)) < threshold

        # 현재 모델의 인라이어 수가 최고 인라이어 수보다 크면 갱신
        if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_params = result.x

    # 최적의 원 모델 파라미터와 인라이어 포인트 반환
    return np.concatenate((best_params, [known_radius])), points[best_inliers], best_inliers

def calc_R(xc, yc):
    """ calculate the distance of each data points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f_2b(c):
    """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

def Df_2b(c):

    """ Jacobian of f_2b
    The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
    xc, yc     = c
    df2b_dc    = np.empty((len(c), x.size))

    Ri = calc_R(xc, yc)
    df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
    df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
    df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

    return df2b_dc

def calculate_direction(o1, a1):
    # o1에서 a1으로 가는 방향 벡터 계산
    direction = (a1[0] - o1[0], a1[1] - o1[1])
    
    # 방향 벡터의 크기 계산
    magnitude = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
    
    # 정규화된 방향 벡터 반환
    normalized_direction = (direction[0] / magnitude, direction[1] / magnitude)
    
    return normalized_direction

def move_point(origin_pt, target_pt, distance = 0.1, radius=None):
    direction = calculate_direction(origin_pt, target_pt)
    # distance = np.linalg.norm(origin_pt - target_pt) - radius
    moved_target_pt = (target_pt[0] - direction[0] * distance, target_pt[1] - direction[1] * distance)
    
    return np.array(moved_target_pt)   

def cartesian_to_polar_for_circle(edge_points, center, min_idx):
    relative_points = edge_points - center
    
    theta = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    theta = np.rad2deg(theta).astype(np.int16)
    theta[theta < 0] += 360
    min_polar_theta = theta[min_idx]
    new_theta = []
    for t in theta:
        if t > min_polar_theta:
            tt = t - min_polar_theta
            new_theta.append(tt)
        else : 
            tt = (360 - min_polar_theta) + t
            
            new_theta.append(tt) 
    new_theta = np.array(new_theta)
        
    return new_theta

def draw_circle_arc(center, point, num_points):
    # 중심과 임의의 점을 이용하여 직선의 기울기를 계산
    slope = (point[1] - center[1]) / (point[0] - center[0])

    # 직선의 각도를 라디안으로 변환
    angle_radians = np.arctan(slope) 

    # 호를 그리기 위한 각도 범위 설정
    theta = np.linspace(angle_radians - np.radians(13), angle_radians + np.radians(0), num_points)
    theta = np.flip(theta)
  
    # 호의 좌표 계산
    arc_x = center[0] + np.cos(theta) * np.linalg.norm(np.array(point) - np.array(center))
    arc_y = center[1] + np.sin(theta) * np.linalg.norm(np.array(point) - np.array(center))

    pts_0 = np.column_stack((arc_x, arc_y))

    ####################
    angle_radians = np.arctan(slope) + np.deg2rad(180)

    # 호를 그리기 위한 각도 범위 설정
    theta = np.linspace(angle_radians - np.radians(13), angle_radians + np.radians(0), num_points)
    theta = np.flip(theta) 
  
    # 호의 좌표 계산
    arc_x = center[0] + np.cos(theta) * np.linalg.norm(np.array(point) - np.array(center))
    arc_y = center[1] + np.sin(theta) * np.linalg.norm(np.array(point) - np.array(center))

    pts_180 = np.column_stack((arc_x, arc_y))

    d1 = np.linalg.norm(pts_0[0] - point)
    d2 = np.linalg.norm(pts_180[0] - point)
    if d1 < d2 : 
        return pts_0
    else :
        return pts_180
    
def cartesian_to_polar_for_circle_inner(edge_points, center, min_idx, max_angle_idx):
    relative_points = edge_points - center
    
    theta = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    theta = np.rad2deg(theta).astype(np.int16)
    theta[theta < 0] += 360
    min_polar_theta = theta[min_idx]
    

    new_theta = []
    for t in theta:
        if t > min_polar_theta:
            tt = t - min_polar_theta
            new_theta.append(tt)
        else : 
            tt = (360 - min_polar_theta) + t
            
            new_theta.append(tt) 

    new_theta = np.array(new_theta)
    max_polar_theta = new_theta[max_angle_idx]

    return new_theta, max_polar_theta

def find_circle_intersection(x0, y0, x1, y1, r):
    if np.sqrt(np.square(x0-x1) + np.square(y0-y1)) >= r*2:
        return 0
    r0 = r
    r1 = r
    
    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return 0
    # One circle within other
    if d < abs(r0-r1):
        return 0
    # coincident circles
    if d == 0 and r0 == r1:
        return 0
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return np.array([[x3, y3],[x4, y4]])

def same_two_side(base_vertex1, base_vertex2, r, c_pt):
    x1, y1 = base_vertex1
    x2, y2 = base_vertex2
    
    pts = find_circle_intersection(x1, y1, x2, y2, r)
    
    d1 = np.linalg.norm(c_pt - pts[0])
    d2 = np.linalg.norm(c_pt - pts[1])

    if d1 < d2 :
        return pts[0]
    else : 
        return pts[1]

def generate_combinations(n, k):
    all_numbers = np.arange(n)
    all_combinations = np.array(list(combinations(all_numbers, k)))
    all_combinations =  np.random.permutation(all_combinations)
    return all_combinations

def calculate_triangle_angles(point1, point2, point3):
    # 세 점 사이의 벡터 계산
    vector1 = point2 - point1
    vector2 = point3 - point1
    vector3 = point3 - point2

    # 각 벡터의 길이 계산
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    length3 = np.linalg.norm(vector3)

    # 코사인 법칙을 사용하여 각을 계산
    angle1 = np.degrees(np.arccos(np.clip(np.dot(vector1, vector2) / (length1 * length2), -1.0, 1.0)))
    angle2 = np.degrees(np.arccos(np.clip(np.dot(-vector1, vector3) / (length1 * length3), -1.0, 1.0)))
    angle3 = 180.0 - angle1 - angle2  # 각의 합은 180도입니다.

    return angle1, angle2, angle3

def calculate_direction(o1, a1):
    # o1에서 a1으로 가는 방향 벡터 계산
    direction = (a1[0] - o1[0], a1[1] - o1[1])
    
    # 방향 벡터의 크기 계산
    magnitude = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
    
    # 정규화된 방향 벡터 반환
    normalized_direction = (direction[0] / magnitude, direction[1] / magnitude)
    
    return normalized_direction

def make_circle(point1, point2, point3):
    # 각 좌표를 (x, y) 형태로 분리
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # 중심 좌표 계산
    A = x2 - x1
    B = y2 - y1
    C = x3 - x1
    D = y3 - y1

    E = A * (x1 + x2) + B * (y1 + y2)
    F = C * (x1 + x3) + D * (y1 + y3)

    G = 2 * (A * (y3 - y2) - B * (x3 - x2))

    if G == 0:
        # 세 점이 일직선에 위치하는 경우
        return None

    center_x = (D * E - B * F) / G
    center_y = (A * F - C * E) / G

    # 반지름 계산
    radius = math.sqrt((center_x - x1)**2 + (center_y - y1)**2)

    return (center_x, center_y, round(radius, 8))


def objective_function(X):
    x1_, y1_ ,x2_, y2_, x3_, y3_, x4_, y4_ = X
    P1, P2, P3, P4 = np.array([x1_, y1_]), np.array([x2_, y2_]), np.array([x3_, y3_]), np.array([x4_, y4_])

    sigma_inv_1, det_sigma_1 = np.linalg.inv(Sigma_1), np.linalg.det(Sigma_1)
    term_P1_1 = -0.5 * np.dot((P1 - mu1).T, np.dot(sigma_inv_1, (P1 - mu1)))
    term_P1_2 = np.log(det_sigma_1)
    term_P1 = term_P1_1 + term_P1_2

    sigma_inv_2, det_sigma_2 = np.linalg.inv(Sigma_2), np.linalg.det(Sigma_2)
    term_P2_1 = -0.5 * np.dot((P2 - mu2).T, np.dot(sigma_inv_2, (P2 - mu2)))
    term_P2_2 = np.log(det_sigma_2)
    term_P2 = term_P2_1 + term_P2_2

    sigma_inv_3, det_sigma_3 = np.linalg.inv(Sigma_3), np.linalg.det(Sigma_3)
    term_P3_1 = -0.5 * np.dot((P3 - mu3).T, np.dot(sigma_inv_3, (P3 - mu3)))
    term_P3_2 = np.log(det_sigma_3)
    term_P3 = term_P3_1 + term_P3_2

    sigma_inv_4, det_sigma_4 = np.linalg.inv(Sigma_4), np.linalg.det(Sigma_4)
    term_P4_1 = -0.5 * np.dot((P4 - mu4).T, np.dot(sigma_inv_4, (P4 - mu4)))
    term_P4_2 = np.log(det_sigma_4)
    term_P4 = term_P4_1 + term_P4_2

    res = -(term_P1 + term_P2 + term_P3 + term_P4)
   
    return res


def rec_constraints(params):
    x1, y1, x2, y2, x3, y3, x4, y4 = params
    
    p1, p2, p3, p4 = np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]), np.array([x4, y4])

    res1 = np.abs((0.4*scale)- np.linalg.norm(p1 - p2)) + np.abs((0.4*scale) - np.linalg.norm(p3 - p4)) + np.abs((0.5*scale) - np.linalg.norm(p2 - p3)) + np.abs((0.5*scale) - np.linalg.norm(p4 - p1)) 

    ox, oy = (x1 + x2 + x3 + x4)/4, (y1 + y2 + y3 + y4)/4 
    
    po = np.array([ox, oy])
    d_dst = np.sqrt((0.4*scale)**2 + (0.5*scale) **2)/2
    OP1, OP2, OP3, OP4 = np.linalg.norm(p1 - po), np.linalg.norm(p2 - po),  np.linalg.norm(p3 - po), np.linalg.norm(p4 - po)

    res2 = np.abs(d_dst - OP1) + np.abs(d_dst - OP2) +  np.abs(d_dst - OP3) +  np.abs(d_dst - OP4)  

    return res1 + res2


def calculate_angles(p1, p2, p3, p4):
    # 각 꼭짓점 사이의 벡터 계산
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    v3 = (p4[0] - p3[0], p4[1] - p3[1])
    v4 = (p1[0] - p4[0], p1[1] - p4[1])

    # 벡터의 내적 계산
    dot_product = lambda v1, v2: v1[0] * v2[0] + v1[1] * v2[1]

    # 각도 계산 함수
    def calculate_angle(v1, v2):
        dot = dot_product(v1, v2)
        norm_v1 = math.sqrt(dot_product(v1, v1))
        norm_v2 = math.sqrt(dot_product(v2, v2))
        cos_theta = dot / (norm_v1 * norm_v2)

        # 아크코사인을 이용하여 각을 계산하고, 라디안에서 도(degree)로 변환
        angle = math.degrees(math.acos(cos_theta))
        return angle

    # 사각형의 각을 계산
    angle1 = calculate_angle(v1, v2)
    angle2 = calculate_angle(v2, v3)
    angle3 = calculate_angle(v3, v4)
    angle4 = calculate_angle(v4, v1)

    return angle1, angle2, angle3, angle4


def sort_centers(pts):
    y_idxs = np.argsort(pts[:, 1]) # y 
    temp_pts = pts[y_idxs[::-1]]

    x_12_idxs = np.argsort(temp_pts[:2, 0])
    x_12_pts = temp_pts[x_12_idxs[::-1]]

    x_34_idxs = np.argsort(temp_pts[2:, 0])
    x_34_pts = temp_pts[2:, :][x_34_idxs]

    sorted_pts = np.concatenate([x_12_pts, x_34_pts], axis=0)
    sorted_idx = find_matching_rows(pts, sorted_pts)
    return sorted_idx

def find_matching_rows(a, b):
    matching_row_indices = []
    for row in b:
        # 현재 행을 복제하여 a와 비교하기 위해 행렬을 확장합니다.
        row_extended = np.tile(row, (a.shape[0], 1))
        # a와 row_extended를 비교하여 일치하는 행의 인덱스를 찾습니다.
        matching_row_index = np.where((a == row_extended).all(axis=1))[0]
        matching_row_indices.append(matching_row_index[0])
    return matching_row_indices

################################################################################

laser_ch = 32
radius = 0.12


for posess in range(5, 21):
    stage1_4_2_errors_100_covariance = []
    blank_data = np.array([[0, 0, 0],
                           [0, 0, 0], 
                           [0, 0, 0],
                           [0, 0, 0]], dtype=np.float64)


    for iter in range(1, 31):
        ## Note: This code processes the collected point cloud data. Please split each LiDAR scan point cloud. 
        files = f'Your LiDAR scan path' ## scan path 
    
   
        full_pts = np.load(files)
        circle_edge_np = data_roi_setting(full_pts)
        edges_etc, edges = edges_detector_laser_num(circle_edge_np) 
        
        circle_laser = filter_circle_edges(edges)
        board_pts, rot, normal_vec = ransac_pca_rot(edges_etc, circle_laser) # 3D point cloud to 2D plane
        circle_edge = (rot @ circle_laser[:, :3].T).T 
        board_pts_rotated = (rot @ board_pts[:, :3].T).T

        plane_normal = normal_vec
        plane_origin = np.array([np.mean(board_pts, axis=0)]).squeeze()

        ## Ray Projection (RP)
        circles_refine = find_intersection_between_line_and_plane(np.array([0, 0, 0,]), circle_laser[:, :3], plane_origin, plane_normal)
        circles_refine_rotated = (rot @ circles_refine.T).T

        four_circles = []
        stage11_centers = [] 
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        scale = 1
        z_pts = []

        ## Circle Edge Points Clustering 
        for i in range(4):
            radius *= scale
            circles_refine_rotated *= scale
            circle_params_known_radius, inliers, in_idx = ransac_circle_fit_known_radius(circles_refine_rotated[:, :2], radius)
            four_circles.append(inliers[:, :2])
            z_pts.append(circles_refine_rotated[in_idx][:, 2])

            circles_refine_rotated = np.delete(circles_refine_rotated, in_idx, axis = 0)


        iter_flag = True
        adjust_edges = []
        ## Circle Edge Points Adjustment
        for edges in four_circles:
            flag = True 

            centerX, centerY = np.mean(edges[:, 0]), np.mean(edges[:, 1])
            center_estimate = centerX, centerY
            x, y = edges[:, 0], edges[:, 1]
            center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)
            init_center = center_2b

            pseudo_centers = []
            min_checks = []
            min_idxs = []
            last_min = None

            adjust_edge = np.empty((0, 2))
            for fire_circle_iter in range(10): 
                if fire_circle_iter == 0: 
                    p_ang = 180
                    dist = cdist(edges, np.array([init_center]))
                    dist = dist.squeeze()
                    min_idx = np.argmin(dist)
                    min_idxs.append(min_idx)
                    moved_step = dist[min_idx] - radius
                    
                    init_center = move_point(edges[min_idx], init_center, moved_step)
                    
                    last_min = edges[min_idx]
                    pseudo_centers.append(init_center)
                    min_checks.append(edges[min_idx])

                    ploar_angs = cartesian_to_polar_for_circle(edges, init_center, min_idx)
                    half_edges = edges[ploar_angs < p_ang]

                else :
                    half_min_idx = -1
                    select_pseudo_idx = -1
                    pseudo_arcs = draw_circle_arc(edges[min_idx], init_center, num_points=8000)
                    
                    last_min_idx = np.where((half_edges == last_min).all(axis=1))[0]

                    if len(last_min_idx) > 0:
                        half_edges = np.delete(half_edges, last_min_idx[0], axis = 0)
                    
                    if len(half_edges) == 0 :
                        flag = False 
                        break
                    for arc_idx, pseudo in enumerate(pseudo_arcs):
                        dist = cdist(half_edges, np.array([pseudo])) 
                        dist = dist.squeeze()
                        
                        if (np.min(dist) < radius):
                            half_min_idx = np.argmin(dist)
                            select_pseudo_idx = arc_idx -1
                            break

                    init_center = pseudo_arcs[arc_idx]
                    min_idx = np.where((edges == half_edges[half_min_idx]).all(axis=1))[0][0]
                    ploar_angs = cartesian_to_polar_for_circle(edges, init_center, min_idx)
                    half_edges = edges[ploar_angs < p_ang]
                    
                    adjust_edge = np.append(adjust_edge, np.array([edges[min_idx]]), axis=0)

                    # Finish criterion 
                    tmp = min_checks.copy()
                    tmp = np.array(tmp)
                    last_min = edges[min_idx]
                    if any(np.all(row == last_min) for row in tmp):
                        break

                    min_idxs.append(min_idx)
                    pseudo_centers.append(init_center)
                    min_checks.append(edges[min_idx])

            if flag == False:
                iter_flag = False 
                break
            for i in range(len(min_idxs)):
                start_i, end_i = i, i+1 
                if end_i == len(min_idxs):
                    end_i = 0

                start_pt, end_pt = min_idxs[start_i], min_idxs[end_i]
            
                start_angs, end_pt_angs = cartesian_to_polar_for_circle_inner(edges, center_2b, start_pt, end_pt)
                inner_pts = edges[start_angs < end_pt_angs]
                if len(inner_pts) == 0:
                    continue
                
                adjust_centre = same_two_side(edges[start_pt], edges[end_pt], radius, center_2b)

                inner_dists = cdist(inner_pts, np.array([adjust_centre])) - radius

                
                for idx, inner_pt in enumerate(inner_pts):
                    moved_pt = move_point(adjust_centre, inner_pt, inner_dists[idx], inner_dists[idx][0])
                    adjust_edge = np.append(adjust_edge, moved_pt.T, axis= 0)
            adjust_edges.append(adjust_edge)

        if iter_flag == False:
            print('Next Iter')
            continue

        cov_2ds = [] 
        mean_2ds = []
        cov_splits = []
        cov_splits_std = []
        scale = 1
  
        ## Circle Center Distribution (CD)
        for id, adjust_edge in enumerate(adjust_edges):
            adjust_edge = np.unique(adjust_edge, axis=0)
            plans = generate_combinations(len(adjust_edge), 3)
            psudo_centers = []

            for plan in plans : 
                i, j ,q = plan 

                x, y, r = make_circle (adjust_edge[i], adjust_edge[j], adjust_edge[q])
                a1, a2, a3 = calculate_triangle_angles(adjust_edge[i], adjust_edge[j], adjust_edge[q])
                angs = np.array([a1, a2, a3])
                if np.max(angs) < 90 :
                    psudo_centers.append(np.array([x, y]))

            psudo_centers = np.array(psudo_centers)
            cov_matrix = np.cov(psudo_centers, rowvar=False)

            mean = np.mean(psudo_centers, axis=0)
            cov_2ds.append(cov_matrix)
            mean_2ds.append(mean)
       
            ##########################
        
        # Optimization with Lagrangian method 
        sorted_cov_2ds = [] 
        sorted_mean_2ds = []
        sorted_cov_splits = []
        mean_3ds = []
        sorted_cov_splits_std = []
        mean_2ds = np.array(mean_2ds)
        tmp_dist = sort_centers(mean_2ds)

        for arg_min in tmp_dist:
            sorted_cov_2ds.append(cov_2ds[arg_min])
            sorted_mean_2ds.append(mean_2ds[arg_min])
            tmp_z = np.mean(z_pts[arg_min])
            mean_3ds.append(np.array([mean_2ds[arg_min][0], mean_2ds[arg_min][1], tmp_z]))

        cov_2ds = sorted_cov_2ds
        mean_2ds = sorted_mean_2ds
        rotated_mean_3ds = (rot.T @ np.array(mean_3ds).T).T

        Sigma_1, Sigma_2, Sigma_3, Sigma_4 = cov_2ds[0], cov_2ds[1], cov_2ds[2], cov_2ds[3]
        mu1, mu2, mu3, mu4 = mean_2ds[0], mean_2ds[1], mean_2ds[2], mean_2ds[3] 
        initial_guess = [mu1[0], mu1[1], mu2[0], mu2[1], mu3[0], mu3[1], mu4[0], mu4[1]]

        constraints = (
            {'type': 'eq', 'fun': rec_constraints},
        )  
        ################### 

        try:
            result = optimize.minimize(objective_function, initial_guess, constraints=constraints, method = 'Trust-constr' , options={'maxiter': 1000})
        except Exception as e : 
            print('Error')
            continue
        
        stage2_split_pred_centers = result.x.reshape(4, 2)
        
        stage2_centers = []
        for i, stage2_split_pred_center in enumerate(stage2_split_pred_centers):
            tmp_pred_center = np.array([stage2_split_pred_center[0], \
                                        stage2_split_pred_center[1], \
                                            mean_3ds[i][2]])
            stage2_centers.append(tmp_pred_center)

        stage2_rotated_mean_3ds = (rot.T @ np.array(stage2_centers).T).T

        ## Accumulate
        blank_data = blank_data + stage2_rotated_mean_3ds
        cum_stage2_centers = blank_data / iter
        stage1_4_2_errors_100_covariance.append(stage2_rotated_mean_3ds)


    stage1_4_2_errors_100_covariance = np.array(stage1_4_2_errors_100_covariance)
    np.save(f'Your save path', stage1_4_2_errors_100_covariance)


