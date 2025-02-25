import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import torch
import stairs
import os
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




class Config:
    fx, fy, cx, cy = 605.9815, 606.1337, 308.5001, 246.4238
    intrinsic_matrix = intrinsic_matrix = np.array([
    [fx,  0, cx],  # 초점 거리 fx, 주점(cx)
    [ 0, fy, cy],  # 초점 거리 fy, 주점(cy)
    [ 0,  0,  1]   # 변환을 위한 마지막 행 (고정)
    
])
    intrinsics = 605.9815, 606.1337, 308.5001, 246.4238
    depth_scale = 1000
    k = 10
    threshold = 0.9
    voxel_size = 50 #mm단위이므로 5cm임

class Point:
    def __init__(self, position, normal=None, isGround=False):
        self.position = position # 3d 좌표
        self.normal = normal # 법벡터
        self.isGround = isGround 

class VoxelGrid:
    def __init__(self, points, voxel_size):
        self.voxel_size = voxel_size
        self.points = points # 각 점의 x, y ,z 좌표를 저장함. points[i] = [x, y ,z]
        self.voxel_grid = {}
        for i, pt in enumerate(points): #  포인트들을 voxelgrid에 할당함
            voxel_key = tuple(np.floor(pt.position / voxel_size).astype(int))
            if voxel_key not in self.voxel_grid:
                self.voxel_grid[voxel_key] = []
            self.voxel_grid[voxel_key].append(i)
        
        # Building a k-d tree for fast k-NN search
        self.tree = NearestNeighbors(n_neighbors=Config.k, algorithm='kd_tree')
        point_positions = np.array([pt.position for pt in points])  # assuming pt.position is a 3D vector
        self.tree.fit(point_positions)

    def getKNN(self, position, k, search_radius, points):
        # Use the k-d tree to find k nearest neighbors within the search radius
        distances, indices = self.tree.radius_neighbors([position], radius=search_radius)
        
        # Filter out the neighbors that are beyond the search radius
        neighbors = [idx for idx in indices[0] if np.linalg.norm(points[idx].position - position) <= search_radius]
        
        return neighbors[:k]  # return the first k neighbors

def detect_horizontal_edges_and_save(image_path, output_path):
    # RGB 이미지 불러오기
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Failed to load image at {image_path}")
        return
    

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 엣지 검출: Canny Edge Detection 사용
    edges = cv2.Canny(gray, 100, 200)

    # 수평 엣지 찾기 위한 커널 정의
    kernel = np.ones((1, 10), np.uint8)  # 수평 방향으로 스트로크를 넓힘
    horizontal_edges = cv2.dilate(edges, kernel, iterations=1)

    # 수평 엣지에 선 그리기
    lines = cv2.HoughLinesP(horizontal_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 선 그리기

    # 이미지 저장
    cv2.imwrite(output_path, img)
    print(f"Image saved at {output_path}")



def preprocessPointCloud(points, voxel_size=Config.voxel_size, k=Config.k, threshold=Config.threshold):
    grid = VoxelGrid(points, voxel_size)
    search_radius = voxel_size * 1.5

    # Using the bottom 80% of the point cloud
    min_y = np.min([pt.position[1] for pt in points])
    max_y = np.max([pt.position[1] for pt in points])
    bottom_80_percent = min_y + (max_y - min_y) * 0.2  # 20% 지점부터 하단 80%

    points_bottom_80 = [pt for pt in points if pt.position[1] <= bottom_80_percent]

    for pt in points_bottom_80:
        neighbors = grid.getKNN(pt.position, k, search_radius, points)
        
        if len(neighbors) < k:
            pt.isGround = False
            continue
        
        # Compute covariance matrix
        mean = np.mean([points[idx].position for idx in neighbors], axis=0)
        covariance = np.cov([points[idx].position - mean for idx in neighbors], rowvar=False)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(covariance)
        normal = eigvecs[:, np.argmin(eigvals)]  # Smallest eigenvalue corresponds to normal

        if normal[2] < 0:
            normal = -normal
        pt.normal = normal
        pt.isGround = np.dot(normal, np.array([0, 0, 1])) > threshold

    # Remove ground points
    points_without_ground = [pt for pt in points if not pt.isGround] # point 객체
    ground_points = [pt for pt in points if pt.isGround]
    return points_without_ground, ground_points

def preprocess_RGBimg(rgb_image, points, ground_points, intrinsics = Config.intrinsics):
    """
    RGB 이미지에서 그라운드 플레인에 해당하는 포인트를 날리는 함수.
    
    :param rgb_image: 원본 RGB 이미지
    :param points: 전체 포인트 클라우드
    :param ground_points: 그라운드로 판단된 포인트들의 인덱스
    :param intrinsics: 카메라 내재 파라미터 (fx, fy, cx, cy)
    :return: 그라운드 플레인이 제거된 RGB 이미지
    """
    # 만약 points가 리스트라면 numpy 배열로 변환
    points = np.array([pt.position for pt in points]) if isinstance(points, list) else points

    # 3D 포인트를 2D 이미지로 투영
    ground_3d_points = points[ground_points]  # ground_points는 인덱스 리스트여야 함
    ground_2d_points = project_to_2d(ground_3d_points, intrinsics)

    # 2D 좌표를 이미지 크기 내에서 유효한 값으로 클리핑
    h, w, _ = rgb_image.shape
    u, v = ground_2d_points[:, 0], ground_2d_points[:, 1]
    u = np.clip(u.astype(int), 0, w - 1)
    v = np.clip(v.astype(int), 0, h - 1)

    # 그라운드 포인트에 해당하는 픽셀을 마스크로 설정
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[v, u] = 1

    # 그라운드 영역을 제거하거나 대체 (예: 0으로 마스킹)
    rgb_image[mask == 1] = 0  # 그라운드 부분을 검정색으로 대체

    return rgb_image



def depth_to_pointcloud(depth_map, intrinsic_matrix = Config.intrinsic_matrix, depth_scale=Config.depth_scale):

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    height, width = depth_map.shape
    points = []
    
    # 각 픽셀에 대해 3D 좌표 계산
    for v in range(height):
        for u in range(width):
            depth_val = depth_map[v, u] / depth_scale  # mm를 meter로 변환
            
            if depth_val == 0:  # 깊이 값이 0인 경우는 건너뛰기
                continue
            
            # 2D 좌표 (u, v)를 3D 좌표 (X, Y, Z)로 변환
            X = (u - cx) * depth_val / fx
            Y = (v - cy) * depth_val / fy
            Z = depth_val
            
            # Point 객체 생성
            pt = Point(position=np.array([X, Y, Z]))
            points.append(pt)
            print(f"Pixel ({u}, {v}): Depth: {depth_val}, X: {X}, Y: {Y}, Z: {Z}")

    
    return points


def project_to_2d(points, depth_scale = Config.depth_scale,intrinsics=Config.intrinsics): 
    """
    3D 포인트를 2D RGB 이미지로 투영하는 함수.
    
    :param points: 3D 포인트 (N x 3) numpy 배열
    :param intrinsics: 카메라의 내재 파라미터 (fx, fy, cx, cy)
    :return: 2D 이미지 좌표 (u, v)
    """
    fx, fy, cx, cy = intrinsics
    u = (points[:, 0] * fx / points[:, 2]) + cx
    v = (points[:, 1] * fy / points[:, 2]) + cy
    return np.column_stack((u, v))


def pointcloud_visualization(points, filename="pointcloud.png"):
    # 포인트들의 x, y, z 좌표 추출
    x_coords = [pt.position[0] for pt in points]
    y_coords = [pt.position[1] for pt in points]
    z_coords = [pt.position[2] for pt in points]

    # 3D 플롯 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 포인트들을 점으로 플로팅
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o', s=1)  # 파란색 점, 크기 1

    # 축 라벨
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 그래프 제목
    ax.set_title('Point Cloud Visualization')

    # 그래프 이미지 저장 (PNG 또는 JPG)
    plt.savefig(filename, dpi=300)
    plt.close()  # 그래프 닫기



#----------------- 데이터 로드 함수
def load_rgb_from_bin(bin_path, frame_idx, height=480, width=640):
    data_path = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(data_path, "data", "meta.txt")

    # 🔹 1) meta.txt에서 프레임 개수 읽기
    try:
        with open(meta_path, "r") as f:
            total_frames = int(f.readline().strip())  # 첫 번째 줄에 저장된 프레임 개수 읽기
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 메타데이터 파일 {meta_path}을 찾을 수 없습니다!")
    except ValueError:
        raise ValueError(f"❌ {meta_path}에서 프레임 개수를 읽을 수 없습니다!")

    # 🔹 2) frame_idx가 유효한지 확인
    if frame_idx >= total_frames or frame_idx < 0:
        raise ValueError(f"⚠️ frame_idx {frame_idx}가 유효하지 않습니다! (총 {total_frames}개 프레임)")

    # 🔹 3) .bin 파일에서 RGB 데이터 로드
    try:
        rgb_data = np.fromfile(bin_path, dtype=np.uint8)

        # 전체 데이터가 (total_frames, H, W, 3) 크기인지 확인
        expected_size = total_frames * height * width * 3
        if len(rgb_data) != expected_size:
            raise ValueError(f"❌ RGB 데이터 크기 불일치! 예상 {expected_size}, 실제 {len(rgb_data)}")

        # 🔹 4) (프레임 개수, H, W, 3) 형태로 reshape
        rgb_data = rgb_data.reshape((total_frames, height, width, 3))

        # 🔹 5) frame_idx에 해당하는 프레임 반환
        rgb_image = rgb_data[frame_idx]

    except Exception as e:
        raise RuntimeError(f"❌ RGB .bin 파일을 로드하는 중 오류 발생: {e}")

    return rgb_image



def align_depth_to_rgb(depth_bin_path, rgb_bin_path, frame_idx, height=480, width=640):
    context = rs.context()
    devices = context.query_devices()
    data_path = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(data_path, "data", "meta.txt")
    try:
        with open(meta_path, "r") as f:
            total_frames = int(f.readline().strip())  # 첫 줄에서 프레임 개수 읽기
        print(f"🔹 meta.txt에서 읽은 프레임 개수: {total_frames}")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 메타데이터 파일 {meta_path}을 찾을 수 없습니다!")
    except ValueError:
        raise ValueError(f"❌ {meta_path}에서 프레임 개수를 읽을 수 없습니다!")

    if len(devices) == 0:
        print("🔹 No device connected, using default intrinsics & loading from .bin files")
        intrinsics = rs.intrinsics()
        intrinsics.width = width
        intrinsics.height = height
        intrinsics.ppx = 308.5001  # 기본 광학 중심 X (cx)
        intrinsics.ppy = 246.4238  # 기본 광학 중심 Y (cy)
        intrinsics.fx = 605.9815  # 기본 초점 거리 X (fx)
        intrinsics.fy = 606.1337  # 기본 초점 거리 Y (fy)
        intrinsics.model = rs.distortion.none  # 왜곡 없음
        intrinsics.coeffs = [0, 0, 0, 0, 0]  
        
        # --- .bin 파일에서 RGB & Depth 불러오기 ---
        depth_map = np.fromfile(depth_bin_path, dtype=np.float32)
        depth_map = depth_map.reshape((total_frames, height, width))
        depth_map = depth_map[frame_idx]

        rgb_image = load_rgb_from_bin(rgb_bin_path, frame_idx)

        if frame_idx >= total_frames:
            raise ValueError(f"⚠️ frame_idx {frame_idx}가 저장된 프레임 개수 {total_frames}보다 큼")

    else:
        try:
            print("✅ Realsense device detected, capturing frames...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipeline.start(config)

            # 카메라 Intrinsics 가져오기
            color_profile = profile.get_stream(rs.stream.color)
            intr = color_profile.as_video_stream_profile().get_intrinsics()
            fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

            # Depth → RGB 정렬 수행
            align_to = rs.stream.color
            align = rs.align(align_to)

            # 프레임 수집 및 정렬
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise RuntimeError("⚠️ Failed to capture frames from Realsense.")

            # numpy 배열 변환
            depth_map = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            rgb_image = np.asanyarray(color_frame.get_data())

            pipeline.stop()

        except RuntimeError:
            print("No device connected (error during capture), using default intrinsics")
            profile = None
            depth_map = np.zeros((height, width), dtype=np.float32)  # 빈 Depth 맵 생성
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)  # 빈 RGB 이미지 생성

    return depth_map, rgb_image

#--------중복박스 있는지 체크 
def check_duplicate(results):
    seen = set()
    duplicates = False
    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in seen:
                    duplicates = True
                else:
                    seen.add(cls_id)
    return duplicates # 불리안 아웃풋임

#------- 중복박스 없애고 가까운것만 반환
def remove_extra_box(results, depth_map):
    all_boxes = []  # 모든 바운딩 박스를 저장할 리스트

    # YOLO 탐지 결과에서 모든 바운딩 박스 수집 (클래스 ID 포함)
    for result in results:
        for box in result.boxes:
            bbox = tuple(map(int, box.xyxy[0]))  # 바운딩 박스 좌표 (x1, y1, x2, y2)
            cls_id = int(box.cls[0])  # 클래스 ID 가져오기
            all_boxes.append((cls_id, bbox))  # (클래스 ID, 바운딩 박스) 저장

    # 탐지된 객체가 없으면 (None, None) 반환
    if not all_boxes:
        return (None, None)

    # 가장 가까운 객체 찾기
    return get_closest_box_with_depth(all_boxes, depth_map)


def get_closest_box_with_depth(boxes, depth_map):
    """ 가장 가까운 바운딩 박스를 선택 (최소 Depth 값 기준) """
    min_depth = float("inf")
    closest_box = None
    closest_cls_id = None

    for cls_id, bbox in boxes:
        x1, y1, x2, y2 = bbox
        roi_depth = depth_map[y1:y2, x1:x2]  

        # 0이 아닌 Depth 값이 있으면 최솟값 계산 
        valid_depths = roi_depth[roi_depth > 0]
        if len(valid_depths) > 0:
            min_roi_depth = np.min(valid_depths)  
            if min_roi_depth < min_depth:
                min_depth = min_roi_depth
                closest_box = bbox
                closest_cls_id = cls_id  # 가장 가까운 박스의 클래스 ID 저장

    return (closest_cls_id, closest_box) 







def extract_plane_ransac(points, threshold=0.01, normal_threshold=0.95):
    """
    Depth 이미지에서 여러 평면을 추출하고, 각 평면의 최소 Depth 값을 계산하여
    가장 가까운 평면을 선택합니다.
    :param depth_map: (H, W) 형태의 Depth 이미지
    :param intrinsic_matrix: 카메라 내적 행렬 (fx, fy, cx, cy 포함)
    :param threshold: RANSAC에서 평면과의 거리 기준
    :param normal_threshold: 노말벡터랑 내적햇을때
    :return: 가장 가까운 평면에 해당하는 포인트들 (inliers)
    """
    # points를 pcd 객체로 변환
    pcd = o3d.geometry.PointCloud()
    xyz = np.array([pt.position for pt in points])
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # RANSAC을 이용해 여러 평면 모델 추출
    planes = []
    for _ in range(10):  # 평면을 n개 추출

        #segment_plane: plane_model: ax + by + cz + d = 0에서 리스트 [a, b, c, d] 반환
        #inliers = [3, 7, 12, 25, 48, 102, ...] 같은 인덱스
        plane_model, inliers = pcd.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=1000)
        inlier_cloud = pcd.select_by_index(inliers)

        #(0, 1, 0)이랑 내적
        normal_vector = np.array(plane_model[:3])
        dot_product = np.dot(normal_vector, np.array([0, 1, 0]))  # (0, 1, 0) 벡터와의 내적

        # 내적값이 임계값 이상이면 추가
        if dot_product > normal_threshold:
            planes.append((plane_model, inlier_cloud))

        # 추출된 평면을 포인트클라우드에서 제외시켜 다음 평면을 찾기 위해
        pcd = pcd.select_by_index(inliers, invert=True)  
    
    # 각 평면의 Depth 계산 (평면에 포함된 점들의 최소 Depth 값)
    min_depth = float('inf')
    closest_plane = None
    closest_normal = None
    closest_inliers = None
    
    for plane_model, inlier_cloud in planes:
        # 평면에 포함된 점들의 깊이 값 계산
        inlier_points = np.asarray(inlier_cloud.points)
        min_plane_depth = np.min(inlier_points[:, 2])  # Z 값이 Depth에 해당
        
        # 가장 작은 Depth 값을 가진 평면을 선택
        if min_plane_depth < min_depth:
            min_depth = min_plane_depth
            closest_plane = inlier_cloud
            closest_normal = np.array(plane_model[:3])
            closest_inliers = inlier_points

    
    return closest_plane, closest_normal, closest_inliers







#-----------바운딩박스의 ROI 크롭하기
def crop_roi(bbox, rgb_image, depth_map):
    x1,y1,x2,y2 = bbox
    rgb_roi = rgb_image[y1:y2, x1:x2, :]
    depth_roi = depth_map[y1:y2, x1:x2]
    return rgb_roi, depth_roi


#--------- 클래스 분류해서 함수 실행 ❗❗❗❗❗❗❗❗❗❗수정필요
def measure_height(cls_id,rgb_roi, depth_roi, model):
    if cls_id ==0:
        angle, height = stairs.measure_height(depth_roi)
    #❗디버깅용
    print("높이 측정중")
    return angle, height









# ----------------------
# 실제 사용 예시
# ----------------------
if __name__ == "__main__":
    # 가정: color_img (BGR)와 YOLO로부터 얻은 bbox, 세그멘테이션 모델 준비
    color_img = cv2.imread("test.jpg")
    yolo_bbox = (100, 200, 400, 500)  # 예시 (x1, y1, x2, y2)

    # 예: 임의의 세그멘테이션 모델 (PyTorch)
    # model = MySegModel(...)
    # model.load_state_dict(torch.load("model.pth"))
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)

    # 여기서는 데모라서 model 대신 None 처리
    model = None
    device = 'cpu'

    # 세그멘테이션 함수 시연 (실제로는 model이 필요)
    # segment_stairs_in_roi 함수 내 model 부분을 직접 수정해서 사용 가능
    # 혹은 아래처럼 "더미"로 예시를 만들 수도 있음
    def dummy_model(x):
        # 입력 x: (1,3,h,w)
        # 가짜로 전부 '1' 클래스라고 치자 (전부 계단)
        return torch.zeros((1, 2, x.shape[2], x.shape[3]), device=x.device) + 0.5

    seg_model = dummy_model

    mask_roi = stairs.segment_stairs_in_roi(color_img, yolo_bbox, seg_model, device=device)
    
    # 후속처리
    edges, lines_p = stairs.postprocess_stair_mask(mask_roi)

    # 시각화 예시
    # ROI 영역만 시각화
    x1, y1, x2, y2 = yolo_bbox
    roi_vis = color_img[y1:y2, x1:x2].copy()

    # 에지 그리기
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_bgr[edges != 0] = (0, 0, 255)  # 빨간색

    # lines_p가 있으면 직선 시각화
    if lines_p is not None:
        for line in lines_p:
            x_start, y_start, x_end, y_end = line[0]
            cv2.line(roi_vis, (x_start, y_start), (x_end, y_end), (0,255,0), 2)

    # 결과 보기
    cv2.imshow("ROI mask", mask_roi*255)  # 0 or 1 => 시각화를 위해 255 곱
    cv2.imshow("ROI edges", edges_bgr)
    cv2.imshow("ROI lines", roi_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
