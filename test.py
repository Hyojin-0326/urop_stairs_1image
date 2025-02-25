import numpy as np
import imageio.v2 as imageio
import pickle

class Config:
    intrinsics = 605.9815, 606.1337, 308.5001, 246.4238  # fx, fy, cx, cy

class Point:
    def __init__(self, position, normal=None, isGround=False):
        self.position = position  # 3D 좌표 (numpy array)
        self.normal = normal  # 법선 벡터 (numpy array, 없으면 None)
        self.isGround = isGround  # 지면 여부 (기본값 False)

def load_depth_map(file_path, depth_scale=0.001):
    """
    16비트 PNG 파일에서 뎁스맵을 읽고, 미터 단위로 변환.
    
    Parameters:
        file_path (str): PNG 뎁스맵 파일 경로.
        depth_scale (float): 깊이 값을 미터 단위로 변환하는 스케일 (기본: 0.001m)
    
    Returns:
        depth_map (numpy.ndarray): (H, W) 크기의 뎁스맵 배열 (미터 단위)
    """
    depth_map = imageio.imread(file_path).astype(np.float32)
    return depth_map * depth_scale  # 미터 단위로 변환

def depth_to_point_objects(depth_map):
    """
    뎁스맵을 사용하여 Point 객체 리스트로 변환.
    
    Parameters:
        depth_map (numpy.ndarray): (H, W) 크기의 뎁스맵 배열 (미터 단위).
        
    Returns:
        List[Point]: 변환된 Point 객체 리스트.
    """
    fx, fy, cx, cy = Config.intrinsics  # 카메라 내부 파라미터 가져오기

    h, w = depth_map.shape
    points = []

    # 각 픽셀의 u, v 좌표 생성
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    # 깊이 값 가져오기
    d = depth_map

    # 공식에 따른 3D 좌표 계산
    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d

    # 각 픽셀을 Point 객체로 변환
    for i in range(h):
        for j in range(w):
            position = np.array([X[i, j], Y[i, j], Z[i, j]])
            point = Point(position)
            points.append(point)
            print('변환중...')

    return points

def save_points(points, file_path):
    """
    Point 객체 리스트를 파일로 저장 (pickle 사용).
    
    Parameters:
        points (List[Point]): 저장할 포인트 리스트.
        file_path (str): 저장할 파일 경로 (.pkl)
    """
    with open(file_path, 'wb') as f:
        pickle.dump(points, f)
    print(f"✅ 포인트 리스트 저장 완료: {file_path}")

def load_points(file_path):
    """
    저장된 Point 객체 리스트를 불러오기 (pickle 사용).
    
    Parameters:
        file_path (str): 불러올 파일 경로 (.pkl)
    
    Returns:
        List[Point]: 불러온 포인트 리스트.
    """
    with open(file_path, 'rb') as f:
        points = pickle.load(f)
    print(f"✅ 포인트 리스트 로드 완료: {file_path}")
    return points

# 사용 예제
if __name__ == "__main__":
    # 16비트 PNG 뎁스맵 로드
    depth_map = load_depth_map("data/depth_data.png")

    # 뎁스맵을 포인트 클라우드로 변환
    points = depth_to_point_objects(depth_map)

    # 포인트 리스트 저장
    save_points(points, "points.pkl")

    # 저장된 포인트 리스트 불러오기
    loaded_points = load_points("points.pkl")

    # 첫 번째 포인트 확인
    print("첫 번째 포인트 좌표:", loaded_points[0].position)