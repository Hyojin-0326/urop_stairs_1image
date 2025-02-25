import torch
import cv2
import numpy as np
import utils
import open3d as o3d



#----------- 파이프라인
def measure_height(points):
    
    closest_plane, closest_normal = utils.extract_plane_ransac(points) #closest_plane은 open3d Pointclout 객체임

    print(f"debug: extract_ransac")

    angle = calculate_angle(closest_normal)

    #plane 안에 해당하는 점(inlier 인덱스를 받든지, inlier points를 받든지) 해서 넘파잉 좌표로 바꾸고 calcu;ate_height에 넘겨야함.







    height=calculate_height(closest_plane)


    print(f'{angle} 라디안, {height} 미터')

    return angle


def calculate_angle(normal_vector):
    # 수평면의 법선 벡터 (z 축 방향)
    horizontal_normal = np.array([0, 1, 0])
    
    # 법선 벡터와 수평면 벡터 사이의 내적 계산
    dot_product = np.dot(normal_vector, horizontal_normal)
    
    # 각도 계산 (radian 단위)
    angle_rad = np.arccos(dot_product / (np.linalg.norm(normal_vector) * np.linalg.norm(horizontal_normal)))
    
    return angle_rad

def calculate_height(pcld_np):
    
    pcld_sorted = pcld_np[np.argsort(pcld_np[:, 2])]  # z 값 기준으로 오름차순 정렬
    
    # 윗쪽 엣지 (상위 10%의 포인트들)
    upper_edge = pcld_sorted[int(0.9 * len(pcld_sorted)):]  # 90% 이상은 윗쪽 엣지
    
    # 아랫쪽 엣지 (하위 10%의 포인트들)
    lower_edge = pcld_sorted[:int(0.1 * len(pcld_sorted))]  # 10% 이하를 아랫쪽 엣지
    
    # 윗쪽 엣지와 아랫쪽 엣지의 평균 z 값을 계산
    upper_edge_mean = np.mean(upper_edge[:, 2])
    lower_edge_mean = np.mean(lower_edge[:, 2])
    
    # 높이는 윗쪽 엣지의 평균 높이에서 아랫쪽 엣지의 평균 높이를 빼서 계산
    height = upper_edge_mean - lower_edge_mean
    
    print("debug: calculate_angle")
    
    return height