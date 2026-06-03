import numpy as np
import scipy.io as sio
import pickle


# model.pkl 파일을 한 번만 불러온다.
with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)


def your_algorithm(d_u, p_bs):
    """
    d_u: 한 사용자에 대한 18개 RTT 값, shape = (18,)
    p_bs: 18개 기지국 좌표, shape = (2, 18)

    return: 예측 위치, shape = (2,)
    """

    mean = MODEL["mean"]
    std = MODEL["std"]
    pca_components = MODEL["pca_components"]
    X_pca = MODEL["X_pca"]
    X_norm = MODEL["X_norm"]
    y = MODEL["y"]
    k_neighbors = MODEL["k_neighbors"]
    epsilon = MODEL["epsilon"]

    # 1. 입력 RTT 정규화
    d_norm = (d_u - mean) / std

    # 2. PCA 공간으로 변환
    d_pca = d_norm @ pca_components.T

    # 3. PCA 공간에서 학습 데이터와의 거리 계산
    diff = X_pca - d_pca
    distances = np.sqrt(np.sum(diff * diff, axis=1))

    # 4. 가장 가까운 K개 이웃 선택
    neighbor_idx = np.argsort(distances)[:k_neighbors]
    neighbor_dist = distances[neighbor_idx]
    neighbor_pos = y[neighbor_idx]

    # 5. 원래 RTT 정규화 공간에서 residual 계산
    neighbor_rtt = X_norm[neighbor_idx]
    rtt_residual = np.sqrt(np.sum((neighbor_rtt - d_norm) ** 2, axis=1))

    # 6. PCA 거리와 RTT residual을 함께 반영한 가중치 계산
    weights = 1.0 / ((neighbor_dist + epsilon) * (rtt_residual + epsilon))

    # 7. 선택된 이웃 위치의 가중 평균으로 최종 위치 추정
    pred = np.sum(neighbor_pos * weights[:, None], axis=0) / np.sum(weights)

    return pred


def main():
    # 1. 입력 데이터 로드
    mat_path = "DH_FR1.mat"

    data = sio.loadmat(mat_path, squeeze_me=False)
    p_bs = np.asarray(data["p_bs"], dtype=float)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float)

    # 2. 사용자 수를 입력 데이터에서 동적으로 계산
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))

    # 3. 각 사용자에 대해 위치 추정 수행
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)

    # 4. 결과 반환
    return p_hat


if __name__ == "__main__":
    p_hat = main()

    # 로컬 테스트용 출력
    # 채점 시에는 main()의 return값이 중요하다.
    data = sio.loadmat("DH_FR1.mat", squeeze_me=False)
    p = np.asarray(data["p"], dtype=float)

    error = np.sqrt(np.sum((p_hat - p) ** 2, axis=0))

    print("p_hat shape:", p_hat.shape)
    print("mean error:", np.mean(error))
    print("median error:", np.median(error))
    print("max error:", np.max(error))