import numpy as np
import scipy.io as sio


def build_model(X_train, y_train, pca_dim):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0

    X_norm = (X_train - mean) / std

    U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)

    pca_components = Vt[:pca_dim, :]
    X_pca = X_norm @ pca_components.T

    model = {
    "mean": mean,
    "std": std,
    "pca_components": pca_components,
    "X_pca": X_pca,
    "X_norm": X_norm,
    "y_train": y_train,
}

    return model


def predict_one(d_u, model, k_neighbors):
    mean = model["mean"]
    std = model["std"]
    pca_components = model["pca_components"]
    X_pca = model["X_pca"]
    y_train = model["y_train"]

    d_norm = (d_u - mean) / std
    d_pca = d_norm @ pca_components.T

    diff = X_pca - d_pca
    distances = np.sqrt(np.sum(diff * diff, axis=1))

    neighbor_idx = np.argsort(distances)[:k_neighbors]
    neighbor_dist = distances[neighbor_idx]
    neighbor_pos = y_train[neighbor_idx]

    epsilon = 1e-6
    weights = 1.0 / (neighbor_dist + epsilon)

    pred = np.sum(neighbor_pos * weights[:, None], axis=0) / np.sum(weights)

    return pred


def evaluate(pca_dim, k_neighbors):
    data = sio.loadmat("DH_FR1.mat", squeeze_me=False)

    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float)

    X = d_hat.T
    y = p.T

    num_data = X.shape[0]

    np.random.seed(0)
    indices = np.random.permutation(num_data)

    train_size = int(num_data * 0.8)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    model = build_model(X_train, y_train, pca_dim)

    y_pred = np.zeros_like(y_val)

    for i in range(X_val.shape[0]):
        y_pred[i] = predict_one(X_val[i], model, k_neighbors)

    error = np.sqrt(np.sum((y_pred - y_val) ** 2, axis=1))

    mean_error = np.mean(error)
    median_error = np.median(error)
    max_error = np.max(error)

    return mean_error, median_error, max_error


def main():
    experiments = [
        (1, 3, 5),
        (2, 4, 5),
        (3, 5, 7),
        (4, 6, 7),
        (5, 8, 9),
    ]

    print("| 실험 | PCA 차원 | KNN 이웃 수 | 평균 오차 | 중앙값 오차 | 최대 오차 |")
    

    best_exp = None
    best_mean = None

    for exp_num, pca_dim, k_neighbors in experiments:
        mean_error, median_error, max_error = evaluate(pca_dim, k_neighbors)

        print(
            f"| {exp_num} | {pca_dim} | {k_neighbors} | "
            f"{mean_error:.6f} | {median_error:.6f} | {max_error:.6f} |"
        )

        if best_mean is None or mean_error < best_mean:
            best_mean = mean_error
            best_exp = (exp_num, pca_dim, k_neighbors, mean_error, median_error, max_error)

    print()
    print("Best experiment")
    print("실험:", best_exp[0])
    print("PCA 차원:", best_exp[1])
    print("KNN 이웃 수:", best_exp[2])
    print("평균 오차:", best_exp[3])
    print("중앙값 오차:", best_exp[4])
    print("최대 오차:", best_exp[5])


if __name__ == "__main__":
    main()