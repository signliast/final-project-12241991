import numpy as np
import scipy.io as sio
import pickle


def train_model():
    mat_path = "DH_FR1.mat"

    data = sio.loadmat(mat_path, squeeze_me=False)

    p_bs = np.asarray(data["p_bs"], dtype=float)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float)

    X = d_hat.T
    y = p.T

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0

    X_norm = (X - mean) / std

    U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)

    pca_dim = 3
    pca_components = Vt[:pca_dim, :]

    X_pca = X_norm @ pca_components.T

    k_neighbors = 5
    epsilon = 1e-6

    model = {
    "mean": mean,
    "std": std,
    "pca_components": pca_components,
    "X_pca": X_pca,
    "X_norm": X_norm,
    "y": y,
    "k_neighbors": k_neighbors,
    "epsilon": epsilon,
}

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("model.pkl 저장 완료")
    print("X shape:", X.shape)
    print("X_pca shape:", X_pca.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    train_model()