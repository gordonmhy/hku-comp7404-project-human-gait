import numpy as np

def train(dataset, Tmax_c, Tmax_d):
    silhouette_shape = dataset[0]["gait_sequence"][0].shape
    U = _init(silhouette_shape)
    U, V = _CSA(dataset, U, Tmax_c, silhouette_shape, 3)

def _init(shape):
    # Step 1
    return np.eye(shape)

def _CSA(dataset, U, Tmax_c, shape, error):
    m, n = shape
    for t in range(Tmax_c):
        # Step 2(a)
        F = []
        for datapoint in dataset:
            for silhouette in datapoint["gait_sequence"]:
                F += [U @ U.T @ silhouette]
        F = np.concatenate(F, axis=1)
        eigval, eigvec = np.linalg.eig(F @ F.T)
        eigvec = eigvec[:, np.argsort(eigval)[::-1]]
        V_new = eigvec[:, :5]
        # Step 2(b)
        G = []
        for datapoint in dataset:
            for silhouette in datapoint["gait_sequence"]:
                G += [silhouette @ V_new @ V_new.T]
        G = np.concatenate(G, axis=1)
        eigval, eigvec = np.linalg.eig(G @ G.T)
        eigvec = eigvec[:, np.argsort(eigval)[::-1]]
        U_new = eigvec[:, :5]
        # Step 2(c)
        U_error = np.linalg.norm(U_new - U, "fro")
        V_error = np.linalg.norm(V_new - V, "fro")
        if t > 1 and U_error < m * error and V_error < n * error:
            break
        U, V = U_new, V_new
    # Step 3
    return U, V
