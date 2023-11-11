import numpy as np

def train(dataset, Tmax_c, Tmax_d, error=3, csa_eig_num=5):
    silhouette_shape = dataset[0]["gait_sequence"][0].shape
    U_c = _init(silhouette_shape)
    U_c, V_c = _CSA(dataset, U_c, Tmax_c, silhouette_shape, error, csa_eig_num)
    reduced_dataset = _project_CSA(dataset, U_c, V_c)
    silhouette_shape = reduced_dataset[0]["gait_sequence"][0].shape
    U_d = _init(silhouette_shape)
    # TODO

def _init(shape):
    # Step 1 or 5
    return np.eye(shape)

def _CSA(dataset, U, Tmax_c, shape, error, csa_eig_num):
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
        V_new = eigvec[:, :csa_eig_num]
        # Step 2(b)
        G = []
        for datapoint in dataset:
            for silhouette in datapoint["gait_sequence"]:
                G += [silhouette @ V_new @ V_new.T]
        G = np.concatenate(G, axis=1)
        eigval, eigvec = np.linalg.eig(G @ G.T)
        eigvec = eigvec[:, np.argsort(eigval)[::-1]]
        U_new = eigvec[:, :csa_eig_num]
        # Step 2(c)
        U_error = np.linalg.norm(U_new - U, "fro")
        V_error = np.linalg.norm(V_new - V, "fro")
        if t > 1 and U_error < m * error and V_error < n * error:
            break
        U, V = U_new, V_new
    # Step 3
    return U, V

def _project_CSA(dataset, U, V):
    # Step 4
    projected_dataset = []
    for datapoint in dataset:
        new_datapoint = datapoint.copy()
        projected_gait_sequence = []
        for silhouette in datapoint["gait_sequence"]:
            projected_gait_sequence += [U.T @ silhouette @ V]
        new_datapoint["gait_sequence"] = projected_gait_sequence
        projected_dataset.append(new_datapoint)
    return projected_dataset
