"""
/!\ This file is the work of Chan Ho Long and Mak Ho Yin from
/!\ the University of Hong Kong for the group project of
/~\ COMP7404 (Computational Intelligence and Machine Learning)
"""

import statistics as stat
import numpy as np
import scipy as sp

class CSA_DATER:

    def __init__(self, Tmax_c=3, Tmax_d=3, error_c=3, error_d=3, m_dash_c=5, n_dash_c=5, m_dash_d=3, n_dash_d=3):
        self.Tmax_c = Tmax_c
        self.Tmax_d = Tmax_d
        self.error_c = error_c
        self.error_d = error_d
        self.m_dash_c = m_dash_c
        self.n_dash_c = n_dash_c
        self.m_dash_d = m_dash_d
        self.n_dash_d = n_dash_d
        self.U = None
        self.V = None


    def train(self, dataset):
        self.dataset = dataset   
        silhouette_shape = dataset[0]["gait_sequence"][0].shape
        U_c, V_c = self._CSA(dataset, self.Tmax_c, silhouette_shape, self.error_c, self.m_dash_c, self.n_dash_c)
        reduced_dataset = self._project_CSA(dataset, U_c, V_c)
        silhouette_shape = reduced_dataset[0]["gait_sequence"][0].shape
        U_d, V_d = self._DATER(reduced_dataset, self.Tmax_d, silhouette_shape, self.error_d, self.m_dash_d, self.n_dash_d)
        self.U, self.V = self._output_final(U_c, U_d, V_c, V_d)
        return self.U, self.V


    def predict(self, probe):
        probe_sequence = self.project(probe["gait_sequence"])
        distances = []
        for gallery in self.dataset:
            gallery_sequence = self.project(gallery["gait_sequence"])
            min_distances = []
            for probe_silhouette in probe_sequence:
                # For each probe silhouette, find the most similar one from the gallery
                dists = [np.linalg.norm(probe_silhouette - gallery_silhouette, ord="fro") for gallery_silhouette in gallery_sequence]
                min_distances.append(min(dists))
            # Obtain the distance between this gallery and the probe
            # (Median among distances between silhouettes of the probe and this gallery)
            median_distance = stat.median(min_distances)
            distances.append(median_distance)
        # Nearest Neighbor (among distances between this probe and all galleries)
        index = np.argmin(distances)
        return self.dataset[index]["subject_id"]


    def project(self, sequence):
        if self.U is None or self.V is None:
            return sequence
        projected_sequence = []
        for silhouette in sequence:
            projected_sequence.append(self.U.T @ silhouette @ self.V)
        return projected_sequence


    def _init(self, m_dash, n_dash):
        # Step 1 or 5
        A = np.random.rand(m_dash, n_dash)
        Q, _ = sp.linalg.qr(A)
        return Q


    def _CSA(self, dataset, Tmax_c, shape, error, m_dash_c, n_dash_c):
        m, n = shape
        U = self._init(m, m_dash_c)
        for t in range(Tmax_c):
            # Step 2(a)
            F = []
            for datapoint in dataset:
                for silhouette in datapoint["gait_sequence"]:
                    F += [U @ U.T @ silhouette]
            F = np.concatenate(F, axis=1)
            eigval, eigvec = sp.linalg.eig(F @ F.T)
            eigvec = eigvec[:, np.argsort(eigval)[::-1]]
            V_new = eigvec[:, :n_dash_c]
            # Step 2(b)
            G = []
            for datapoint in dataset:
                for silhouette in datapoint["gait_sequence"]:
                    G += [silhouette @ V_new @ V_new.T]
            G = np.concatenate(G, axis=1)
            eigval, eigvec = sp.linalg.eig(G @ G.T)
            eigvec = eigvec[:, np.argsort(eigval)[::-1]]
            U_new = eigvec[:, :m_dash_c]
            # Step 2(c)
            if t > 1:
                U_error = np.linalg.norm(U_new - U, "fro")
                V_error = np.linalg.norm(V_new - V, "fro")
                if U_error < m * error and V_error < n * error:
                    return U_new, V_new
            U, V = U_new, V_new
        # Step 3
        return U, V


    def _project_CSA(self, dataset, U, V):
        # Step 4
        projected_dataset = []
        for datapoint in dataset:
            new_datapoint = datapoint.copy()
            projected_gait_sequence = []
            for silhouette in datapoint["gait_sequence"]:
                projected_gait_sequence.append(U.T @ silhouette @ V)
            new_datapoint["gait_sequence"] = projected_gait_sequence
            projected_dataset.append(new_datapoint)
        return projected_dataset


    def _DATER(self, dataset, Tmax_d, shape, error, m_dash_d, n_dash_d):
        m, n = shape
        U = self._init(m, m_dash_d)
        # Group by class and calculate overall mean
        classes = dict()
        Xavg = []
        for datapoint in dataset:
            label = datapoint["subject_id"]
            if label not in classes.keys():
                classes[label] = dict()
                classes[label]["samples"] = list()
            classes[label]["samples"] += datapoint["gait_sequence"]
            Xavg += datapoint["gait_sequence"]
        Xavg = np.mean(Xavg, axis=0)
        # Calculate mean of samples by class
        for label, class_info in classes.items():
            class_info["Xavg_c"] = np.mean(class_info["samples"], axis=0)
        for t in range(Tmax_d):
            # Step 6(a)
            Xuavg = U.T @ Xavg
            Su_b, Su_w = [None] * 2
            for label, class_info in classes.items():
                Xuavg_c = U.T @ class_info["Xavg_c"]
                n_c = len(class_info["samples"])
                delta = n_c * ((Xuavg_c - Xuavg).T @ (Xuavg_c - Xuavg))
                Su_b = delta if Su_b is None else Su_b + delta
                for silhouette in class_info["samples"]:
                    Xu_i = U.T @ silhouette
                    delta = (Xu_i - Xuavg_c).T @ (Xu_i - Xuavg_c)
                    Su_w = delta if Su_w is None else Su_w + delta
            eigval, eigvec = sp.linalg.eig(Su_b, Su_w)
            eigvec = eigvec[:, np.argsort(eigval)[::-1]]
            V_new = eigvec[:, :n_dash_d]
            # Step 6(b)
            Xvavg = Xavg @ V_new
            Sv_b, Sv_w = [None] * 2
            for label, class_info in classes.items():
                Xvavg_c = class_info["Xavg_c"] @ V_new
                delta = n_c * ((Xvavg_c - Xvavg) @ (Xvavg_c - Xvavg).T)
                Sv_b = delta if Sv_b is None else Sv_b + delta
                for silhouette in class_info["samples"]:
                    Xv_i = silhouette @ V_new
                    delta =  (Xv_i - Xvavg_c) @ (Xv_i - Xvavg_c).T
                    Sv_w = delta if Sv_w is None else Sv_w + delta
            eigval, eigvec = sp.linalg.eig(Sv_b, Sv_w)
            eigvec = eigvec[:, np.argsort(eigval)[::-1]]
            U_new = eigvec[:, :m_dash_d]
            # Step 6(c)
            if t > 1:
                U_error = np.linalg.norm(U_new - U, "fro")
                V_error = np.linalg.norm(V_new - V, "fro")
                if U_error < m * error and V_error < n * error:
                    return U_new, V_new
            U, V = U_new, V_new
        # Step 7
        return U, V


    def _output_final(self, U_c, U_d, V_c, V_d):
        # Step 8
        return U_c @ U_d, V_c @ V_d
