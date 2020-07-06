import open3d as o3d
import numpy as np
from scipy import linalg
import pdb

# Mainly quated from Error analysis for circle fitting algorithms
# https://arxiv.org/pdf/0907.0421.pdf
def sphere_fitting(points):
    n = points.shape[0]
    radius_ = points[:, 0]*points[:, 0] \
        + points[:, 1] * points[:, 1] \
        + points[:, 2]*points[:, 2]
    Z = np.c_[radius_, points, np.ones(n)]
    M = Z.transpose().dot(Z)/n
    P = np.zeros([5, 5])
    T = np.zeros([5, 5])

    P[4, 0] = P[0, 4] = -2
    P[1, 1] = P[2, 2] = P[3, 3] = 1
    T[0, 0] = 4 * M[0, 4]
    T[0, 1] = T[1, 0] = 2 * M[0, 3]
    T[0, 2] = T[2, 0] = 2 * M[0, 2]
    T[0, 3] = T[3, 0] = 2 * M[0, 1]
    T[1, 1] = T[2, 2] = T[3, 3] = 1
    H = 2*T-P
    if(np.sum(np.isnan(M)) > 0 or np.sum(np.isnan(H)) > 0):
        coeff = np.zeros(4)
        status = False
        return points, coeff, status

    eigvals, eigvecs = linalg.eig(M, H)
    eigvals[np.where(eigvals < 0)] = np.inf
    sort_idx = np.argsort(np.abs(eigvals))
    min_eig_var_idx = sort_idx[0]
    _coeff = eigvecs[:, min_eig_var_idx]
    coeff = np.zeros(4)
    coeff[0] = - _coeff[1]/(2*_coeff[0])
    coeff[1] = - _coeff[2]/(2*_coeff[0])
    coeff[2] = - _coeff[3]/(2*_coeff[0])
    coeff[3] = np.sqrt(
        (_coeff[1]*_coeff[1] + _coeff[2]*_coeff[2] +
         _coeff[3] * _coeff[3] - 4 * _coeff[0] * _coeff[4])
        / (4*_coeff[0] * _coeff[0])
    )
    status = True
    if(np.sum(np.isnan(coeff)) > 0):
        status = False
    return coeff, status


sphere_points = np.random.rand(3000, 3) - np.array([0.5, 0.5, 0.5])
sphere_points /= np.linalg.norm(sphere_points, axis=1, keepdims=True) * 0.30
sphere_points += np.random.rand(3000, 3) * 0.05
coeffs, _ = sphere_fitting(sphere_points)

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=coeffs[3])
mesh_sphere.translate(coeffs[:3])
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(sphere_points)
o3d.visualization.draw_geometries([pcd, mesh_sphere])
