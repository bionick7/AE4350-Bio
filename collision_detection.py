import numpy as np
import matplotlib.pyplot as plt
from time import time


# https://github.com/LeanderSilur/Snippets/blob/master/bezier/distance_point_curve/bezier.py


def get_bezier_coeffs(points: np.ndarray) -> np.ndarray:
    co_coeffs = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    coeffs = np.multiply(co_coeffs.reshape((4, 4, 1)), points.reshape((1, 4, 2)))
    return np.sum(coeffs, axis=1).reshape(-1, 4, 2)


def bezier_eval(coeffs: np.ndarray, t) -> np.ndarray:
    exp3_1 = np.array([[[3, 3], [2, 2], [1, 1], [0, 0]]], dtype=np.float32)
    if type(t) != np.ndarray:
        t = np.array(t)
    pts = coeffs * np.power(t, exp3_1)
    return np.sum(pts, axis = 1)


def np_real_roots(coefficients, EPSILON=1e-6):
    r = np.roots(coefficients)
    return r.real[abs(r.imag) < EPSILON]


def bezier_distance(coeffs: np.ndarray, pt: np.ndarray) -> np.ndarray:
    exp4 = np.array([[4], [3], [2], [1], [0]], dtype=np.float32)

    # These are the coefficients of the derivatives d/dx and d/(d/dx).
    da = 6 *  np.sum(coeffs[:, 0] * coeffs[:, 0])
    db =10 *  np.sum(coeffs[:, 0] * coeffs[:, 1])
    dc = 4 * (np.sum(coeffs[:, 1] * coeffs[:, 1])       + 2*np.sum(coeffs[:, 0] * coeffs[:, 2]))
    dd = 6 * (np.sum(coeffs[:, 0] *(coeffs[:, 3] - pt)) +   np.sum(coeffs[:, 1] * coeffs[:, 2]))
    de = 2 * (np.sum(coeffs[:, 2] * coeffs[:, 2]))      + 4*np.sum(coeffs[:, 1] *(coeffs[:, 3] - pt))
    df = 2 *  np.sum(coeffs[:, 2] *(coeffs[:, 3] - pt))

    dda = 5*da
    ddb = 4*db
    ddc = 3*dc
    ddd = 2*dd
    dde = de
    dcoeffs = np.stack([da, db, dc, dd, de, df])
    ddcoeffs = np.stack([dda, ddb, ddc, ddd, dde]).reshape(-1, 1)
    
    # Calculate the real extremes, by getting the roots of the first
    # derivativ of the distance function.
    extrema = np_real_roots(dcoeffs)
    # Remove the roots which are out of bounds of the clipped range [0, 1].
    dd_clip = (np.sum(ddcoeffs * np.power(extrema, exp4)) >= 0) & (extrema > 0) & (extrema < 1)
    minima = extrema[dd_clip]

    # Add the start and end position as possible positions.
    potentials = np.concatenate((minima, np.array([0,1])))

    # Calculate the points at the possible parameters t and 
    # get the index of the closest
    points = bezier_eval(coeffs, potentials.reshape(-1, 1, 1))
    distances = np.linalg.norm(points - pt, axis = 1)
    min_dist_index = np.argmin(distances)

    return points[min_dist_index,:], distances[min_dist_index]


def tests():
    bezier_pts = np.array([[100, 100], [100, 200], [500, 30], [400, 200]])
    bezier_coeffs = get_bezier_coeffs(bezier_pts)
    bezier_evals = bezier_eval(bezier_coeffs, np.linspace(0, 1, 100).reshape(-1,1,1))
    np.random.seed(0)
    fig = plt.figure()
    axis = fig.add_subplot()
    axis.plot(bezier_pts[:,0],   bezier_pts[:,1],   'rx')
    axis.plot(bezier_pts[:,0],   bezier_pts[:,1],   'r:')
    axis.plot(bezier_evals[:,0], bezier_evals[:,1], 'b-')

    dist_pts = np.random.random((5, 2)) * 200
    dist_pts[:,0] *= 2
    axis.plot(dist_pts[:,0], dist_pts[:,1], 'go')
    for i in range(5):
        closest_pt, dist = bezier_distance(bezier_coeffs, dist_pts[i])
        print(closest_pt - dist_pts[i], dist)
        axis.plot(closest_pt[0], closest_pt[1], 'gx')
    axis.axes.set_aspect('equal')
    axis.grid()
    fig.show()
    plt.show()

    t0 = time()
    for i in range(1000):
        bezier_distance(bezier_coeffs, dist_pts[0])
    print(f"{time() - t0} ms")


if __name__ == "__main__":
    tests()