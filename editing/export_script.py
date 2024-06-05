import bpy
import numpy as np

def vec_as_np(v):
    return np.array([v.x, v.y])

spline = bpy.data.curves[0].splines[0]
N = len(spline.bezier_points)
bezier_pts = np.zeros((N,4,2))
for i in range(N):
    this_pt = spline.bezier_points[i]
    next_pt = spline.bezier_points[(i+1)%N]
    bezier_pts[i,0] = vec_as_np(this_pt.co)
    bezier_pts[i,1] = vec_as_np(this_pt.handle_right)
    bezier_pts[i,2] = vec_as_np(next_pt.handle_left)
    bezier_pts[i,3] = vec_as_np(next_pt.co)
    #print("Bezier " + str(i),
    #    this_pt.co, this_pt.handle_right,
    #    next_pt.handle_left, next_pt.co
    #)
np.savetxt("/home/nick/Documents/School/AE4350 Bio/editing/track1.dat", bezier_pts)