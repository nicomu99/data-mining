import collections
import numpy as np

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vx_size_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]       # Step 1: Flip coordinates from IRC to CRI
    origin_a = np.array(origin_xyz)
    vx_size_xyz = np.array(vx_size_xyz)

    # Steps 2 - 4: Scale, mult with directions, add offset
    coords_xyz = (direction_a @ (cri_a * vx_size_xyz)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vx_size_xyz, direction_a):
    # Inverse transformation
    origin_a = np.array(origin_xyz)
    vx_size_xyz = np.array(vx_size_xyz)
    coord_a = np.array(coord_xyz)

    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vx_size_xyz   # Inverse of steps 2 - 4
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))    # Flip coordinates