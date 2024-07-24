import numpy as np
import pycolmap
import os
from pathlib import Path

def get_gravity(rotations: np.ndarray):
    """
    rotations are assumed to be world -> cam
    also assumes colmap conventions (x=right, y=down, z = forward)
    """
    n_rots, three, three = rotations.shape
    plane_normals = rotations[:,0,:]
    correlation = np.einsum("nc, nd -> cd", plane_normals, plane_normals)
    vals, vecs = np.linalg.eigh(correlation)
    # this gives us best gravity up-to-sign.
    up_to_sign_gravity = vecs[:,0]
    # get sign just by inner product with camera downs
    camera_downs = rotations[:,1]
    gravity = up_to_sign_gravity * np.sign((up_to_sign_gravity[None]*camera_downs).sum())
    return gravity

def get_gravity_from_colmap_reconstruction(reconstruction: pycolmap.Reconstruction):
    rotations = np.stack([x.cam_from_world.rotation.matrix() for x in reconstruction.images.values()])
    return get_gravity(rotations)

def get_gravity_from_colmap_reconstruction_path(reconstruction_path: os.PathLike):
    return get_gravity_from_colmap_reconstruction(pycolmap.Reconstruction(reconstruction_path))

if __name__ == "__main__":
    gravity_vec = get_gravity_from_colmap_reconstruction_path("data/chess/triangulated")
    print("hej")
