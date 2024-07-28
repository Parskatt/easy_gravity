from typing import Optional
import numpy as np
import pycolmap
import os
from enum import Enum

class Method(Enum):
    ROTATION_PLANES = "rotation_planes"
    AVG_CAM_DOWN = "avg_cam_down"
    CAM_POSITIONS = "cam_positions"

def get_gravity_from_rotation_planes(rotations: np.ndarray):
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

def get_gravity_from_avg_cam_down(rotations: np.ndarray):
    """
    rotations are assumed to be world -> cam
    also assumes colmap conventions (x=right, y=down, z = forward)
    """
    n_rots, three, three = rotations.shape
    camera_downs = rotations[:,1]
    gravity:np.ndarray = camera_downs.mean(axis=0)
    gravity = gravity/np.linalg.norm(gravity)
    return gravity

def get_gravity_from_camera_positions(positions: np.ndarray, rotations: np.ndarray):
    """
    """
    n_pos, three = positions.shape
    camera_downs = rotations[:,1]
    positions = positions - positions.mean(axis=0, keepdims=True)
    C = np.einsum("nc, nd -> cd", positions, positions)
    vals, vecs = np.linalg.eigh(C)
    up_to_sign_gravity = vecs[:,0]
    gravity = up_to_sign_gravity * np.sign((up_to_sign_gravity[None]*camera_downs).sum())
    return gravity


def get_gravity_from_colmap_reconstruction(
        reconstruction: pycolmap.Reconstruction, 
        method: Optional[Method|str] = Method.ROTATION_PLANES):
    if method == "im_feeling_lucky":
        method = np.random.choice(list(Method))
    else:
        method = Method(method)
    rotations = np.stack([x.cam_from_world.rotation.matrix() for x in reconstruction.images.values()])
    if method == Method.ROTATION_PLANES:
        return get_gravity_from_rotation_planes(rotations)
    elif method == Method.AVG_CAM_DOWN:
        return get_gravity_from_avg_cam_down(rotations)
    elif method == Method.CAM_POSITIONS:
        positions = np.stack([x.cam_from_world.inverse().translation for x in reconstruction.images.values()])
        # need the rotations here for a way to get the sign
        return get_gravity_from_camera_positions(positions, rotations)

def get_gravity_from_colmap_reconstruction_path(
        reconstruction_path: os.PathLike, 
        method: Optional[Method|str] = Method.ROTATION_PLANES):
    return get_gravity_from_colmap_reconstruction(pycolmap.Reconstruction(reconstruction_path), method = method)
