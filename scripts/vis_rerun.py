from easy_gravity import get_gravity_from_colmap_reconstruction
import pycolmap
import rerun as rr
import numpy as np

reconstr = pycolmap.Reconstruction("data/7scenes_sfm_triangulated/stairs/triangulated")

gravity = get_gravity_from_colmap_reconstruction(reconstr)
xyz = [pt_3d.xyz for pt_3d in reconstr.points3D.values()]
colors = [pt_3d.color for pt_3d in reconstr.points3D.values()]

rr.init("gravity_vis", spawn = True)
rr.log("/", rr.ViewCoordinates.RDF)
rr.log("world/pts_3d", rr.Points3D(positions=xyz, colors=colors))
rr.log("world/est_gravity", rr.Arrows3D(origins=np.zeros_like(gravity), vectors=gravity))

