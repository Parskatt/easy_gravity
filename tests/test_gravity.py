def test_gravity():
    from easy_gravity import get_gravity_from_colmap_reconstruction_path
    path = "data/7scenes_sfm_triangulated/heads/triangulated"
    for method in ["rotation_planes","avg_cam_down","cam_positions"]:
        gravity = get_gravity_from_colmap_reconstruction_path(path, method=method)
        print(gravity)


if __name__ == "__main__":
    test_gravity()