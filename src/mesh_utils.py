import open3d as o3d
import numpy as np

def get_render(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    # Add the mesh to the visualizer
    vis.add_geometry(mesh)

    # Set up camera parameters
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    # intrinsic = params.intrinsic
    extrinsic = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 5],
        [0, 0, 0, 1]
    ])
    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("./data/cup/rendered.png", do_render=True)
    vis.capture_depth_point_cloud("./data/cup/depth.ply", do_render=True)
    vis.destroy_window()
    