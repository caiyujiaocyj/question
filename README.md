[GUI] block_1Ftest_0_0_1.pts: pred=6, gt_fn=3, fn_pts=250
FEngine (64 bits) created at 0x561befb5c1a0 (threading is enabled)
FEngine resolved backend: OpenGL
Traceback (most recent call last):
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 882, in <module>
    main()
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 878, in main
    vis_single_block_gui(lst_path_pc[0], dir_save_for_vis)
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 855, in vis_single_block_gui
    visualize_gui_one_block(lst_pred, lst_gt_fn, pts_fn)
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 744, in visualize_gui_one_block
    scene.set_background_color(gui.Color(1, 1, 1))  # 白底更清晰
TypeError: set_background_color(): incompatible function arguments. The following argument types are supported:
    1. (self: open3d.cuda.pybind.visualization.rendering.Open3DScene, arg0: numpy.ndarray[numpy.float32[4, 1]]) -> None


scene.set_background_color(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))  




Invoked with: <open3d.cuda.pybind.visualization.rendering.Open3DScene object at 0x7fa6e0bce730>, <open3d.cuda.pybind.visualization.gui.Color object at 0x7fa6e0bce970>
