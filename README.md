def save_ply(points, filename, colors=None):
    """
    将点云保存为PLY文件
    :param points: numpy数组，形状为(n, 3)
    :param filename: 输出文件名，如'output.ply'
    :param colors: numpy数组，形状为(n, 3)
    """
    if isinstance(points, list):
        if colors is None:
            from src.utils.vis_utils import LST_COLOR
            lst_num_pt = [len(p) for p in points]
            colors = np.vstack([np.array([LST_COLOR[i]]).repeat(lst_num_pt[i], 0) for i in range(len(lst_num_pt))])
        points = np.vstack(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True, compressed=False, print_progress=True)

    with open(filename, 'w') as f:
        # 写入PLY文件头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property uchar alpha\n")
        f.write("end_header\n")

        # 写入点云数据
        if colors is not None:
            if colors.max() <= 1:
                colors = (colors * 255).astype(int)
            for point, color in zip(points, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])} {127}\n")
        else:
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f'Saved as {filename}')
