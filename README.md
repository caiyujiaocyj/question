def centralize_data(data, central=True):
    """
    将点云数据中心化（减去均值）并记录偏移量
    参数:
        data: 点云数据，形状为 (N, 3+) 的 NumPy 数组，前3列必须是 [x, y, z]
        central: 是否执行中心化
    返回:
        data_centralized: 中心化后的数据（如果 central=False 则返回原数据）
        offset: 三个方向的偏移量 [x_offset, y_offset, z_offset]
    """
    if not central:
        return data, np.zeros(3)  # 不中心化时返回原数据和零偏移

    # 提取前3列坐标（x, y, z）
    xyz = data[:, :3]

    # 计算各方向的均值（偏移量）
    offset = np.mean(xyz, axis=0)

    # 中心化：减去偏移量
    data_centralized = data.copy()
    data_centralized[:, :3] = xyz - offset

    return data_centralized, offset
