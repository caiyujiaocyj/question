
    # -------------------------------------------------------
    #  在 generate_pred 内保存每个 cluster 的点（保存 PLY）
    # -------------------------------------------------------
    save_cluster_root = os.path.join(out_dir, "save_for_cluster")
    os.makedirs(save_cluster_root, exist_ok=True)

    block_name = cloud.split('/')[-1].split('.')[0]
    save_block_dir = os.path.join(save_cluster_root, block_name)
    os.makedirs(save_block_dir, exist_ok=True)

    unique_labels = np.unique(lbl)
    valid_labels = unique_labels[unique_labels >= 0]

    for lb in valid_labels:
        pts = surf[lbl == lb]  # cluster 点
        ply_path = os.path.join(save_block_dir, f"{lb}.ply")

        # 颜色设置：使用已有的 LST_COLOR 或纯色
        if pts.shape[0] > 0:
            colors = np.ones((pts.shape[0], 3)) * (lb * 37 % 255 / 255.0)  # 默认颜色
        else:
            colors = None

        save_ply(pts, ply_path, colors=colors)