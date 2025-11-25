
    save_cluster_root = os.path.join(out_dir, "save_for_cluster")
    os.makedirs(save_cluster_root, exist_ok=True)

    # block 名字 (如 block_1F_test_3_5)
    block_name = cloud.split('/')[-1].split('.')[0]
    save_block_dir = os.path.join(save_cluster_root, block_name)
    os.makedirs(save_block_dir, exist_ok=True)

    # 获取有效标签
    unique_labels = np.unique(lbl)
    valid_labels = unique_labels[unique_labels >= 0]

    # 保存每个 cluster 的各种信息，方便后续单独拟合
    for lb in valid_labels:
        mask = (lbl == lb)

        # pipe_function 里拟合所需的几类数据：
        surf_cluster = surf[mask]     # 表面点
        c_cluster    = c[mask]        # centerline 候选点
        r_cluster    = r[mask]        # 半径
        d_cluster    = d[mask]        # 方向

        # 在当前 pipe 点数组中的索引（0..n_pipe-1）
        ind_in_pipe_array = np.where(mask)[0]

        # 在原始点云中的索引（load_txt_data 返回的 instance_id）
        # 如果 instance_id 是 numpy 数组并且长度一致，就一起保存
        cluster_data = {
            "surf": surf_cluster,
            "centerline": c_cluster,
            "radii": r_cluster,
            "dirs": d_cluster,
            "ind_in_pipe_array": ind_in_pipe_array,
        }

        if isinstance(instance_id, np.ndarray) and len(instance_id) == len(c):
            cluster_data["instance_id"] = instance_id[mask]

        # 如果有法向量，也一并存下
        if isinstance(sur_normals, np.ndarray) and len(sur_normals) == len(c):
            cluster_data["sur_normals"] = sur_normals[mask]

        # 保存为 npz，文件名还是用 cluster id
        save_path = os.path.join(save_block_dir, f"{lb}.npz")
        np.savez(save_path, **cluster_data)