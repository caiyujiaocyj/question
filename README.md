
    # -------------------------------------------------------
    #  在 generate_pred 内保存每个 cluster 的点
    # -------------------------------------------------------
    # 保存路径为 out_dir/save_for_cluster/block_name/
    save_cluster_root = os.path.join(out_dir, "save_for_cluster")
    os.makedirs(save_cluster_root, exist_ok=True)

    # block 名字 (如 block_1F_test_3_5)
    block_name = cloud.split('/')[-1].split('.')[0]
    save_block_dir = os.path.join(save_cluster_root, block_name)
    os.makedirs(save_block_dir, exist_ok=True)

    # 获取有效标签
    unique_labels = np.unique(lbl)
    valid_labels = unique_labels[unique_labels >= 0]

    # 保存每个 cluster 的 surf 点（你也可以换成 c）
    for lb in valid_labels:
        pts = surf[lbl == lb]  # 所有属于该 cluster 的点
        np.save(os.path.join(save_block_dir, f"{lb}.npy"), pts)