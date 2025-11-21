完善这份代码，输出文件放置在def proc(file）函数下out输出路径下，在该路径下新建一个名为save_for_cluster的文件夹，在此文件夹下，再以block的名字为文件夹，在文件夹下，使得 将相同标签的点单独保存为一个文件，文件名称就是标签名称，不保存标签为-1的情况，
        # -------------------------------------------------------
        # 保存每个 cluster 的点
        # -------------------------------------------------------
        # 创建路径: out/save_for_cluster/block_name/
        save_cluster_root = os.path.join(out, "save_for_cluster")
        os.makedirs(save_cluster_root, exist_ok=True)

        block_name = file_name.split('.')[0]  # 例如 block_1Ftest_10_3_1
        save_block_dir = os.path.join(save_cluster_root, block_name)
        os.makedirs(save_block_dir, exist_ok=True)

        # lbl 与 surf, c 对应数量一致，取 label >= 0 的 cluster
        unique_labels = np.unique(lbl)
        valid_labels = unique_labels[unique_labels >= 0]

        # 保存每个簇的点（这里保存的是 surf 点，如需保存 c 可替换 surf -> c）
        for lb in valid_labels:
            pts = surf[lbl == lb]  # 按标签筛选点
            np.save(os.path.join(save_block_dir, f"{lb}.npy"), pts)