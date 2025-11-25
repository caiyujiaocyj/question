import numpy as np
import pipe_function as pn 

def fit_from_saved_cluster(cluster_npz_path):
    data = np.load(cluster_npz_path)
    c = data["centerline"]
    r = data["radii"]
    d = data["dirs"]
    surf = data["surf"]

    # 单个 cluster 的 label 全部置为 0 即可
    labels = np.zeros(len(c), dtype=int)

    segs, _ = pn.extract_segments(c, r, d, labels, surf)
    segs, raw_r, new_r = pn.filter_segments(segs)

    # 一般来说这里 segs 只有一根 cylinder
    return segs

segs = fit_from_saved_cluster("/media/samsung/samsung/mc.wei/code/fitting_latest/exp/NRDK/baseline_s2/cascade_result_seg_baseline_s1/cylinder_fitting_wo_merge__11_25_16_54/save_for_cluster/block_1Ftest_0_0_1/0.npz")
seg = segs[0]
print(seg.centerline_start, seg.centerline_end, seg.radius)
