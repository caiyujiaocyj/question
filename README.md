"""
使用 Open3D GUI (SceneWidget) 可视化点云 + 柱体，并在柱体旁显示其 id。

运行示例：
python 2222.py \
  --dir_root exp/NRDK/baseline_s2/cascade_result_seg_baseline_s1/
  --rel_dir_fitted cylinder_fitting_mcwei_20251020

可选：指定一个具体 .pts
python visualize_gui_labels.py \
  --dir_root exp/NRDK/baseline_s2/result_gt_seg \
  --rel_dir_fitted cylinder_fitting_mcwei_20251020 \
  --file block_1Ftest_11_7_2.pts
"""



import os
import glob
import argparse
import pickle
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


# =========================
# 参数解析
# =========================
def parse_args():
    ap = argparse.ArgumentParser("Open3D GUI visualize (with 3D labels)")
    ap.add_argument("--dir_root", type=str, default="exp/NRDK/baseline_s2/cascade_result_seg_baseline_s1/",
                    help="包含 .pts 的目录")
    ap.add_argument("--rel_dir_fitted", type=str, default="cylinder_fitting_mcwei_20251020",
                    help="相对 dir_root 的拟合结果目录（其中应有 save_for_vis/*.pkl）")
    ap.add_argument("--file", type=str, default="",
                    help="可选：仅可视化该 .pts 文件（文件名或绝对/相对路径）")
    return ap.parse_args()


# =========================
# 几何/颜色/标签工具
# =========================
def color_from_id(cid):
    """根据 id 生成稳定颜色（或可替换为你的 LST_COLOR）。"""
    try:
        h = hash(int(cid)) & 0xFFFFFF
    except Exception:
        h = hash(str(cid)) & 0xFFFFFF
    return [(h >> 0 & 0xFF) / 255.0, (h >> 8 & 0xFF) / 255.0, (h >> 16 & 0xFF) / 255.0]


def create_cylinder_line(start, end, radius=0.01, color=(1, 0, 0)):
    """用细圆柱段画线（比 LineSet 更显眼）。"""
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    v = end - start
    L = np.linalg.norm(v)
    if L < 1e-8:
        return None
    axis = v / L

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=24)
    mesh.paint_uniform_color(color)

    # 把默认 +Z 旋转到 axis
    z = np.array([0, 0, 1.0], float)
    k = np.cross(z, axis)
    s = np.linalg.norm(k)
    c = float(np.dot(z, axis))
    if s > 1e-8:
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
        mesh.rotate(R, center=(0, 0, 0))
    elif c < 0:  # 反向 180°
        R = np.diag([1, -1, -1])
        mesh.rotate(R, center=(0, 0, 0))

    mesh.translate((start + end) / 2.0)
    return mesh


def label_pos_from_cylinder(start, end, radius, extra=0.06):
    """把标签放到管道中心沿轴向外移 radius+extra 的位置，避免被遮挡。"""
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    center = (start + end) / 2.0
    v = end - start
    L = np.linalg.norm(v)
    axis = v / L if L > 1e-8 else np.array([0, 0, 1.0], float)
    return center + axis * (float(radius) + float(extra))


def add_label_gui_or_fallback(scene_widget, scene, pos, text, color=(0, 0, 0), scale=1.0, sphere_radius=0.04):
    """
    优先使用 SceneWidget.add_3d_label；若不可用则放彩色小球兜底（并打印）。
   """
    try:
        label = scene_widget.add_3d_label(np.asarray(pos, float).reshape(3,), str(text))
        # 0.19.0 支持 .color / .scale
        label.color = gui.Color(float(color[0]), float(color[1]), float(color[2]))
        label.scale = float(scale)
        return True
    except Exception:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sph.translate(pos)
        sph.paint_uniform_color(np.asarray(color, float))
        scene.add_geometry(f"label_sphere_{text}", sph, rendering.MaterialRecord())
        print(f"[fallback] label -> sphere at {np.asarray(pos).tolist()}, text={text}")
        return False


# =========================
# 读数据 + 选择 FN / 预测
# =========================
THR_NUM_PTS_VALID = 10


def point_to_line_distance_v2(pts, A, B):
    A = np.array(A)
    B = np.array(B)
    AB = B - A
    if np.linalg.norm(AB) < 1e-10:
        return np.linalg.norm(pts - A, axis=1)
    AP = pts - A
    cross_product = np.cross(AP, AB)
    cross_norm = np.linalg.norm(cross_product, axis=1)
    return cross_norm / np.linalg.norm(AB)


def get_pc_mask_inside_cylinder(pts, start, end, radius, len_ext=0.02):
    direction = end - start
    len_cyl = np.linalg.norm(direction)
    if len_cyl < 1e-6:
        return np.zeros_like(pts[:, 0], dtype=bool)
    direction = direction / len_cyl

    mask_inside_along_normal = point_to_line_distance_v2(pts, start, end) < radius
    start_ext = start - direction * len_ext
    end_ext = end + direction * len_ext
    ES = start_ext - end_ext
    SE = -ES
    SP = pts - start_ext
    EP = pts - end_ext
    mask_inside_along_direction = (np.sum(SP * SE, axis=1) > 0) & (np.sum(EP * ES, axis=1) > 0)
    return mask_inside_along_normal & mask_inside_along_direction


def robust_load_pkl(dir_save_for_vis, base):
    """
    兼容两种命名：
      save_for_vis/<name>.pts.pkl  或 save_for_vis/<name>.pkl
      save_for_vis/eval_<name>.pts.pkl 或 eval_<name>.pkl
    """
    stem = os.path.splitext(base)[0]
    # 主结果
    path_pkl = os.path.join(dir_save_for_vis, f"{base}.pkl")
    if not os.path.isfile(path_pkl):
        path_pkl = os.path.join(dir_save_for_vis, f"{stem}.pkl")
    res_fitting = pickle.load(open(path_pkl, "rb"))
    # 评估
    path_eval = os.path.join(dir_save_for_vis, f"eval_{base}.pkl")
    if not os.path.isfile(path_eval):
        path_eval = os.path.join(dir_save_for_vis, f"eval_{stem}.pkl")
    res_eval = pickle.load(open(path_eval, "rb"))
    return res_fitting, res_eval


# =========================
# GUI 可视化（单个 block）
# =========================
def visualize_block_gui(path_pts, dir_save_for_vis):
    """
    - 加载 path_pts 及其对应的 save_for_vis/*.pkl
    - 构造两类柱体：
        A) 预测里与 FN 点相交的柱（lst_pred）
        B) FN 的 GT 柱（lst_gt_fn，id = gt 的 index）
    - 显示点云（FN 周围的点）+ 两类柱体，并为每根柱体贴上 id 标签
    """
    base = os.path.basename(path_pts)
    print(f"[vis] {base}")

    # 读取 pkls
    res_fitting, res_eval = robust_load_pkl(dir_save_for_vis, base)

    gt = res_eval["gt"]
    lst_idx_gt_unmatched = res_eval["ug"]  # FN 的 gt 索引
    recall = (len(gt) - len(lst_idx_gt_unmatched)) / (len(gt) + 1e-6)
    print(f"  gt={len(gt)}, fn={len(lst_idx_gt_unmatched)}, recall={round(recall, 3)}")

    if not lst_idx_gt_unmatched:
        print("  没有 FN，直接结束。")
        return

    # 加载点云（原始 17 维）
    pc = np.loadtxt(path_pts)  # shape (n, 17)
    # 只取拟合用到的点
    ind_pc = res_fitting["ind_pts"]
    pc = pc[ind_pc]
    # 只保留语义为管道的点
    pc = pc[pc[:, 10] == 1]

    # 计算所有 FN 的点掩码 & 有效 FN 列表
    mask_pc_fn = np.zeros_like(pc[:, 0], dtype=bool)
    lst_idx_gt_unmatched_valid = []
    for idx_gt in lst_idx_gt_unmatched:
        cyl = res_eval["gt"][idx_gt]
        mask_i = get_pc_mask_inside_cylinder(pc[:, :3], cyl["start"], cyl["end"], cyl["radius"] + 0.02, len_ext=0.02)
        if int(mask_i.sum()) > THR_NUM_PTS_VALID:
            lst_idx_gt_unmatched_valid.append(idx_gt)
        mask_pc_fn |= mask_i
    if not lst_idx_gt_unmatched_valid:
        print("  没有点数充足的 FN 管道。")
        return
    print(f"  有效 FN 索引: {lst_idx_gt_unmatched_valid}")

    # 收集 FN 的 GT 柱体（每个 dict 含 start/end/radius/id）
    lst_gt_fn = []
    for idx_gt in lst_idx_gt_unmatched_valid:
        g = gt[idx_gt].copy()
        g["id"] = idx_gt  # 按你的需求，id 用 crop 后的 gt 索引
        lst_gt_fn.append(g)

    # 过滤出与 FN 点相交的预测柱体
    pc_fitted = pc.copy()
    pc_fitted[:, 11] = -1
    pc_fitted[:, 12] = -1
    ind_pc_fn = mask_pc_fn.nonzero()[0]
    lst_pred = []
    for i, res in enumerate(res_fitting["results"]):
        ind_clustered = res["ind_clustered"]
        ind_inlier = res["ind_inlier"]
        start = res["start"]; end = res["end"]; radius = res["radius"]
        # 与 FN 点有交集就保留
        if len(np.intersect1d(ind_pc_fn, ind_inlier)):
            lst_pred.append(dict(start=start, end=end, radius=radius, id=i + 1))

    # 仅显示 FN 周围的点
    pc_fn = pc[mask_pc_fn]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc_fn[:, :3])
    # 灰白色点
    gray = np.full((pc_fn.shape[0], 3), 0.8, dtype=float)
    cloud.colors = o3d.utility.Vector3dVector(gray)

    # ====== GUI 构建 ======
    app = gui.Application.instance
    app.initialize()

    win = app.create_window(f"O3D GUI – {base}", 1280, 800)
    w3d = gui.SceneWidget()
    scene = rendering.Open3DScene(win.renderer)
    w3d.scene = scene
    win.add_child(w3d)

    # 背景和材料
    w3d.scene.set_background([1, 1, 1, 1])  # 白底
    mat_pts = rendering.MaterialRecord()
    mat_pts.shader = "defaultUnlit"
    mat_pts.point_size = 2.5 * win.scaling

    # 添加点云
    scene.add_geometry("points_fn", cloud, mat_pts)

    # 添加“预测柱体”——一种颜色系（用 id 上色）
    for i, c in enumerate(lst_pred):
        col = color_from_id(c["id"])
        # 线宽取半径的 40%，并保证最小可见
        mesh = create_cylinder_line(c["start"], c["end"], radius=max(0.4 * float(c["radius"]), 0.01), color=col)
        if mesh is not None:
            scene.add_geometry(f"pred_{i}", mesh, rendering.MaterialRecord())
        # 标签（id）
        pos = label_pos_from_cylinder(c["start"], c["end"], c["radius"], extra=max(0.08, 0.2 * float(c["radius"])))
        add_label_gui_or_fallback(w3d, scene, pos, f"pred:{c['id']}", color=col, scale=1.1)

    # 添加“FN 的 GT 柱体”——另一种颜色系（同样按 id 上色）
    for i, c in enumerate(lst_gt_fn):
        col = color_from_id(c["id"])
        mesh = create_cylinder_line(c["start"], c["end"], radius=max(0.4 * float(c["radius"]), 0.01), color=col)
        if mesh is not None:
            scene.add_geometry(f"gtfn_{i}", mesh, rendering.MaterialRecord())
        # 标签（id）
        pos = label_pos_from_cylinder(c["start"], c["end"], c["radius"], extra=max(0.08, 0.2 * float(c["radius"])))
        # 给 GT 标签加个“gt:”前缀，避免与 pred 混淆
        add_label_gui_or_fallback(w3d, scene, pos, f"gt:{c['id']}", color=col, scale=1.2)

    # 相机
    bbox = scene.bounding_box
    w3d.setup_camera(60.0, bbox, bbox.get_center())

    # 运行窗口（关闭窗口后程序结束）
    app.run()


# =========================
# 主流程
# =========================
def main():
    args = parse_args()
    dir_root = args.dir_root
    dir_save_for_vis = os.path.join(args.dir_root, args.rel_dir_fitted, "save_for_vis")

    # 选择一个 .pts
    if args.file:
        # 支持给文件名或完整路径
        path_pts = args.file if os.path.isabs(args.file) else os.path.join(dir_root, args.file)
        if not os.path.isfile(path_pts):
            raise FileNotFoundError(f"file not found: {path_pts}")
        visualize_block_gui(path_pts, dir_save_for_vis)
        return

    lst_pts = sorted(glob.glob(os.path.join(dir_root, "*.pts")))
    if not lst_pts:
        raise FileNotFoundError(f"no .pts under {dir_root}")
    # 默认展示第一个
    visualize_block_gui(lst_pts[0], dir_save_for_vis)


if __name__ == "__main__":
    main()
