#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open3D GUI 可视化（SceneWidget）：
- 柱体绘制方式保持“线段拼柱体”的风格（中心线 + 顶/底环 + 母线）。
- 同时展示：整块原始点云（或整块管道点）、FN 周边点、预测柱体(与FN点相交)、FN 的 GT 柱体。
- 每根柱体旁显示 id 标签（SceneWidget.add_3d_label，失败时自动用彩色小球兜底）。
- 读取逻辑参照你的 pipeline（dir_root/*.pts + rel_dir_fitted/save_for_vis 里的 pkl）。

运行示例：
python visualize_20251111.py \
  --dir_root exp/NRDK/baseline_s2/cascade_result_seg_baseline_s1/ \
  --rel_dir_fitted cylinder_fitting_mcwei_20251020

可选：指定一个 .pts
python visualize_20251111.py \
  --dir_root exp/NRDK/baseline_s2/cascade_result_seg_baseline_s1/ \
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
    ap.add_argument("--dir_root", type=str,
                    default="exp/NRDK/baseline_s2/cascade_result_seg_baseline_s1/",
                    help="包含 .pts 的目录")
    ap.add_argument("--rel_dir_fitted", type=str,
                    default="cylinder_fitting_mcwei_20251020",
                    help="相对 dir_root 的拟合结果目录（其中应有 save_for_vis/*.pkl）")
    ap.add_argument("--file", type=str, default="",
                    help="可选：仅可视化该 .pts 文件（文件名或绝对/相对路径）")
    # 如果你想显示“完整原始点云（含非管道）”，把 --show_all_pc 设为 1
    ap.add_argument("--show_all_pc", type=int, default=0,
                    help="1=显示点云中全部点（未筛 cls==1），0=只显示管道点")
    return ap.parse_args()


# =========================
# 颜色/几何/标签工具
# =========================
LST_COLOR = [[0.6, 0.6, 0.6]] + [np.random.rand(3).tolist() for _ in range(100000)] + [[0, 0, 0]]
THR_NUM_PTS_VALID = 10  # FN 点数阈值（过小则跳过该 FN）


def color_from_id(cid):
    """根据 id 生成稳定颜色（也可改成用 LST_COLOR[cid+1]）"""
    try:
        h = hash(int(cid)) & 0xFFFFFF
    except Exception:
        h = hash(str(cid)) & 0xFFFFFF
    return [(h >> 0 & 0xFF) / 255.0, (h >> 8 & 0xFF) / 255.0, (h >> 16 & 0xFF) / 255.0]


def create_cylinder_line(start, end, radius=0.002, color=(1, 0, 0)):
    """用细圆柱段画线（更显眼），保持“线段柱体”的风格。"""
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    v = end - start
    L = np.linalg.norm(v)
    if L < 1e-8:
        return None
    axis = v / L

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=8)
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


def add_cylinder_lines_geoms(cyl, color, line_radius=0.002):
    """
    构造一根“线段风格”的圆柱体，返回若干 TriangleMesh：
      - 中心线 + 顶环 + 底环 + 母线（隔一个取一条）
    """
    geoms = []

    if isinstance(cyl, dict):
        start = np.array(cyl['start'], float)
        end = np.array(cyl['end'], float)
        radius = float(cyl['radius'])
    elif isinstance(cyl, list) and len(cyl) >= 3:
        start = np.array(cyl[0], float)
        end = np.array(cyl[1], float)
        radius = float(cyl[2])
    else:
        return geoms

    v = end - start
    height = np.linalg.norm(v)
    if height < 1e-6:
        return geoms
    v_unit = v / height

    # 中心线
    center_line = create_cylinder_line(start, end, radius=line_radius, color=color)
    if center_line is not None:
        geoms.append(center_line)

    # 正交基
    if abs(v_unit[0]) > 0.5:
        u = np.cross(v_unit, [0, 1, 0])
    else:
        u = np.cross(v_unit, [1, 0, 0])
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(v_unit, u)
    w = w / (np.linalg.norm(w) + 1e-12)

    # 顶/底环
    num_segments = 16
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    top_points = [start + radius * (np.cos(a) * u + np.sin(a) * w) for a in angles]
    bottom_points = [end + radius * (np.cos(a) * u + np.sin(a) * w) for a in angles]

    for i in range(num_segments):
        t = create_cylinder_line(top_points[i], top_points[(i + 1) % num_segments], radius=line_radius, color=color)
        b = create_cylinder_line(bottom_points[i], bottom_points[(i + 1) % num_segments], radius=line_radius, color=color)
        if t is not None:
            geoms.append(t)
        if b is not None:
            geoms.append(b)

    # 母线（隔一个）
    for i in range(0, num_segments, 2):
        e = create_cylinder_line(top_points[i], bottom_points[i], radius=line_radius, color=color)
        if e is not None:
            geoms.append(e)

    return geoms


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
# 距离/掩码（与原逻辑一致）
# =========================
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


# =========================
# GUI 可视化（整块点云 + FN 点 + 柱体 + 标签）
# =========================
def visualize_gui_one_block(lst_pred, lst_gt_fn, pts_fn=None, pts_full=None):
    """
    在一个 GUI 窗口里显示：
      - 预测柱体（与 FN 点相交的那些） + 标签 pred:id
      - FN 的 GT 柱体 + 标签 gt:id
      - 整块点云（灰） + FN 周边点（深灰更粗）
    """
    app = gui.Application.instance
    app.initialize()

    win = app.create_window("Cylinder Visualization (GUI / line-style)", 1280, 900)
    w3d = gui.SceneWidget()
    scene = rendering.Open3DScene(win.renderer)
    w3d.scene = scene
    win.add_child(w3d)

    # 背景：需要 RGBA float32 数组
    scene.set_background_color(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))  # 白底

    # ========== 点云 ==========
    # 整块点云（更浅的灰 + 较小点）
    if pts_full is not None and len(pts_full) > 0:
        cloud_full = o3d.geometry.PointCloud()
        cloud_full.points = o3d.utility.Vector3dVector(pts_full)
        col_full = np.full((pts_full.shape[0], 3), 0.92, dtype=float)  # 浅灰
        cloud_full.colors = o3d.utility.Vector3dVector(col_full)
        mat_full = rendering.MaterialRecord()
        mat_full.shader = "defaultUnlit"
        mat_full.point_size = 2.0 * win.scaling
        scene.add_geometry("points_full", cloud_full, mat_full)

    # FN 周边点（更深灰 + 更大点）
    if pts_fn is not None and len(pts_fn) > 0:
        cloud_fn = o3d.geometry.PointCloud()
        cloud_fn.points = o3d.utility.Vector3dVector(pts_fn)
        col_fn = np.full((pts_fn.shape[0], 3), 0.60, dtype=float)  # 深灰
        cloud_fn.colors = o3d.utility.Vector3dVector(col_fn)
        mat_fn = rendering.MaterialRecord()
        mat_fn.shader = "defaultUnlit"
        mat_fn.point_size = 3.5 * win.scaling
        scene.add_geometry("points_fn", cloud_fn, mat_fn)

    # ========== 柱体（线段风格）+ 标签 ==========
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    all_vertices = []

    # 预测柱体
    for c in lst_pred:
        col = color_from_id(c.get("id", 0))
        geoms = add_cylinder_lines_geoms(c, col, line_radius=0.002)
        for g in geoms:
            name = f"pred_{c.get('id',0)}_{id(g)}"
            scene.add_geometry(name, g, mat)
            all_vertices.append(np.asarray(g.vertices))

        pos = label_pos_from_cylinder(c["start"], c["end"], c["radius"], extra=max(0.08, 0.2 * float(c["radius"])))
        add_label_gui_or_fallback(w3d, scene, pos, f"pred:{c['id']}", color=col, scale=1.0)

    # FN 的 GT 柱体
    for c in lst_gt_fn:
        col = color_from_id(c.get("id", 0))
        geoms = add_cylinder_lines_geoms(c, col, line_radius=0.002)
        for g in geoms:
            name = f"gt_{c.get('id',0)}_{id(g)}"
            scene.add_geometry(name, g, mat)
            all_vertices.append(np.asarray(g.vertices))

        pos = label_pos_from_cylinder(c["start"], c["end"], c["radius"], extra=max(0.08, 0.2 * float(c["radius"])))
        add_label_gui_or_fallback(w3d, scene, pos, f"gt:{c['id']}", color=col, scale=1.1)

    # 相机自动对齐
    if all_vertices:
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(np.vstack(all_vertices))
        )
        w3d.setup_camera(60.0, bbox, bbox.get_center())

    app.run()


# =========================
# 单个 block 的数据组织（与原流程一致）
# =========================
def vis_single_block_gui(path_pc, dir_save_for_vis, show_all_pc=False):
    name = os.path.basename(path_pc)

    # 读取保存的拟合结果与评估
    res_fitting = pickle.load(open(os.path.join(dir_save_for_vis, f'{os.path.basename(path_pc)}.pkl'), 'rb'))
    res_eval = pickle.load(open(os.path.join(dir_save_for_vis, f'eval_{os.path.basename(path_pc)}.pkl'), 'rb'))

    gt = res_eval['gt']
    lst_idx_gt_unmatched = res_eval['ug']

    if not lst_idx_gt_unmatched:
        print(f"[skip] {name}：没有 FN")
        return

    # 读 .pts
    pc_full = np.loadtxt(path_pc)        # (n, 17) 原始（含非管道）
    pc = pc_full.copy()

    if not show_all_pc:
        # 只显示管道点（cls==1）
        # 下采样点（拟合使用的索引）
        ind_pc = res_fitting['ind_pts']
        pc = pc[ind_pc]
        pc = pc[pc[:, 10] == 1]  # 管道点
        pts_full = pc[:, :3]     # 整块管道点
    else:
        pts_full = pc_full[:, :3]  # 整块原始点云（包含非管道）

    # 用于筛 FN 周边点的 pc（与原逻辑一致，用“管道点空间”来筛）
    pc_for_fn = pc
    # FN 周边点掩码
    mask_pc_fn = np.zeros_like(pc_for_fn[:, 0], dtype=bool)
    lst_idx_gt_unmatched_valid = []
    for idx_gt in lst_idx_gt_unmatched:
        cyl = gt[idx_gt]
        mask_i = get_pc_mask_inside_cylinder(pc_for_fn[:, :3], cyl['start'], cyl['end'], cyl['radius'] + 0.02, len_ext=0.02)
        if int(mask_i.sum()) > THR_NUM_PTS_VALID:
            lst_idx_gt_unmatched_valid.append(idx_gt)
        mask_pc_fn |= mask_i

    if not lst_idx_gt_unmatched_valid:
        print(f"[skip] {name}：FN 点太少")
        return

    # 收集 GT(FN) 柱体，并补 id=gt 索引
    lst_gt_fn = []
    for idx_gt in lst_idx_gt_unmatched_valid:
        g = gt[idx_gt].copy()
        g["id"] = idx_gt
        lst_gt_fn.append(g)

    # 预测结果中过滤出与 FN 点有交集的柱体
    lst_pred = []
    ind_pc_fn = mask_pc_fn.nonzero()[0]
    for i, res in enumerate(res_fitting['results']):
        ind_inlier = res['ind_inlier']
        if len(np.intersect1d(ind_pc_fn, ind_inlier)):
            lst_pred.append(dict(start=res['start'], end=res['end'], radius=res['radius'], id=i + 1))

    # 仅保留可视化用到的 FN 周边点（灰色点当背景）
    pts_fn = pc_for_fn[mask_pc_fn][:, :3] if np.any(mask_pc_fn) else None

    print(f"[GUI] {name}: pred={len(lst_pred)}, gt_fn={len(lst_gt_fn)}, "
          f"fn_pts={0 if pts_fn is None else len(pts_fn)}, full_pts={len(pts_full)} "
          f"(show_all_pc={int(show_all_pc)})")

    visualize_gui_one_block(lst_pred, lst_gt_fn, pts_fn=pts_fn, pts_full=pts_full)


# =========================
# 主函数
# =========================
def main():
    args = parse_args()
    dir_root = args.dir_root
    dir_save_for_vis = os.path.join(args.dir_root, args.rel_dir_fitted, 'save_for_vis')

    # 选择 .pts
    if args.file:
        path_pc = args.file if os.path.isabs(args.file) else os.path.join(dir_root, args.file)
        if not os.path.isfile(path_pc):
            raise FileNotFoundError(f"file not found: {path_pc}")
        vis_single_block_gui(path_pc, dir_save_for_vis, show_all_pc=bool(args.show_all_pc))
        return

    lst_path_pc = sorted(glob.glob(os.path.join(dir_root, "*.pts")))
    if not lst_path_pc:
        raise FileNotFoundError(f"在 {dir_root} 下没有找到 .pts 文件。")
    # GUI 应用一次一个窗口：默认展示第一个
    vis_single_block_gui(lst_path_pc[0], dir_save_for_vis, show_all_pc=bool(args.show_all_pc))


if __name__ == "__main__":
    main()
