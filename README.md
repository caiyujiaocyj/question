#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在原 visualize_20251021_cyj.py 基础上修复：
1) IndexError: pcd 应为 dict，现已统一为 dict 结构传入 vis_custom；
2) 为柱体添加 ID 标签：优先 add_3d_label，fallback 为彩色小球；
3) 保留全部原功能与“线段风格”柱体绘制。
"""

import copy
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm
import argparse
import open3d as o3d
from open3d import geometry

# --------------------------------
# 配置
# --------------------------------
LST_COLOR = [[0.6, 0.6, 0.6]] + [np.random.rand(3).tolist() for _ in range(100000)] + [[0, 0, 0]]
THR_NUM_PTS_VALID = 10


# --------------------------------
# 参数
# --------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir_root', default='exp/NRDK/baseline_s2/result_gt_seg', type=str, help='')
    parser.add_argument('--rel_dir_fitted', type=str, default='cylinder_fitting_mcwei_20250929', help='')
    args = parser.parse_args()
    args.dir_fitted = os.path.join(args.dir_root, args.rel_dir_fitted)
    args.dir_save_for_vis = os.path.join(args.dir_root, args.rel_dir_fitted, 'save_for_vis')
    return args


# --------------------------------
# 几何/筛选工具
# --------------------------------
def point_to_line_distance_v2(pts, A, B):
    """
    :param pts: (N,3)
    :param A: (3,)
    :param B: (3,)
    :return: (N,)
    """
    A = np.array(A)
    B = np.array(B)
    AB = B - A
    if np.linalg.norm(AB) < 1e-10:
        return np.linalg.norm(pts - A, axis=1)
    AP = pts - A
    cross_product = np.cross(AP, AB)
    cross_norm = np.linalg.norm(cross_product, axis=1)
    dis = cross_norm / np.linalg.norm(AB)
    return dis


def get_pc_mask_inside_cylinder(pts, start, end, radius, len_ext=0.02):
    """
    :param pts: (N,3)
    :param start,end: (3,)
    :param radius: float
    :param len_ext: float
    :return: mask_inside (N,) bool
    """
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


# --------------------------------
# 线段风格圆柱体（保持原样）
# --------------------------------
def create_cylinder_line(start, end, radius=0.001, color=[1, 0, 0]):
    """
    用细圆柱体画一条线段（视觉上更清晰）
    """
    direction = np.subtract(end, start)
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None

    direction_unit = direction / length

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length,
        resolution=4
    )

    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction_unit)
    rotation_angle = np.arccos(np.dot(z_axis, direction_unit))

    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        cylinder.rotate(rotation_matrix, center=(0, 0, 0))

    cylinder.translate(np.mean([start, end], axis=0))
    cylinder.paint_uniform_color(color)
    return cylinder


def add_cylinder(vis, cyl, color, transparent=False, line_radius=0.002):
    """
    使用圆柱体线段画“柱体轮廓”（原逻辑保持不变）
    """
    if isinstance(cyl, dict):
        start = np.array(cyl['start'])
        end = np.array(cyl['end'])
        radius = cyl['radius']
    elif isinstance(cyl, list):
        start = np.array(cyl[0])
        end = np.array(cyl[1])
        radius = cyl[2]
    else:
        raise TypeError(type(cyl))

    v = end - start
    height = np.linalg.norm(v)
    if height < 1e-6:
        print(f"Cylinder has zero height, skipped.")
        return
    v_unit = v / height

    # 1) 中心线
    center_line = create_cylinder_line(start, end, radius=line_radius, color=color)
    if center_line is not None:
        vis.add_geometry(center_line)

    # 2) 两个正交基向量
    if abs(v_unit[0]) > 0.5:
        u = np.cross(v_unit, [0, 1, 0])
    else:
        u = np.cross(v_unit, [1, 0, 0])
    u = u / np.linalg.norm(u)
    w = np.cross(v_unit, u)
    w = w / np.linalg.norm(w)

    # 3) 顶/底圆
    num_segments = 16
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    top_points = []
    bottom_points = []
    for angle in angles:
        top_points.append(start + radius * (np.cos(angle) * u + np.sin(angle) * w))
        bottom_points.append(end + radius * (np.cos(angle) * u + np.sin(angle) * w))

    for i in range(num_segments):
        top_edge = create_cylinder_line(top_points[i], top_points[(i + 1) % num_segments], radius=line_radius, color=color)
        if top_edge is not None:
            vis.add_geometry(top_edge)
        bottom_edge = create_cylinder_line(bottom_points[i], bottom_points[(i + 1) % num_segments], radius=line_radius, color=color)
        if bottom_edge is not None:
            vis.add_geometry(bottom_edge)

    # 4) 母线（隔一个画一条）
    for i in range(0, num_segments, 2):
        edge = create_cylinder_line(top_points[i], bottom_points[i], radius=line_radius, color=color)
        if edge is not None:
            vis.add_geometry(edge)
    return


# --------------------------------
# 新增：标签（优先 add_3d_label，失败则小球兜底）
# --------------------------------
def add_label_with_fallback(vis, position, text, color):
    """
    优先使用 vis.add_3d_label()；如果该 API 不可用（或 headless），用彩色小球代替。
    """
    pos = np.asarray(position, float).reshape(3,)
    try:
        vis.add_3d_label(pos, str(text))
    except Exception:
        # fallback: 小球
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        vis.add_geometry(sphere)
        print(f"[fallback] label -> sphere at {pos.tolist()} text={text}")


# --------------------------------
# 可视化回调：保持原样 + 在此处加标签
# --------------------------------
def vis_custom(data, vis):
    """
    data = [pcd_dict, lst_param]
    pcd_dict 需要包含：
        - 'pts': Nx3
        - 'id' : N   （用于分颜色）
    lst_param：管道参数列表，每个元素至少含 'start','end','radius'，若含 'id' 则用于标签。
    """
    pcd, lst_param = data
    pts = pcd['pts']
    id_arr = pcd['id']

    # 点云（按 id 分色）
    for id_ins in np.unique(id_arr[:, ]):
        i_color = int(id_ins)
        color = LST_COLOR[i_color + 1]
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts[id_arr == id_ins])
        p.paint_uniform_color(color)
        vis.add_geometry(p)

    # 柱体 + 标签
    for param in lst_param:
        if param is None:
            continue
        if 'radius' in param:
            color = LST_COLOR[param.get('id', 1) + 1]
            add_cylinder(vis, param, color=color)

            # —— 新增：在柱体旁显示 ID —— #
            if 'id' in param:
                start = np.asarray(param['start'], float)
                end = np.asarray(param['end'], float)
                radius = float(param['radius'])
                center = (start + end) / 2.0
                axis = end - start
                n = np.linalg.norm(axis)
                axis_dir = axis / n if n > 1e-6 else np.array([0, 0, 1.0])
                pos = center + axis_dir * (radius + 0.08)  # 沿轴向外移，避免遮挡
                add_label_with_fallback(vis, pos, f"ID:{param['id']}", color)
    return


# --------------------------------
# 两组对比可视化（保留原交互）
# --------------------------------
def vis_compare_2_pc(func_vis, lst_data1, lst_data2, path_save_base, custom_keys_for_save=('A', 'S', 'D'),
                     full_screen=False, show_ids=False):
    """
    Press 'T' to switch between two data.
    Press 'S' to save visualization results of two data as PNGs.
    Press 'Esc' to close current visualization and check next block.
    """

    def toggle_cloud(vis):
        current_view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        global show_1st
        show_1st = not show_1st
        vis.clear_geometries()
        if show_1st:
            func_vis(lst_data1, vis)
        else:
            func_vis(lst_data2, vis)
        vis.get_view_control().convert_from_pinhole_camera_parameters(current_view)
        vis.update_renderer()
        return True

    def custom_save_screenshot(k):
        def callback(vis):
            global show_1st
            suffix = '_correct' if show_1st else '_origin'
            path_save = path_save_base.replace('.png', '') + f'{suffix}.png'
            path_save = os.path.join(os.path.dirname(path_save), f'{k}_{os.path.basename(path_save)}')
            os.makedirs(os.path.dirname(path_save), exist_ok=True)
            vis.capture_screen_image(path_save)

            if lst_data2 is None:
                return True

            current_view = vis.get_view_control().convert_to_pinhole_camera_parameters()
            vis.clear_geometries()
            show_1st = not show_1st
            if show_1st:
                func_vis(lst_data1, vis)
            else:
                func_vis(lst_data2, vis)
            vis.get_view_control().convert_from_pinhole_camera_parameters(current_view)
            vis.poll_events()
            vis.update_renderer()

            suffix = '_correct' if suffix == '_origin' else '_origin'
            path_save2 = path_save_base.replace('.png', '') + f'{suffix}.png'
            path_save2 = os.path.join(os.path.dirname(path_save2), f'{k}_{os.path.basename(path_save2)}')
            vis.capture_screen_image(path_save2)
            vis.destroy_window()
            return True
        return callback

    try:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        if full_screen:
            vis.create_window(window_name=os.path.basename(path_save_base).replace('.png', ''))
        else:
            vis.create_window(width=800, height=800, window_name=os.path.basename(path_save_base).replace('.png', ''))

        os.makedirs(os.path.dirname(path_save_base), exist_ok=True)

        global show_1st
        show_1st = True
        func_vis(lst_data1, vis)
        if lst_data2 is not None:
            vis.register_key_callback(ord("T"), toggle_cloud)
        for key in custom_keys_for_save:
            vis.register_key_callback(ord(key), custom_save_screenshot(key))
        print("按 T 键切换，A/S/D 保存两组截图，Esc 关闭窗口")
        vis.run()
    except Exception as e:
        print(e)


# --------------------------------
# 单个 block 的数据组织与可视化调用（修复点在这里）
# --------------------------------
def vis_single_block(path_pc, args):
    dir_save_for_vis = args.dir_save_for_vis
    dir_vis = dir_save_for_vis.replace('save_for_vis', 'vis')
    name = os.path.basename(path_pc).replace('pkl', '')

    # 读取拟合与评估结果
    res_fitting = pickle.load(open(os.path.join(dir_save_for_vis, f'{os.path.basename(path_pc)}.pkl'), 'rb'))
    res_eval = pickle.load(open(os.path.join(dir_save_for_vis, f'eval_{os.path.basename(path_pc)}.pkl'), 'rb'))

    gt = res_eval['gt']
    lst_idx_pred_unmatched = res_eval.get('up', [])
    lst_idx_gt_unmatched = res_eval.get('ug', [])
    lst_pair = res_eval.get('m', [])

    recall = (len(gt) - len(lst_idx_gt_unmatched)) / (len(gt) + 1e-6)
    print(f'{path_pc}: gt = {len(gt)}, fn = {len(lst_idx_gt_unmatched)}, recall = {round(recall, 3)}')

    # 没有 FN 不可视化
    if not lst_idx_gt_unmatched:
        return

    # 加载原始点云 (n,17)
    pc = np.loadtxt(path_pc)

    # 下采样点 & 仅管道点
    ind_pc = res_fitting['ind_pts']  # indices into pc
    pc = pc[ind_pc]
    pc = pc[pc[:, 10] == 1]  # cls==1 为管道

    # 找到所有 FN 管道周围点
    mask_pc_fn = np.zeros_like(pc[:, 0], dtype=bool)
    lst_idx_gt_unmatched_valid = []
    for idx_gt in lst_idx_gt_unmatched:
        cyl = res_eval['gt'][idx_gt]
        mask_i = get_pc_mask_inside_cylinder(pc[:, :3], cyl['start'], cyl['end'], cyl['radius'] + 0.02, len_ext=0.02)
        if int(mask_i.sum()) > THR_NUM_PTS_VALID:
            lst_idx_gt_unmatched_valid.append(idx_gt)
        mask_pc_fn |= mask_i

    if not lst_idx_gt_unmatched_valid:
        return

    # 组装 “FN 的 GT 柱体” 列表，并补上 id（= gt 索引）
    lst_gt_fn = []
    for idx_gt in lst_idx_gt_unmatched_valid:
        cyl = res_eval['gt'][idx_gt].copy()
        cyl['id'] = idx_gt
        lst_gt_fn.append(cyl)

    # 预测集合中过滤出“与 FN 点相交”的柱体
    pc_fitted = copy.deepcopy(pc)
    pc_fitted[:, 11] = -1  # cluster id
    pc_fitted[:, 12] = -1  # inlier id

    lst_pred = []
    ind_pc_fn = np.nonzero(mask_pc_fn)[0]
    for i in range(len(res_fitting['results'])):
        res_i = res_fitting['results'][i]
        ind_clustered = res_i['ind_clustered']
        ind_inlier = res_i['ind_inlier']
        assert len(np.setdiff1d(ind_inlier, ind_clustered)) == 0
        d = res_i['d']
        start = res_i['start']
        end = res_i['end']
        radius = res_i['radius']
        length = res_i['length']
        pc_fitted[ind_clustered, 11] = i + 1
        pc_fitted[ind_inlier, 12] = i
        pc[:, 6:9] = d
        if len(np.intersect1d(ind_pc_fn, ind_inlier)):  # 与 FN 点相交
            lst_pred.append(dict(start=start, end=end, radius=radius, id=i + 1))

    # ====== 关键修复：把点云打包为 dict，而不是直接传 ndarray ======
    # 输入：FN 周边点
    pc_fn = pc[mask_pc_fn]
    pcd = dict(
        pts=pc_fn[:, :3],
        id=pc_fn[:, 10],        # 语义（1=pipe），仅用于分色
        direct=pc_fn[:, 6:9],   # 可选
        normals=pc_fn[:, 3:6],  # 可选
    )
    # 输出：FN 周边的聚类/内点标识
    pc_fitted_fn = pc_fitted[mask_pc_fn]
    pcd_fitted = dict(
        pts=pc_fitted_fn[:, :3],
        id=pc_fitted_fn[:, 11],         # cluster id
        id_inlier=pc_fitted_fn[:, 12],  # inlier id
    )

    # 可视化对比：第一组（预测柱 + 聚类点），第二组（FN GT 柱 + FN 周边点）
    os.makedirs(os.path.join(args.dir_root, 'vis'), exist_ok=True)
    path_vis = os.path.join(args.dir_root, 'vis', f'{os.path.basename(path_pc)}.png')

    vis_compare_2_pc(
        vis_custom,
        [pcd_fitted, lst_pred],
        [pcd,        lst_gt_fn],
        path_vis,
        show_ids=True
    )
    return


# --------------------------------
# 主入口
# --------------------------------
def main(args):
    lst_path_pc = glob.glob(os.path.join(args.dir_root, "*.pts"))
    for i, path_pc in enumerate(lst_path_pc):
        print(f'visualize: {i}')
        vis_single_block(path_pc, args)
    return


if __name__ == '__main__':
    main(parse_args())
