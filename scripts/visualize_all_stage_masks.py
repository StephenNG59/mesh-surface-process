"""
对于mesh，可视化gt、stage1、stage2得到的masks
以.txt形式记录，每个顶点的label
正确预测的较透明，错误预测的颜色较明显
"""
import os
import numpy as np
from vedo import Mesh, Plotter, settings


# 设置路径
gt_dir = R'E:\wyc\Data\teeth3ds+\new\lower_preprocessed\npz(arap)'
stage1_dir = R'E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_16cls_lower_yolo11n\predict_labels(per_vertex)'
# stage2_dir = R'E:\wyc\Data\teeth3ds+\new\640NKM\final_predictions\640px,bnd_size20,bnd_wt10,non_bnd_wt1,bs4,s1.0,cls1,cha7'
# output_dir = R'E:\wyc\Data\teeth3ds+\new\640NKM\all_stages_visualize'
stage2_dir = R"E:\wyc\Data\teeth3ds+\new\single_tooth_meshes\lower\test\final_prediction_output(teegraph_lower)"
output_dir = R'E:\wyc\Data\teeth3ds+\new\single_tooth_meshes\lower\test\all_stages_visualize'
teeth_classes = 16

# 启用交互（如果交互的话，按Q退出交互会自动保存截图，按ESC退出则不会）
enable_interactive = False


PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
    ]

def map_gt_labels(gt_labels):
    # 创建一个映射字典
    label_mapping = {}

    if teeth_classes == 8:
        # 将11~18映射到1~8
        for i in range(11, 19):
            label_mapping[i] = i - 10
        # 将21~28映射到1~8
        for i in range(21, 29):
            label_mapping[i] = i - 20
        # 将31~38映射到1~8
        for i in range(31, 39):
            label_mapping[i] = i - 30
        # 将41~48映射到1~8
        for i in range(41, 49):
            label_mapping[i] = i - 40
    elif teeth_classes == 16:
        # 将11~18,21~28映射到1~16
        for i in range(11, 19):
            label_mapping[i] = i - 10
        for i in range(21, 29):
            label_mapping[i] = i - 12
        # 将31~38,41~48映射到1~16
        for i in range(31, 39):
            label_mapping[i] = i - 30
        for i in range(41, 49):
            label_mapping[i] = i - 32

    # 将0映射为0
    label_mapping[0] = 0

    # 对每个GT标签应用映射
    mapped_labels = [label_mapping.get(label, label) for label in gt_labels]
    
    return np.array(mapped_labels)


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    # 设置 vedo 全局参数
    settings.use_depth_peeling = True  # 防止透明叠加出错
    settings.use_parallel_projection = True

    # 遍历所有 GT .npz 文件
    for filename in os.listdir(stage2_dir)[:]:
        if not filename.endswith('.npy'):
            continue

        name = filename[:-4]
        gt_path = os.path.join(gt_dir, name + '.npz')
        stage1_path = os.path.join(stage1_dir, name + '.txt')
        stage2_path = os.path.join(stage2_dir, filename)

        # 读取GT mesh 和 labels
        if not os.path.exists(gt_path):
            print(f"Missing: {gt_path}")
            continue
        data = np.load(gt_path)
        V = data['V']
        F = data['F']
        gt_labels = data['labels'].astype(int)
        gt_labels = map_gt_labels(gt_labels)

        # 读取 Stage1 labels
        if not os.path.exists(stage1_path):
            print(f"Missing: {stage1_path}")
            continue
        stage1_labels = np.loadtxt(stage1_path, dtype=int)

        # 读取 Stage2 labels
        if not os.path.exists(stage2_path):
            print(f"Missing: {stage2_path}")
            continue
        stage2_labels = (np.load(stage2_path)[:, 0] + 1).astype(int)  # 结果以-1为起始

        # 创建 vedo Mesh
        mesh_gt = Mesh([V, F])#.cmap('jet', gt_labels)
        mesh_s1 = Mesh([V, F])#.cmap('jet', stage1_labels)
        mesh_s2 = Mesh([V, F])#.cmap('jet', stage2_labels)

        # 创建透明度数组（255 = 不透明，50 = 半透明）
        alpha_s1 = np.where(stage1_labels == gt_labels, 100, 255)
        alpha_s2 = np.where(stage2_labels == gt_labels, 100, 255)

        # 上色
        palette = np.asarray(PALETTE)
        colors_s1 = np.concatenate([palette[stage1_labels], alpha_s1.reshape(-1, 1)], axis=1)
        colors_s2 = np.concatenate([palette[stage2_labels], alpha_s2.reshape(-1, 1)], axis=1)
        mesh_gt.pointcolors = palette[gt_labels]
        mesh_s1.pointcolors = colors_s1
        mesh_s2.pointcolors = colors_s2

        # 设置位置
        # x_size = V[:, 0].max() - V[:, 0].min()
        # mesh_s1.pos(x_size, 0, 0)
        # mesh_s2.pos(2*x_size, 0, 0)

        # 渲染并保存图片
        vp = Plotter(N=3, size=(1920, 640), title=name, offscreen=not enable_interactive)
        vp.at(0).show(mesh_gt, 'gt', zoom='tight', axes=1)
        vp.at(1).show(mesh_s2, 'stage2', zoom='tight', axes=1)
        vp.at(2).show(mesh_s1, 'stage1', zoom='tight', axes=1)
        if enable_interactive:
            vp.camera.SetFocalPoint(mesh_gt.center_of_mass())
            vp.interactive()
        # vp = Plotter(offscreen=not enable_interactive, size=(1200, 400), title=name)
        # vp.show(mesh_gt, mesh_s1, mesh_s2, axes=1, zoom='tight', interactive=enable_interactive)
        out_path = os.path.join(output_dir, name + '_vis.png')
        vp.screenshot(out_path)
        
        if not enable_interactive:
            vp.close()

        print(f"Saved: {out_path}")
