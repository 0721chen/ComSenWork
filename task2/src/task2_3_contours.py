# -*- coding: utf-8 -*-
"""task2-3：用 OpenCV 检测 example1.jpg 的轮廓并保存，计算轮廓的总面积与总周长。

example1.jpg 是一张**真实照片**（不是白底图形），所以不能用"傻瓜式 Otsu 一遍"了事：

1. 灰度 + 高斯模糊后做 Otsu 自动阈值，取"暗色主体"分支（自然照片里主体通常比背景暗）；
2. 再补一个"低饱和 + 高亮"分支（白色羽毛/反光区域饱和度低、亮度高），
   把被高光切成两段的主体补回来 —— 只用 Otsu 会漏掉一大块；
3. 两分支取并集 → 闭运算填孔 + 开运算去噪；
4. `findContours(RETR_EXTERNAL)` 取外轮廓，按面积过滤（图像面积的 0.05%）；
5. 绘制轮廓（不同颜色 + 编号 + 外接框 + 面积/周长标签）；
6. 统计总面积、总周长；再用 `RETR_TREE` 把孔洞扣除得到"真实前景面积"。

输出：
    outputs/images/task2_3_contours_annotated.png   绘制了轮廓的图像
    outputs/images/task2_3_mask.png                 分割掩码
    outputs/images/task2_3_hierarchy.png            含孔洞的层级轮廓
    outputs/figures/fig_task2_3_contours.png        流程与统计图
    outputs/data/task2_3_contours.json              总面积/总周长等数值
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, FIG_DIR, IMG_DIR, dump_json, ensure_dirs, imwrite,
    resolve_asset, save_fig, setup_matplotlib, show_panel,
)

MIN_AREA_RATIO = 0.0005      # 面积阈值：图像面积的 0.05%（1036×584 ≈ 302 px²）
COLORS = [(60, 76, 231), (74, 195, 139), (219, 152, 52), (60, 210, 235),
          (182, 89, 155), (34, 126, 230), (94, 189, 168), (160, 160, 160)]


def build_mask(img: np.ndarray):
    """返回 (灰度图, 合并掩码, Otsu 阈值, 暗色分支, 低饱和高亮分支)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 分支 1：暗色主体
    thr, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 分支 2：低饱和 + 高亮（白色羽区、反光、云）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_sat = ((hsv[..., 1] < 70) & (hsv[..., 2] > 185)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(dark, low_sat)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return gray, mask, float(thr), dark, low_sat


def analyze(img: np.ndarray) -> dict:
    gray, mask, otsu_thr, dark, low_sat = build_mask(img)
    h, w = gray.shape
    min_area = MIN_AREA_RATIO * h * w

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=lambda c: -cv2.contourArea(c))

    items = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.005 * perim, True)
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] else 0
        x, y, bw, bh = cv2.boundingRect(c)
        items.append({
            "id": i + 1,
            "area_px2": round(float(area), 1),
            "perimeter_px": round(float(perim), 1),
            "area_approx_poly_px2": round(float(cv2.contourArea(approx)), 1),
            "perimeter_approx_poly_px": round(float(cv2.arcLength(approx, True)), 1),
            "vertices_approx": int(len(approx)),
            "centroid": [cx, cy],
            "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
            "circularity_4piA_P2": round(float(4 * np.pi * area / (perim ** 2)) if perim else 0.0, 3),
            "area_ratio_pct": round(float(area / (h * w) * 100), 3),
        })

    # 带孔洞的净面积（RETR_TREE + 层级奇偶）
    tree, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []
    net = 0.0
    for idx, c in enumerate(tree):
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        depth, p = 0, hierarchy[idx][3]
        while p != -1:
            depth += 1
            p = hierarchy[p][3]
        net += a if depth % 2 == 0 else -a

    return dict(gray=gray, mask=mask, otsu=otsu_thr, dark=dark, low_sat=low_sat,
                contours=contours, items=items, net_area=net, min_area=min_area)


def main() -> None:
    ensure_dirs(FIG_DIR, IMG_DIR, DATA_DIR)
    setup_matplotlib()
    src = resolve_asset("example1.jpg")
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    res = analyze(img)
    items, contours = res["items"], res["contours"]

    total_area = sum(it["area_px2"] for it in items)
    total_perim = sum(it["perimeter_px"] for it in items)
    total_area_poly = sum(it["area_approx_poly_px2"] for it in items)
    total_perim_poly = sum(it["perimeter_approx_poly_px"] for it in items)

    print(f"图像 {w}×{h}  Otsu 阈值 = {res['otsu']:.0f}  面积阈值 = {res['min_area']:.0f} px²")
    print(f"检测到轮廓 {len(items)} 个")
    print(f"总面积 = {total_area:.1f} px²，总周长 = {total_perim:.1f} px")
    print(f"(多边形近似) 总面积 = {total_area_poly:.1f} px²，总周长 = {total_perim_poly:.1f} px")
    print(f"(扣除孔洞) 真实前景面积 = {res['net_area']:.1f} px²")

    # ---------------- 绘制标注图
    canvas = img.copy()
    for it, c in zip(items, contours):
        color = COLORS[(it["id"] - 1) % len(COLORS)]
        cv2.drawContours(canvas, [c], -1, color, 3, cv2.LINE_AA)
        x, y, bw, bh = it["bbox_xywh"]
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (255, 255, 255), 1, cv2.LINE_AA)
        label = f"#{it['id']} A={it['area_px2']:.0f}px2 P={it['perimeter_px']:.0f}px"
        ty = max(16, y - 8)
        cv2.putText(canvas, label, (max(4, x), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (max(4, x), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    tree_canvas = img.copy()
    tree, _ = cv2.findContours(res["mask"], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tree_canvas, tree, -1, (0, 0, 255), 2, cv2.LINE_AA)

    imwrite(IMG_DIR / "task2_3_contours_annotated.png", canvas)
    imwrite(IMG_DIR / "task2_3_mask.png", res["mask"])
    imwrite(IMG_DIR / "task2_3_hierarchy.png", tree_canvas)

    # ---------------- 汇总图
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85], hspace=0.25, wspace=0.18)
    show_panel(fig.add_subplot(gs[0, 0]), img, f"原图 example1.jpg ({w}×{h})")
    show_panel(fig.add_subplot(gs[0, 1]), res["dark"], f"分支1: Otsu 暗色主体 (阈值 {res['otsu']:.0f})")
    show_panel(fig.add_subplot(gs[0, 2]), res["low_sat"], "分支2: 低饱和高亮区(白色/反光)")
    show_panel(fig.add_subplot(gs[1, 0]), res["mask"], "两分支并集 + 闭/开运算")
    show_panel(fig.add_subplot(gs[1, 1:]), canvas, "轮廓标注（编号 / 面积 / 周长 / 外接框）")

    ax1 = fig.add_subplot(gs[2, 0])
    ids = [f"#{it['id']}" for it in items]
    ax1.bar(ids, [it["area_px2"] for it in items], color="tab:blue", alpha=0.85)
    ax1.set_title(f"各轮廓面积（总面积 {total_area:.0f} px²）", fontsize=10)
    ax1.set_ylabel("px²")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[2, 1])
    xs = np.arange(len(items))
    ax2.bar(xs - 0.2, [it["perimeter_px"] for it in items], 0.38, label="arcLength(原始)", color="tab:orange")
    ax2.bar(xs + 0.2, [it["perimeter_approx_poly_px"] for it in items], 0.38,
            label="多边形近似", color="tab:green", alpha=0.85)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(ids, fontsize=8)
    ax2.set_title(f"各轮廓周长（总周长 {total_perim:.0f} px）", fontsize=10)
    ax2.set_ylabel("px")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 2])
    ax3.axis("off")
    rows = [[it["id"], f"{it['area_px2']:.0f}", f"{it['area_ratio_pct']:.2f}%",
             f"{it['perimeter_px']:.0f}", f"{it['circularity_4piA_P2']:.2f}",
             f"{it['vertices_approx']}"] for it in items[:8]]
    if rows:
        tbl = ax3.table(cellText=rows, colLabels=["#", "面积px²", "占比", "周长px", "圆度", "近似顶点"],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.3)
    ax3.set_title("轮廓属性表（前 8 个）", fontsize=10)

    fig.suptitle(f"task2-3 轮廓检测：{len(items)} 个轮廓，总面积 {total_area:.0f} px²，"
                 f"总周长 {total_perim:.0f} px", fontsize=13)
    save_fig(fig, "fig_task2_3_contours.png")

    dump_json({
        "file": src.name,
        "image_size": [w, h],
        "otsu_threshold": res["otsu"],
        "min_area_px2": round(res["min_area"], 1),
        "n_contours": len(items),
        "total_area_px2": round(total_area, 1),
        "total_perimeter_px": round(total_perim, 1),
        "total_area_ratio_pct": round(total_area / (w * h) * 100, 3),
        "total_area_approx_poly_px2": round(total_area_poly, 1),
        "total_perimeter_approx_poly_px": round(total_perim_poly, 1),
        "net_area_with_holes_px2": round(float(res["net_area"]), 1),
        "image_area_px2": w * h,
        "contours": items,
    }, DATA_DIR / "task2_3_contours.json")


if __name__ == "__main__":
    main()
