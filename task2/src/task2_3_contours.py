# -*- coding: utf-8 -*-
"""task2-3：检测 example1.jpg 里**飞机**的轮廓并保存，计算它的总面积与总周长。

example1.jpg 是真实照片：**天空背景 + 一架飞机 + 若干云**。直接做 Otsu/阈值会把云也当成目标，
所以这里分两步：

第一步（分割）：把"暗色主体"和"贴着主体的白色高光"合起来
    · 灰度 + 高斯模糊 → Otsu 反相阈值 → 得到暗色主体（飞机机身、机翼、尾翼）；
    · 只保留与主体**连通/紧邻**的低饱和高亮像素（机身白色涂装），而**不去全局**抓高亮，
      这样远处的云就不会被并进来（这是上一版的错误）；
    · 闭运算补孔 + 开运算去噪 → 得到一个干净的前景掩码。

第二步（筛选）：用可解释的判据把云剔掉，只留飞机
    对每个候选轮廓计算 4 个量：
      · 面积占比     占比过小的是噪点；
      · **边缘密度** 区域内 Canny 边缘像素占比 —— 飞机有机身线条/机翼边缘（约 12%），云是光滑的（0%）；
      · **亮度**     区域平均亮度 V / 灰度 —— 云很亮（V≈200），飞机偏暗（V≈128）；
      · 实心度/长宽比 作为辅助说明。
    判定规则：
      · 边缘密度 ≥ 3%  且  平均亮度 V ≤ 170  → **飞机**
      · 边缘密度 < 2%  且  平均亮度 V > 180  → **云**（排除）
      · 其余 → 低置信候选（排除，但仍记录到 json 里）

输出：
    outputs/images/task2_3_contours_annotated.png   只画飞机轮廓的图像
    outputs/images/task2_3_mask.png                 分割掩码
    outputs/images/task2_3_rejected.png             被排除的候选（云等）示意
    outputs/figures/fig_task2_3_contours.png        流程 / 判据 / 统计
    outputs/data/task2_3_contours.json              飞机轮廓的总面积/总周长 + 每个候选的判据
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, FIG_DIR, IMG_DIR, dump_json, ensure_dirs, imwrite,
    resolve_asset, save_fig, setup_matplotlib, show_panel,
)

MIN_AREA_RATIO = 0.0005      # 面积阈值：图像面积的 0.05%
GROW_RATIO = 0.02            # 只补"紧邻主体"的白色像素：膨胀半径 = 图像对角线的 2%
EDGE_MIN_PCT = 3.0           # 判为"有结构的目标"的边缘密度下限（%）
V_MAX_AIRCRAFT = 170.0       # 飞机区域平均亮度上限（云通常 > 190）
V_MIN_CLOUD = 180.0          # 云的亮度下限
EDGE_MAX_CLOUD = 2.0         # 云的边缘密度上限（%）
COLOR_PLANE = (60, 76, 231)
COLOR_CLOUD = (150, 150, 150)


# ---------------------------------------------------------------- 分割
def build_mask(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ① 暗色主体（飞机机身、机翼、尾翼；云很亮，不会被选中）
    thr, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ② 低饱和高亮（机身白色涂装）—— 全局抓会连云一起抓进来，所以做两版：
    #    raw  : 全局低饱和高亮（候选更全，云也在里面 → 交给第二步的判据筛）
    #    clean: 只取"紧邻主体"的那部分（更严的分割，作为对照展示）
    low_sat = ((hsv[..., 1] < 70) & (hsv[..., 2] > 185)).astype(np.uint8) * 255
    h, w = gray.shape
    grow_r = max(5, int(GROW_RATIO * np.hypot(h, w)))
    seed = cv2.dilate(dark, np.ones((grow_r | 1, grow_r | 1), np.uint8), iterations=1)
    body_white = cv2.bitwise_and(seed, low_sat)

    def merge(a, b):
        m = cv2.bitwise_or(a, b)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    return dict(gray=gray, hsv=hsv, mask=merge(dark, low_sat), mask_clean=merge(dark, body_white),
                dark=dark, low_sat=low_sat, body_white=body_white,
                otsu=float(thr), edges=cv2.Canny(gray, 50, 150), grow_r=grow_r)


# ---------------------------------------------------------------- 候选 + 判据
def candidates(img: np.ndarray, res: dict) -> list[dict]:
    gray, hsv, edges = res["gray"], res["hsv"], res["edges"]
    h, w = gray.shape
    min_area = MIN_AREA_RATIO * h * w

    cnts, _ = cv2.findContours(res["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    cnts.sort(key=lambda c: -cv2.contourArea(c))

    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        x, y, bw, bh = cv2.boundingRect(c)
        region = np.zeros_like(gray)
        cv2.drawContours(region, [c], -1, 255, -1)
        inside = region > 0
        v_mean = float(hsv[..., 2][inside].mean())
        s_mean = float(hsv[..., 1][inside].mean())
        g_mean = float(gray[inside].mean())
        edge_pct = float((edges[inside] > 0).mean() * 100)
        hull = cv2.convexHull(c)
        solidity = float(area / max(cv2.contourArea(hull), 1e-6))

        if edge_pct >= EDGE_MIN_PCT and v_mean <= V_MAX_AIRCRAFT:
            kind, keep = "飞机", True
        elif edge_pct < EDGE_MAX_CLOUD and v_mean > V_MIN_CLOUD:
            kind, keep = "云", False
        else:
            kind, keep = "低置信候选", False

        approx = cv2.approxPolyDP(c, 0.005 * perim, True)
        out.append({
            "id": len(out) + 1,
            "type": kind,
            "kept": keep,
            "area_px2": round(area, 1),
            "perimeter_px": round(perim, 1),
            "area_ratio_pct": round(area / (h * w) * 100, 3),
            "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
            "aspect_w_over_h": round(bw / max(bh, 1), 2),
            "mean_gray": round(g_mean, 1),
            "mean_V": round(v_mean, 1),
            "mean_S": round(s_mean, 1),
            "edge_density_pct": round(edge_pct, 2),
            "solidity": round(solidity, 3),
            "area_approx_poly_px2": round(float(cv2.contourArea(approx)), 1),
            "perimeter_approx_poly_px": round(float(cv2.arcLength(approx, True)), 1),
            "vertices_approx": int(len(approx)),
            "contour": c,
        })
    return out


def main() -> None:
    ensure_dirs(FIG_DIR, IMG_DIR, DATA_DIR)
    setup_matplotlib()
    src = resolve_asset("example1.jpg")
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    res = build_mask(img)
    cands = candidates(img, res)

    kept = [c for c in cands if c["kept"]]
    rejected = [c for c in cands if not c["kept"]]

    print(f"图像 {w}×{h}  Otsu 阈值 = {res['otsu']:.0f}  白色补全半径 = {res['grow_r']} px")
    print(f"候选轮廓 {len(cands)} 个：")
    print(f"{'#':>2} {'类型':<8} {'面积px²':>9} {'占比':>7} {'亮度V':>7} {'边缘密度':>8} {'实心度':>6}  判定")
    for c in cands:
        print(f"{c['id']:>2} {c['type']:<8} {c['area_px2']:>9.0f} {c['area_ratio_pct']:>6.2f}% "
              f"{c['mean_V']:>7.1f} {c['edge_density_pct']:>7.2f}% {c['solidity']:>6.2f}  "
              f"{'保留(飞机)' if c['kept'] else '排除'}")

    total_area = sum(c["area_px2"] for c in kept)
    total_perim = sum(c["perimeter_px"] for c in kept)
    total_area_poly = sum(c["area_approx_poly_px2"] for c in kept)
    total_perim_poly = sum(c["perimeter_approx_poly_px"] for c in kept)
    print(f"\n✅ 飞机轮廓 {len(kept)} 个：总面积 = {total_area:.1f} px²（占画面 {total_area/(w*h)*100:.2f}%），"
          f"总周长 = {total_perim:.1f} px")
    print(f"   多边形近似：面积 {total_area_poly:.1f} px²，周长 {total_perim_poly:.1f} px")
    print(f"   已排除 {len(rejected)} 个云/干扰候选：" +
          ", ".join(f"#{c['id']}({c['type']},{c['area_px2']:.0f}px²)" for c in rejected))

    # ---------------- 标注图（只画飞机）
    canvas = img.copy()
    for c in kept:
        cv2.drawContours(canvas, [c["contour"]], -1, COLOR_PLANE, 3, cv2.LINE_AA)
        x, y, bw, bh = c["bbox_xywh"]
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (255, 255, 255), 1, cv2.LINE_AA)
        label = f"AIRCRAFT #{c['id']}  A={c['area_px2']:.0f}px2  P={c['perimeter_px']:.0f}px"
        cv2.putText(canvas, label, (max(4, x), max(16, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (max(4, x), max(16, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PLANE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"aircraft: A={total_area:.0f}px2  P={total_perim:.0f}px", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, f"aircraft: A={total_area:.0f}px2  P={total_perim:.0f}px", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    imwrite(IMG_DIR / "task2_3_contours_annotated.png", canvas)
    imwrite(IMG_DIR / "task2_3_mask.png", res["mask"])

    # 被排除的候选示意
    rej_canvas = img.copy()
    for c in rejected:
        cv2.drawContours(rej_canvas, [c["contour"]], -1, COLOR_CLOUD, 2, cv2.LINE_AA)
        x, y, bw, bh = c["bbox_xywh"]
        txt = f"REJECT #{c['id']} {c['type']} V={c['mean_V']:.0f} edge={c['edge_density_pct']:.1f}%"
        cv2.putText(rej_canvas, txt, (max(4, x), max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(rej_canvas, txt, (max(4, x), max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CLOUD, 1, cv2.LINE_AA)
    imwrite(IMG_DIR / "task2_3_rejected.png", rej_canvas)

    # ---------------- 汇总图
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.28, wspace=0.18)
    show_panel(fig.add_subplot(gs[0, 0]), img, f"原图 example1.jpg（{w}×{h}，天空+飞机+云）")
    show_panel(fig.add_subplot(gs[0, 1]), res["dark"], f"① Otsu 暗色主体（阈值 {res['otsu']:.0f}）")
    show_panel(fig.add_subplot(gs[0, 2]), res["body_white"],
               f"② 只补紧邻机身的白漆（半径 {res['grow_r']}px）")
    show_panel(fig.add_subplot(gs[1, 0]), res["mask"], "③ 候选掩码：暗色 + 全局高亮（云也在里面）")
    show_panel(fig.add_subplot(gs[1, 1]), canvas, "④ 判据筛选后：只保留飞机轮廓")
    show_panel(fig.add_subplot(gs[1, 2]), rej_canvas, f"⑤ 被排除的候选（{len(rejected)} 个云/干扰）")

    ax1 = fig.add_subplot(gs[2, 0])
    labels = [f"#{c['id']} {c['type']}" for c in cands]
    colors = ["#2e7d32" if c["kept"] else "#9e9e9e" for c in cands]
    ax1.bar(labels, [c["edge_density_pct"] for c in cands], color=colors)
    ax1.axhline(EDGE_MIN_PCT, color="red", ls="--", lw=1, label=f"飞机判定线 {EDGE_MIN_PCT}%")
    ax1.set_title("判据①：边缘密度（飞机有结构、云是光滑的）", fontsize=10)
    ax1.set_ylabel("%")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[2, 1])
    ax2.bar(labels, [c["mean_V"] for c in cands], color=colors)
    ax2.axhline(V_MAX_AIRCRAFT, color="red", ls="--", lw=1, label=f"飞机亮度上限 {V_MAX_AIRCRAFT:.0f}")
    ax2.axhline(V_MIN_CLOUD, color="gray", ls=":", lw=1, label=f"云亮度下限 {V_MIN_CLOUD:.0f}")
    ax2.set_title("判据②：区域平均亮度 V", fontsize=10)
    ax2.set_ylabel("V (0-255)")
    ax2.tick_params(axis="x", labelsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 2])
    ax3.axis("off")
    rows = [[f"#{c['id']}", c["type"], f"{c['area_px2']:.0f}", f"{c['perimeter_px']:.0f}",
             f"{c['edge_density_pct']:.1f}%", f"{c['mean_V']:.0f}", "保留" if c["kept"] else "排除"]
            for c in cands]
    tbl = ax3.table(cellText=rows, colLabels=["#", "类型", "面积px²", "周长px", "边缘密度", "亮度V", "结论"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    ax3.set_title("候选判据表", fontsize=10)

    fig.suptitle(f"task2-3 飞机轮廓检测：{len(kept)} 个飞机轮廓，总面积 {total_area:.0f} px²，"
                 f"总周长 {total_perim:.0f} px（已排除 {len(rejected)} 个云/干扰）", fontsize=13)
    save_fig(fig, "fig_task2_3_contours.png")

    dump_json({
        "file": src.name,
        "image_size": [w, h],
        "target": "飞机（飞机以外的云/干扰候选已排除）",
        "otsu_threshold": res["otsu"],
        "body_white_grow_radius_px": res["grow_r"],
        "criteria": {
            "min_area_ratio": MIN_AREA_RATIO,
            "aircraft": f"edge_density >= {EDGE_MIN_PCT}% 且 mean_V <= {V_MAX_AIRCRAFT}",
            "cloud": f"edge_density < {EDGE_MAX_CLOUD}% 且 mean_V > {V_MIN_CLOUD}",
        },
        "n_contours": len(kept),
        "total_area_px2": round(total_area, 1),
        "total_perimeter_px": round(total_perim, 1),
        "total_area_ratio_pct": round(total_area / (w * h) * 100, 3),
        "total_area_approx_poly_px2": round(total_area_poly, 1),
        "total_perimeter_approx_poly_px": round(total_perim_poly, 1),
        "n_rejected": len(rejected),
        "rejected_total_area_px2": round(sum(c["area_px2"] for c in rejected), 1),
        "aircraft_contours": [{k: v for k, v in c.items() if k != "contour"} for c in kept],
        "all_candidates": [{k: v for k, v in c.items() if k != "contour"} for c in cands],
    }, DATA_DIR / "task2_3_contours.json")


if __name__ == "__main__":
    main()
