# -*- coding: utf-8 -*-
"""生成"可以直接提交"的材料包。

按题目《提交方式》的要求（代码放 GitHub，其余打包发邮箱）生成：

    提交/
    ├── ComSen2026_提交报告.md        合并报告（Markdown，仓库里也能看）
    ├── ComSen2026_提交报告.html      同一份报告的自包含网页（浏览器打开→打印为 PDF）
    ├── 提交说明与模板.md             邮件正文模板 / GitHub Issue 模板 / 检查清单
    ├── 图片/                         按题目要求重命名的所有结果图
    ├── 视频/                         task2 要求提交的三个结果视频
    ├── 模型/                         task4 训练出的 best.pt / best.onnx
    └── ComSen2026_task2-3-4_提交包.zip   上面这些的压缩包（不含代码）

用法：
    python tools/build_submission.py
"""

from __future__ import annotations

import html
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "提交"

TASK2, TASK3, TASK4 = ROOT / "task2", ROOT / "task3", ROOT / "task4_yolo"

# 视频：源文件 → 提交名
VIDEOS = [
    (TASK2 / "outputs/videos/example2_motion_compare.mp4", "task2_题四_example2_帧差法vsMOG2_标注视频.mp4"),
    (TASK2 / "outputs/videos/example3_tracking.mp4", "task2_题五_example3_小球追踪结果.mp4"),
    (TASK2 / "outputs/videos/example4_tracking.mp4", "task2_题五_example4_小球追踪结果.mp4"),
]

# 模型：源文件 → 提交名
MODELS = [
    (TASK4 / "outputs/runs/drone_yolov8n/weights/best.pt", "task4_无人机检测_best.pt"),
    (TASK4 / "outputs/runs/drone_yolov8n/weights/best.onnx", "task4_无人机检测_best.onnx"),
]

# 额外补充的图片（题目点名要的"单独结果图"，报告正文里只是以文字路径提到）
EXTRA_IMAGES = [
    (TASK2 / "outputs/images/task2_3_contours_annotated.png", "任务2_题三_轮廓标注图.png"),
    (TASK2 / "outputs/images/task2_3_mask.png", "任务2_题三_轮廓分割掩码.png"),
    (TASK2 / "outputs/images/channel_B.png", "任务2_题一_通道B.png"),
    (TASK2 / "outputs/images/channel_G.png", "任务2_题一_通道G.png"),
    (TASK2 / "outputs/images/channel_R.png", "任务2_题一_通道R.png"),
    (TASK4 / "outputs/images/pred_coco-bear.jpg", "任务4_题二_推理_coco-bear.jpg"),
    (TASK4 / "outputs/images/pred_coco-bus.jpg", "任务4_题二_推理_coco-bus.jpg"),
    (TASK4 / "outputs/images/pred_coco-person.jpg", "任务4_题二_推理_coco-person.jpg"),
]

IMAGE_PREFIX = {"task2": "任务2", "task3": "任务3", "task4": "任务4"}


# ---------------------------------------------------------------- Markdown → HTML
def md_to_html(md: str) -> str:
    """极简 Markdown 渲染（标题/段落/列表/表格/图片/代码/粗体/行内代码/引用/分割线）。"""
    lines = md.split("\n")
    out, i = [], 0
    in_list, in_ol = False, False

    def close_list():
        nonlocal in_list, in_ol
        if in_list:
            out.append("</ul>")
            in_list = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def inline(t: str) -> str:
        t = html.escape(t)
        t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
                   r'<img alt="\1" src="\2">', t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        return t

    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if s.startswith("```"):
            close_list()
            lang = s[3:].strip()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            out.append(f'<pre class="code"{f" data-lang={lang}" if lang else ""}>' + "\n".join(buf) + "</pre>")
            i += 1
            continue

        if not s:
            close_list()
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            close_list()
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,})$", s):
            close_list()
            out.append("<hr>")
            i += 1
            continue

        if s.startswith(">"):
            close_list()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote>" + "<br>".join(inline(b) for b in buf) + "</blockquote>")
            continue

        if s.startswith("|") and s.count("|") >= 2:
            close_list()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for r in rows if not re.match(r"^[-: ]+$", "".join(r))]
            if body:
                head, rest = body[0], body[1:]
                tbl = ["<table>", "<thead><tr>"]
                tbl += [f"<th>{inline(c)}</th>" for c in head]
                tbl.append("</tr></thead><tbody>")
                for r in rest:
                    tbl.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
                tbl.append("</tbody></table>")
                out.append("".join(tbl))
            continue

        m_ul = re.match(r"^[-*+]\s+(.*)$", s)
        m_ol = re.match(r"^\d+[.)]\s+(.*)$", s)
        if m_ul or m_ol:
            if m_ul and not in_list:
                close_list()
                out.append("<ul>")
                in_list = True
            if m_ol and not in_ol:
                close_list()
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{inline((m_ul or m_ol).group(1))}</li>")
            i += 1
            continue

        close_list()
        out.append(f"<p>{inline(s)}</p>")
        i += 1

    close_list()
    return "\n".join(out)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>ComSen2026 任务 2 / 3 / 4 提交报告</title>
<style>
  body {{ font-family: "Microsoft YaHei", "SimSun", sans-serif; max-width: 1000px; margin: 24px auto;
         padding: 0 24px; line-height: 1.75; color: #222; font-size: 15px; }}
  h1 {{ border-bottom: 3px solid #1f4e79; padding-bottom: 8px; color: #1f4e79; }}
  h2 {{ border-left: 5px solid #1f4e79; padding-left: 10px; margin-top: 32px; color: #1f4e79; }}
  h3 {{ color: #2e6da4; margin-top: 22px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #bbb; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #1f4e79; color: #fff; }}
  tr:nth-child(even) td {{ background: #f5f8fb; }}
  code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-family: Consolas, monospace; }}
  pre.code {{ background: #f7f7f9; border: 1px solid #e1e1e8; padding: 10px; overflow-x: auto;
             font-family: Consolas, monospace; font-size: 13px; }}
  blockquote {{ border-left: 4px solid #ccc; margin: 10px 0; padding: 6px 12px; color: #555;
                background: #fafafa; }}
  hr {{ border: none; border-top: 1px dashed #ccc; margin: 26px 0; }}
  @media print {{ body {{ max-width: none; }} h2 {{ page-break-after: avoid; }}
                  img, table {{ page-break-inside: avoid; }} }}
</style></head><body>
{content}
</body></html>
"""


# ---------------------------------------------------------------- Markdown → Word（.docx）
def md_to_docx(md: str, out_path: Path) -> None:
    """把合并后的 Markdown 转成 Word（标题/段落/列表/表格/图片/代码块）。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "微软雅黑"
    style.element.rPr.rFonts.set(__import__("docx").oxml.ns.qn("w:eastAsia"), "微软雅黑")
    style.font.size = Pt(10.5)

    lines = md.split("\n")
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        # 代码块
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            run = p.add_run("\n".join(buf))
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            p.paragraph_format.left_indent = Inches(0.2)
            continue

        # 表格
        if s.startswith("|") and s.count("|") >= 2:
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for r in rows if not re.match(r"^[-: ]+$", "".join(r))]
            if body:
                ncol = max(len(r) for r in body)
                tbl = doc.add_table(rows=0, cols=ncol)
                tbl.style = "Light Grid Accent 1"
                for ri, r in enumerate(body):
                    cells = tbl.add_row().cells
                    for ci in range(ncol):
                        txt = r[ci] if ci < len(r) else ""
                        txt = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", txt)
                        txt = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", txt)
                        txt = re.sub(r"\*\*([^*]+)\*\*", r"\1", txt)
                        txt = re.sub(r"`([^`]+)`", r"\1", txt)
                        cells[ci].text = txt
                        for para in cells[ci].paragraphs:
                            for run in para.runs:
                                run.font.size = Pt(8)
                                if ri == 0:
                                    run.font.bold = True
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", m.group(2))
            if level == 1:
                doc.add_heading(text, level=0)
            else:
                doc.add_heading(text, level=min(level - 1, 4))
            i += 1
            continue

        # 分割线
        if re.match(r"^(-{3,}|\*{3,})$", s):
            doc.add_paragraph("_" * 60)
            i += 1
            continue

        # 图片
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m_img:
            alt, link = m_img.group(1), m_img.group(2)
            src = (out_path.parent / link)
            if src.exists():
                try:
                    doc.add_picture(str(src), width=Inches(6.2))
                    cap = doc.add_paragraph(alt)
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in cap.runs:
                        run.font.size = Pt(9)
                        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
                except Exception:
                    pass
            i += 1
            continue

        # 引用
        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            text = " ".join(b for b in buf if b)
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
            p = doc.add_paragraph(text)
            p.paragraph_format.left_indent = Inches(0.3)
            for run in p.runs:
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            continue

        # 列表
        m_ul = re.match(r"^[-*+]\s+(.*)$", s)
        m_ol = re.match(r"^\d+[.)]\s+(.*)$", s)
        if m_ul or m_ol:
            text = (m_ul or m_ol).group(1)
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
            text = re.sub(r"`([^`]+)`", r"\1", text)
            if m_ul:
                doc.add_paragraph(text, style="List Bullet")
            else:
                doc.add_paragraph(text, style="List Number")
            i += 1
            continue

        # 普通段落（可能夹着行内图片）
        text = s
        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"〔图：\1〕", text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        doc.add_paragraph(text)
        i += 1

    doc.save(str(out_path))
    print(f"[docx  ] {out_path}")


# ---------------------------------------------------------------- 报告合并
def rewrite_images(md: str, base: Path, prefix: str, img_dir: Path) -> tuple[str, list[Path]]:
    """把报告里的图片链接改写到 图片/ 目录，并返回需要复制的 (源目录, 新名) 列表。"""
    used: list[tuple[Path, str]] = []

    def repl(m):
        alt, link = m.group(1), m.group(2)
        if link.startswith("http"):
            return m.group(0)
        src = (base / link).resolve()
        if not src.exists():
            return m.group(0)
        new_name = f"{prefix}_{src.stem}{src.suffix}"
        if not any(n == new_name for _, n in used):
            used.append((src, new_name))
        return f"![{alt}](图片/{new_name})"

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)
    return md, [u[0] for u in used], [u[1] for u in used]


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    img_dir = OUT / "图片"
    vid_dir = OUT / "视频"
    mdl_dir = OUT / "模型"
    for d in (img_dir, vid_dir, mdl_dir):
        d.mkdir(parents=True, exist_ok=True)

    parts = []
    copied: list[tuple[Path, str]] = []

    def add_report(path: Path, prefix: str) -> None:
        md = path.read_text(encoding="utf-8")
        md, srcs, names = rewrite_images(md, path.parent, prefix, img_dir)
        for src, name in zip(srcs, names):
            dst = img_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied.append((src, name))
        parts.append(md)

    # 封面
    cover = f"""# ComSen2026 算法组招新题 · 提交报告

> **提交内容**：task-2（OpenCV 与 Pillow）、task-3（MLP 识别 MNIST）、task-4 方向1 题1~3（YOLO 目标识别）
> **班别**：＿＿＿＿＿　**姓名**：＿＿＿＿＿　**学号**：＿＿＿＿＿
> **代码仓库**：＿＿＿＿＿＿（GitHub 公开仓库链接）
> **环境**：Windows 11 · Python 3.13.7 · opencv-contrib-python 4.11 · PyTorch 2.14.1 (CPU) · ultralytics 8.4.182

## 本包内容

| 目录 / 文件 | 说明 |
| --- | --- |
| `ComSen2026_提交报告.md` | 本报告的 Markdown 源文件 |
| `ComSen2026_提交报告.html` | 本报告的网页版（浏览器打开 → 打印 → 另存为 PDF） |
| `ComSen2026_提交报告.docx` | 本报告的 Word 版（直接可交） |
| `图片/` | 三份报告用到的全部结果图（按题号命名） |
| `视频/` | task-2 要求提交的三个结果视频（题四 1 个、题五 2 个） |
| `模型/` | task-4 训练得到的 `best.pt` 与导出的 `best.onnx` |
| `提交说明与模板.md` | 邮件正文模板、GitHub Issue 模板、提交前检查清单 |

## 题目要求 → 本包文件 对照表

| 题目要求 | 本包中的材料 |
| --- | --- |
| task-2 题一：打开显示图片、分离三个通道 | `图片/任务2_fig_task2_1_open.png`、`图片/任务2_fig_task2_1_channels.png` |
| task-2 题二：缩放/旋转/平移/翻转 | `图片/任务2_fig_task2_2_geometric.png` |
| task-2 题三：轮廓检测 + 总面积/总周长 | `图片/任务2_题三_轮廓标注图.png`（单独标注图）、`图片/任务2_题三_轮廓分割掩码.png`、`图片/任务2_fig_task2_3_contours.png`（流程与统计）；报告 §题三：**4 个轮廓 / 99 321 px² / 4 161.5 px** |
| task-2 题四：MOG2 vs 不用背景减除（标注视频 + 分析） | **`视频/task2_题四_example2_帧差法vsMOG2_标注视频.mp4`**、`图片/任务2_fig_task2_4_motion_stats.png`、`图片/任务2_fig_task2_4_frames.png` |
| task-2 题五：example3/4 小球追踪（标注视频 + 分析） | **`视频/task2_题五_example3_小球追踪结果.mp4`**、**`视频/task2_题五_example4_小球追踪结果.mp4`**、`图片/任务2_fig_task2_5_tracking_example3.png`、`图片/任务2_fig_task2_5_tracking_example4.png` |
| task-3 题一：深度学习概念理解 | 报告 §题一（概念逐条 + 与实验数字对应） |
| task-3 题二：最小二乘拟合（真实函数/拟合函数图 + 参数 + R²） | `图片/任务3_fig_task3_2_surface.png`、`_slice.png`、`_loss.png`、`_scatter.png`；报告 §题二（参数表、**R²(train)=0.9511 / R²(test)=0.9499**） |
| task-3 题三：MLP 识别 MNIST（Loss / Train Acc / Test Acc + 可视化） | `图片/任务3_fig_task3_3_curves.png`、`_predictions.png`、`_confusion.png`、`_misclassified.png`；报告 §题三（**测试准确率 98.15%**） |
| task-4 题一：学习 YOLO API | `图片/任务4_task4_1_api_cheatsheet.png` + 报告 §题1（接口速查表） |
| task-4 题二：预训练权重推理三张图片 | **`图片/任务4_题二_推理_coco-bear.jpg`**、**`_coco-bus.jpg`**、**`_coco-person.jpg`**（三张标注图）+ `图片/任务4_task4_2_predictions.png`（对比图）；报告 §题2 |
| task-4 题三：无人机数据集训练 + 保存模型与训练结果绘图 | `模型/task4_无人机检测_best.pt`、`模型/task4_无人机检测_best.onnx`、`图片/任务4_task4_3_results.png`、`_confusion_matrix.png`、`_PR_curve.png`、`_F1_curve.png`、`_labels.jpg`、`_summary.png`；报告 §题3（**mAP@0.5 = 0.6844**） |

> 说明：报告正文里出现的 `outputs/figures/...`、`outputs/videos/...` 这类路径，指的是**代码仓库内**的位置；
> 本提交包内的对应文件已按上表放在 `图片/`、`视频/`、`模型/` 目录（文件名见上表与目录）。

---

"""
    parts.append(cover)
    add_report(TASK2 / "REPORT.md", "任务2")
    parts.append("\n\n---\n\n")
    add_report(TASK3 / "REPORT.md", "任务3")
    parts.append("\n\n---\n\n")
    add_report(TASK4 / "REPORT.md", "任务4")

    merged = "\n".join(parts)

    # 去掉各报告里重复的一级标题（合并成一份文档，只在封面保留一个 H1）
    merged = re.sub(r"^# (task-2|task-3|task-4)[^\n]*\n", "", merged, flags=re.M)

    (OUT / "ComSen2026_提交报告.md").write_text(merged, encoding="utf-8")
    print(f"[doc   ] {OUT / 'ComSen2026_提交报告.md'}")

    html_doc = HTML_TEMPLATE.format(content=md_to_html(merged))
    (OUT / "ComSen2026_提交报告.html").write_text(html_doc, encoding="utf-8")
    print(f"[html  ] {OUT / 'ComSen2026_提交报告.html'}")

    try:
        md_to_docx(merged, OUT / "ComSen2026_提交报告.docx")
    except Exception as e:
        print(f"[docx 失败] {type(e).__name__}: {e}")

    # 视频与模型
    for src, name in VIDEOS:
        if src.exists():
            shutil.copy2(src, vid_dir / name)
            print(f"[video ] {vid_dir / name}")
        else:
            print(f"[缺失  ] {src}")
    for src, name in MODELS:
        if src.exists():
            shutil.copy2(src, mdl_dir / name)
            print(f"[model ] {mdl_dir / name}")
        else:
            print(f"[缺失  ] {src}")
    for src, name in EXTRA_IMAGES:
        if src.exists():
            shutil.copy2(src, img_dir / name)
            print(f"[image ] {img_dir / name}")
        else:
            print(f"[缺失  ] {src}")

    # 提交说明与模板
    (OUT / "提交说明与模板.md").write_text(build_guide(copied), encoding="utf-8")
    print(f"[doc   ] {OUT / '提交说明与模板.md'}")

    # 打包（不含代码）
    zip_path = OUT / "ComSen2026_task2-3-4_提交包.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(OUT))
    size = zip_path.stat().st_size / 1024 / 1024
    print(f"[zip   ] {zip_path}  ({size:.1f} MB)")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n提交目录: {OUT}\n图片 {len(copied)} 张 / 视频 {len(VIDEOS)} 个 / 模型 {len(MODELS)} 个 "
          f"/ 总计 {total/1024/1024:.1f} MB")


def build_guide(copied) -> str:
    imgs = "\n".join(f"| `图片/{n}` |" for _, n in copied)
    return f"""# 提交说明（交什么 / 交到哪 / 怎么交）

题目《提交方式》原文要求：

> 请将除代码外文件(如视频,图片,报告)打包发送到 **3813579841@qq.com**，
> 代码请在 github 建立一个 **Public** 的仓库，并按 **班别-姓名-学号-仓库链接** 的格式
> 发送到题目仓库(https://github.com/LunaAmandatio/ComSen2026_Algorithm_Recruitment)的 **Issues** 下。

所以一共三个去处，按下面顺序做即可。

---

# 一、GitHub 仓库（放"代码"）

## 1. 建仓库
在 GitHub 上 New repository → 名称随意（如 `ComSen2026_homework`）→ **Public** → 不要勾选任何初始化文件。

## 2. 上传哪些东西（在项目根目录执行）

```bash
git init
git add .
git commit -m "ComSen2026 task2/task3/task4 题1-3"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

**会被自动排除**（`.gitignore` 已写好，无需手动删）：

| 排除项 | 原因 |
| --- | --- |
| `task4_yolo/dataset/`（236 MB 无人机数据集） | 体积大，`python task4_yolo/src/fetch_dataset.py` 可一键重新下载 |
| `task4_yolo/weights/`（预训练 yolov8n.pt） | 运行时自动下载 |
| `task3/data/*.gz`（MNIST 原始数据） | 首次运行自动下载 |
| `提交/*.zip`（提交包） | 单独发邮件，别塞进仓库 |
| `__pycache__/`、`.tmp/` 等 | 临时文件 |

**必须进仓库的**：`task2/`、`task3/`、`task4_yolo/`（不含 dataset/weights）、`tools/`、`run_all.py`、
`requirements.txt`、`SUBMISSION.md`（当 README）、`task*/REPORT.md`。

> 建议：把 `SUBMISSION.md` 的内容复制到 GitHub 仓库的 `README.md`，别人点进仓库就能看到说明。

---

# 二、邮件（放"除代码外的文件"）

1. 附件：`ComSen2026_task2-3-4_提交包.zip`（约 77 MB；若邮箱限制 20 MB，
   可去掉 `模型/` 目录再压缩，模型在 GitHub 里也有）
2. 收件人：**3813579841@qq.com**
3. 正文模板：

```
主题：ComSen2026 算法组招新题 提交 - ＿＿班-＿＿＿-＿＿＿＿

老师您好：

附件是我完成的题目提交材料（代码已在 GitHub 公开仓库：＿＿＿＿＿＿＿），包含：
1) task-2《OpenCV 与 Pillow 的使用》：实验报告 + 结果图 + 3 个结果视频
   （example2 帧差法/MOG2 对比、example3 与 example4 小球追踪）；
2) task-3《MLP 识别 MNIST》：实验报告 + 拟合曲面/切片/残差图 + Loss/Acc 曲线 + 混淆矩阵；
3) task-4 方向1《基于 YOLO 的目标识别》题1~题3：实验报告 + 三张图片推理结果 +
   无人机数据集训练曲线/混淆矩阵/PR 曲线 + 训练好的模型（best.pt、best.onnx）。

班级：＿＿＿＿　姓名：＿＿＿＿　学号：＿＿＿＿　仓库：＿＿＿＿＿＿＿＿

顺祝工作顺利！
```

---

# 三、题目仓库 Issues（放"报名信息"）

打开 https://github.com/LunaAmandatio/ComSen2026_Algorithm_Recruitment/issues → New issue，
标题与正文都按这个格式写：

```
班别-姓名-学号-仓库链接
计科2301-张三-2023010101-https://github.com/zhangsan/ComSen2026_homework
```

---

# 四、提交前检查清单

- [ ] GitHub 仓库是 **Public**，点开能看到 `task2/src`、`task3/src`、`task4_yolo/src`
- [ ] 仓库首页 README 写清"目录结构 + 如何运行"（直接用 `SUBMISSION.md`）
- [ ] 仓库里**没有** 236 MB 的数据集（确认 `.gitignore` 生效：`git status` 看不到 `task4_yolo/dataset`）
- [ ] 邮件附件是本目录的 `ComSen2026_task2-3-4_提交包.zip`
- [ ] 邮件正文写了 `班别-姓名-学号-仓库链接`
- [ ] Issue 已按同样格式提交
- [ ] 三个视频能播放：`视频/task2_题四_*.mp4`、`视频/task2_题五_*.mp4`
- [ ] 报告能打开：`ComSen2026_提交报告.docx`（Word）或 `.html`（浏览器）
- [ ] 封面已填写 班别/姓名/学号/仓库链接

---

# 五、本目录文件清单

| 文件 / 目录 | 大小 | 说明 |
| --- | --- | --- |
| `ComSen2026_提交报告.docx` | 7.8 MB | **Word 版报告，28 张图已内嵌** |
| `ComSen2026_提交报告.html` | 63 KB | 网页版（打开 → 打印 → 另存 PDF） |
| `ComSen2026_提交报告.md` | 50 KB | Markdown 源文件 |
| `ComSen2026_task2-3-4_提交包.zip` | 77 MB | 上面这些 + 图片 + 视频 + 模型 |
| `图片/` | 36 张 | 三题全部结果图（按题号命名） |
| `视频/` | 3 个 | task-2 要求提交的三个结果视频 |
| `模型/` | 2 个 | task-4 的 `best.pt`（6.16 MB）/ `best.onnx`（12.09 MB） |

## 图片清单（报告里引用的 {len(copied)} 张）

| 文件 |
| --- |
{imgs}

---

# 六、复现命令（写进仓库 README）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python run_all.py                 # task-2 + task-3 + 自检（约 5~8 分钟）
python run_all.py --with-task4    # 再加 task-4 题1~题3（CPU 上 YOLO 训练约 16 分钟）
python tools/verify_outputs.py    # 只做交付自检
python tools/build_submission.py  # 重新生成"提交/"材料包
```

> 已知坑：`pip install ultralytics` 会把 `opencv-python 5.x` 装进来并覆盖 contrib 版 `cv2`
> （OpenCV 5 里没有 CSRT/KCF tracker），导致 task-2 题五报错。修复：
> `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python==4.11.0.86`
"""


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    build()
