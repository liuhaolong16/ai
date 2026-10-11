# -*- coding: utf-8 -*-
"""
make_graphical_abstract.py — 概念框架图 / 研究设计图（Graphical Abstract）

用途：本文的**概念框架图**（不是理论框架 —— 本文不含理论机制主张，见 §1.5）。
把「数据 → 设计 → 主张与证据」三带与六条编号关系画在一张图上；每条编号关系在正文
都有对应小节，并在图注中指明。

★ 三道机械校验（缺一不可 —— 前两道都曾「全绿而产物是坏的」）：
  1. assert_glyphs —— **字形覆盖**。首版用了带圈数字与叉号/双箭头，Arial 的 cmap 里没有，
     PDF 会渲染成豆腐块，而 figkit 的规格校验**照样 PASS**。
  2. assert_layout —— **几何不重叠 / 文字不溢出**。第二版布局的行距是实际行高的 2.4 倍，
     方框互相压叠、文字全部越界，而**前两道校验一起 PASS**。
  3. figkit.check_pdf_or_raise —— 矢量 / 字体族 / Type3 / 字号下限 / 页面尺寸。
  ⇒ 教训：**规格检查只查它声明的那些性质，没声明的它一律放行。**
     每撞一次「绿灯但产物坏」，就把这一性质补成一条**可失败的检查**，而不是去修那一次
     （对照本方案修正十四「绿灯也要交叉确认」、判据质量第 8 条「判据要与失败形态配对」）。

输出：figures/graphical_abstract.pdf + .png
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(os.path.dirname(os.path.dirname(_HERE)), "figkit"), _HERE):
    if os.path.exists(os.path.join(_cand, "figkit.py")):
        sys.path.insert(0, _cand)
        break
import figkit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# 图目录：运行时布局为 <run>/figures；仓库布局为 <repo>/figures（脚本在 <repo>/code/）
_FIGC = [os.path.join(_HERE, "figures"), os.path.join(os.path.dirname(_HERE), "figures")]
FIGDIR = next((p for p in _FIGC if os.path.isdir(p)), _FIGC[0])

FIG_W, FIG_H = 6.73, 4.00     # 双栏宽（inch）
FS, FS_T = 7.0, 8.0           # 正文 / 小标题（SPEC 下限 7.0）
LINE_H = 0.042                # 行距（轴单位）；实测：(7/72)/4.0*1.7 ≈ 0.041
PAD = 0.014                   # 框内上下留白

_TEXT = []                    # [(artist, s)]
_BOXES = []                   # [(x, y, w, h, label)]
_TEXTBOX = []                 # [(artist, box_index)]


def T(ax, x, y, s, box_idx=None, **kw):
    t = ax.text(x, y, s, **kw)
    _TEXT.append((t, s))
    if box_idx is not None:
        _TEXTBOX.append((t, box_idx))
    return t


def box(ax, x, y_top, w, lines, fc, ec, dashed=False):
    """从 y_top 向下画一个高度自适应的框；返回框底 y。"""
    h = 2 * PAD + len(lines) * LINE_H
    y = y_top - h
    _BOXES.append((x, y, w, h, lines[0]))
    idx = len(_BOXES) - 1
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=0.9,
                               linestyle=(0, (3, 2)) if dashed else "-", zorder=1))
    ty = y_top - PAD
    for i, txt in enumerate(lines):
        T(ax, x + w / 2, ty, txt, box_idx=idx,
          ha="center", va="top", zorder=3,
          fontsize=FS_T if i == 0 else FS,
          fontweight="bold" if i == 0 else "normal")
        ty -= LINE_H
    return y


def arrow(ax, x0, y0, x1, y1, label=None, color="#555555"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.9,
                                shrinkA=0, shrinkB=0))
    if label:
        T(ax, (x0 + x1) / 2, (y0 + y1) / 2 + 0.012, label, ha="center", va="bottom",
          fontsize=FS, color=color)


# ─────────────────────────── 校验 1：字形覆盖 ───────────────────────────
def assert_glyphs():
    from fontTools.ttLib import TTFont
    fam = matplotlib.rcParams["font.family"]
    if isinstance(fam, str):
        fam = [fam]
    path = matplotlib.font_manager.findfont(
        matplotlib.font_manager.FontProperties(
            family=fam + matplotlib.rcParams["font.sans-serif"]))
    cmap = TTFont(path).getBestCmap()
    missing = sorted({c for _, s in _TEXT for c in s
                      if not c.isspace() and ord(c) not in cmap})
    if missing:
        raise SystemExit("FAIL glyph coverage - render as tofu: %s (font %s)"
                         % (" ".join("U+%04X %s" % (ord(c), c) for c in missing),
                            os.path.basename(path)))
    print("OK  [1/3] glyph coverage PASS (%d unique chars, %s)"
          % (len({c for _, s in _TEXT for c in s}), os.path.basename(path)))


# ─────────────────── 校验 2：几何不重叠 / 文字不越界 ───────────────────
def assert_layout(fig, ax):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def disp(xy):
        return ax.transData.transform(xy)

    def box_disp(b):
        x, y, w, h = b[:4]
        (x0, y0), (x1, y1) = disp((x, y)), disp((x + w, y + h))
        return Bbox.from_extents(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    # (a) 框两两不重叠
    bad = []
    for i in range(len(_BOXES)):
        for j in range(i + 1, len(_BOXES)):
            if box_disp(_BOXES[i]).overlaps(box_disp(_BOXES[j])):
                bad.append("框重叠：%r 与 %r" % (_BOXES[i][4], _BOXES[j][4]))
    # (b) 框内文字不越界
    for t, bi in _TEXTBOX:
        tb, bb = t.get_window_extent(r), box_disp(_BOXES[bi])
        if not (bb.x0 - 1 <= tb.x0 and tb.x1 <= bb.x1 + 1
                and bb.y0 - 1 <= tb.y0 and tb.y1 <= bb.y1 + 1):
            bad.append("文字越界：框 %r 内的 %r" % (_BOXES[bi][4], t.get_text()))
    if bad:
        raise SystemExit("FAIL layout - %d 处几何问题：\n  - %s"
                         % (len(bad), "\n  - ".join(bad[:8])))
    print("OK  [2/3] layout PASS (%d boxes pairwise disjoint, %d texts inside their boxes)"
          % (len(_BOXES), len(_TEXTBOX)))


def main():
    figkit.apply_style()
    fig, ax = figkit.new_fig(width=FIG_W, height=FIG_H)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    P = figkit.PALETTE
    e_d, e_g, e_r = P["blue"], P["vermillion"], P["green"]
    c_d, c_g, c_r = "#EAF3FA", "#FDF3E7", "#EDF7F1"

    for x0, w, fc, ec in ((0.005, 0.300, c_d, e_d),
                          (0.345, 0.300, c_g, e_g),
                          (0.685, 0.310, c_r, e_r)):
        ax.add_patch(plt.Rectangle((x0, 0.055), w, 0.885, facecolor=fc, edgecolor=ec,
                                   linewidth=0.7, alpha=0.5, zorder=0))
    T(ax, 0.155, 0.912, "Data", ha="center", fontsize=FS_T, fontweight="bold", color=e_d)
    T(ax, 0.495, 0.912, "Design", ha="center", fontsize=FS_T, fontweight="bold", color=e_g)
    T(ax, 0.840, 0.912, "Claims and evidence",
      ha="center", fontsize=FS_T, fontweight="bold", color=e_r)

    CX1, CX2, CX3 = 0.025, 0.365, 0.702
    CW = 0.268

    # ---- 带 1 ----
    y = box(ax, CX1, 0.878, CW, [
        "NASA PCoE Data Set",
        "34 cells, B0005-B0056",
        "ambient_temperature read",
        "from each cycle record,",
        "not from filenames",
    ], "#FFFFFF", e_d)
    T(ax, CX1 + CW / 2, y - 0.020, "\u25bc", ha="center", va="top", fontsize=8, color=e_d)
    y = box(ax, CX1, y - 0.050, CW, [
        "Within-cell MAD filter",
        "median \u00b1 3\u00b71.4826\u00b7MAD of",
        "log duration, log mean|I|,",
        "rest fraction",
        "26 cells / 2,084 cycles",
        "4 / 24 / 43 \u00b0C",
    ], "#FFFFFF", e_d)
    T(ax, CX1 + CW / 2, y - 0.018, "6 protocol-mixed cells", ha="center", va="top",
      fontsize=FS, style="italic", color="#555555")
    T(ax, CX1 + CW / 2, y - 0.040, "recorded separately (S1)", ha="center", va="top",
      fontsize=FS, style="italic", color="#555555")

    # ---- 带 2 ----
    box(ax, CX2, 0.878, CW, [
        "(1) Protocol A",
        "within-temperature LOCO",
        "train/test share T,",
        "different cells",
    ], "#FFFFFF", e_g)
    box(ax, CX2, 0.640, CW, [
        "(2) Protocol B",
        "cross-temperature LOTO",
        "train on two T,",
        "hold out the third",
    ], "#FFFFFF", e_g)
    box(ax, CX2, 0.402, CW, [
        "(3) Feature ablation",
        "F1 (11 indirect)",
        "+cycle_idx \u2192 F2",
        "+cap_ah \u2192 F3",
        "Sec. 3.1",
    ], "#FFFFFF", e_g)

    # ---- 带 3 ----
    box(ax, CX3, 0.878, CW, [
        "(4) Boundary",
        "within-T works,",
        "cross-T fails",
        "Table 2, Fig. 2",
    ], "#FFFFFF", e_r)
    box(ax, CX3, 0.640, CW, [
        "(5) Tautology",
        "cap_ah inflates",
        "protocol A",
        "Fig. 3",
    ], "#FFFFFF", e_r)
    box(ax, CX3, 0.402, CW, [
        "(6) Joint failure",
        "11 leave-one-feature",
        "out runs; min stays < -1",
        "Table 3, Fig. 4",
    ], "#FFFFFF", e_r)
    box(ax, CX3, 0.206, CW, [
        "\u00d7 SHAP ranking",
        "\u2192 causation",
        "falsified (Sec. 5.3)",
    ], "#FFFFFF", e_r, dashed=True)

    # ---- 箭头 ----
    arrow(ax, 0.298, 0.790, 0.360, 0.790, "apply")
    arrow(ax, 0.298, 0.560, 0.360, 0.560)
    arrow(ax, 0.638, 0.790, 0.697, 0.790, "evaluate")
    arrow(ax, 0.638, 0.560, 0.697, 0.560)
    arrow(ax, 0.638, 0.320, 0.697, 0.320)

    T(ax, 0.5, 0.028, "Boundary quantified and candidate mechanisms excluded, "
                      "not a method that generalizes",
      ha="center", va="bottom", fontsize=FS, fontweight="bold")

    out_pdf = os.path.join(FIGDIR, "graphical_abstract.pdf")
    out_png = os.path.join(FIGDIR, "graphical_abstract.png")

    assert_glyphs()
    assert_layout(fig, ax)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    figkit.check_pdf_or_raise(out_pdf)
    print("OK  [3/3] figkit spec PASS")
    print("WROTE", out_pdf)
    print("WROTE", out_png)


if __name__ == "__main__":
    main()
