import numpy as np
from matplotlib.patches import Patch
# stack_plotter_function.py  (very top of file)

# stack_plotter_function.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def _grey_hex(level: float) -> str:
    g = int(max(0, min(1, level)) * 255)
    return f"#{g:02x}{g:02x}{g:02x}"

def plot_stacked_phases_with_grey_boundaries(
    # FILL from grid (smooth)
    df_fill,                          # must contain columns: Ba_Zr_ratio, BaZrS3, Ba4Zr3S10, Ba3Zr2S7, ZrO2
    # optional MARKERS from measured (sparse)
    df_mark=None,                     # same columns as above but only at measured x
    figsize=(6.8, 4.6),
    xlim=(0.42, 0.59),
    ylim=(0.0, 1.00),
    fill_colors=dict(ZrO2="#ffb3a8", BZS="#bfe6bf", B4="#b7d5f5", B3="#d8c6f6"),
    fill_alpha=dict(ZrO2=0.22, BZS=0.40, B4=0.35, B3=0.35),
    grey_level=0.72,
    b3_grey_level=None,
    boundary_marker='o',
    boundary_edgewidth=1.0,
    boundary_size=50,
    boundary_alpha=0.7,
    decimate=4,
    # dashed boundary styling (thin dashed requested)
    dashed_all_boundaries=True,
    dashed_lw=1.0,
    dashed_alpha=0.9,
    # decorations
    xlabel_text="[Ba] / ([Ba] + [Zr])",
    ylabel_text="Phase fraction",
    title_text="",
    labelsize=14,
    ticklabelsize=14,
    titlesize=None,
    show_guides=True,
    guide_positions=(0.50, 4/7, 0.60),
    show_legend=False,
    annotations=None,
    region_lines=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # --- sanitize decimate ---
    if decimate is None or (isinstance(decimate, (int, float)) and decimate <= 0):
        decimate = 1
    else:
        decimate = int(max(1, round(float(decimate))))

    # ---------- FILL & DASHED LINES from GRID ----------
    order_f = np.argsort(df_fill["Ba_Zr_ratio"].values)
    xf  = df_fill["Ba_Zr_ratio"].values[order_f].astype(float)
    yBf = df_fill["BaZrS3"].values[order_f].astype(float)
    y4f = df_fill["Ba4Zr3S10"].values[order_f].astype(float)
    y3f = df_fill["Ba3Zr2S7"].values[order_f].astype(float)
    yZf = df_fill["ZrO2"].values[order_f].astype(float)

    Ff = np.vstack([yBf, y4f, y3f, yZf])
    Ff = np.clip(Ff, 0.0, None)
    sf = Ff.sum(axis=0); sf[sf == 0] = 1.0
    Ff /= sf
    yBf, y4f, y3f, yZf = Ff

    s1f = yBf
    s2f = yBf + y4f
    s3f = yBf + y4f + y3f

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # stacked fills from GRID
    base = np.zeros_like(xf)
    for y, c, a in zip([yBf, y4f, y3f, yZf],
                       [fill_colors["BZS"], fill_colors["B4"], fill_colors["B3"], fill_colors["ZrO2"]],
                       [fill_alpha["BZS"],  fill_alpha["B4"],  fill_alpha["B3"],  fill_alpha["ZrO2"]]):
        ax.fill_between(xf, base, base + y, facecolor=c, edgecolor='none', alpha=a, zorder=1)
        base += y

    # thin dashed boundaries from GRID
    if dashed_all_boundaries:
        edge_grey = _grey_hex(grey_level)
        line_kw = dict(linestyle="--", linewidth=dashed_lw, alpha=dashed_alpha, color=edge_grey, zorder=3)
        ax.plot(xf, s1f, **line_kw)
        ax.plot(xf, s2f, **line_kw)
        ax.plot(xf, s3f, **line_kw)

    # ---------- MARKERS only at MEASURED x ----------
    if df_mark is not None and len(df_mark):
        order_m = np.argsort(df_mark["Ba_Zr_ratio"].values)
        xm  = df_mark["Ba_Zr_ratio"].values[order_m].astype(float)
        yBm = df_mark["BaZrS3"].values[order_m].astype(float)
        y4m = df_mark["Ba4Zr3S10"].values[order_m].astype(float)
        y3m = df_mark["Ba3Zr2S7"].values[order_m].astype(float)
        yZm = df_mark["ZrO2"].values[order_m].astype(float)

        Fm = np.vstack([yBm, y4m, y3m, yZm])
        Fm = np.clip(Fm, 0.0, None)
        sm = Fm.sum(axis=0); sm[sm == 0] = 1.0
        Fm /= sm
        yBm, y4m, y3m, yZm = Fm

        s1m = yBm
        s2m = yBm + y4m
        s3m = yBm + y4m + y3m

        edge_grey = _grey_hex(grey_level)
        edge_b3   = _grey_hex(grey_level if b3_grey_level is None else b3_grey_level)

        idx = np.arange(0, len(xm), decimate)
        mkw = dict(facecolors='none', edgecolors=edge_grey,
                   linewidths=boundary_edgewidth, s=boundary_size,
                   alpha=boundary_alpha, zorder=5)
        ax.scatter(xm[idx], s1m[idx], marker=boundary_marker, **mkw)
        ax.scatter(xm[idx], s2m[idx], marker=boundary_marker, **mkw)
        ax.scatter(xm[idx], s3m[idx], marker=boundary_marker, **mkw)

        # optional explicit Ba3Zr2S7 markers (all non-zero)
        tol = 1e-8
        mask_b3 = (y3m > tol)
        if np.any(mask_b3):
            xi = xm[mask_b3]
            yi = s2m[mask_b3]  # lower boundary for B3 band
            ax.scatter(xi, yi, marker=boundary_marker,
                       facecolors='none', edgecolors=edge_b3,
                       linewidths=boundary_edgewidth, s=boundary_size*1.1,
                       alpha=boundary_alpha, zorder=6)

    # axes & cosmetics
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel_text, fontsize=labelsize, labelpad=10)
    ax.set_ylabel(ylabel_text, fontsize=labelsize, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
    if show_guides:
        for xv in guide_positions:
            if xlim[0] <= xv <= xlim[1]:
                ax.axvline(x=xv, color='k', lw=1.0, ls=':', alpha=0.8, zorder=2)

    # annotations / region_lines identical to your previous version...
    # (keep your existing code here)

    ax.grid(True, alpha=0.25, lw=0.6)
    plt.tight_layout()
    return fig, ax




ann1 = [
    {"text": "BaZrS$_3$", "xy": (0.46, 0.35), "size": 15, "color": "black"},
    {"text": "Ba$_4$Zr$_3$S$_{10}$", "xy": (0.555, 0.78), "size": 15},
    # {"text": "Ba$_3$Zr$_2$S$_7$",    "xy": (0.57, 0.40), "size": 15},
    {"text": "ZrO$_2$", "xy": (0.44, 0.93), "size": 15},
    {"text": r"Ba$_3$Zr$_2$S$_7$",
     "xy": (0.555, 0.55),  # arrow head (where the label sits)
     "xytext": (0.58, 0.45),  # arrow tail (where the line starts)
     "size": 15,
     "color": "black",
     "ha": "left", "va": "bottom",  # tweak alignment if needed
     "arrowprops": {  # optional styling
         "arrowstyle": "-",  # simple line
         "lw": 1.2,
         "color": "black",
         "shrinkA": 0, "shrinkB": 0
     }
     }
]

ann = [
    {"text": "BaZrS$_3$", "xy": (0.46, 0.35), "size": 15, "color": "black"},
    {"text": "Ba$_4$Zr$_3$S$_{10}$", "xy": (0.555, 0.78), "size": 15},
    {"text": "ZrO$_2$", "xy": (0.44, 0.93), "size": 15},
    # Ba3Zr2S7 label (no arrow here; line is handled by region_lines)
    {"text": r"Ba$_3$Zr$_2$S$_7$", "xy": (0.552, 0.5), "size": 15,
     "ha": "left", "va": "bottom", "color": "black"},
]

region_lines = [
    {"xy1": (0.58, 0.56), "xy2": (0.585, 0.655),
     "style": "-", "color": "black", "lw": 1.2, "alpha": 1.0}
    # or use an arrow head:
    # {"xy1": (0.57, 0.55), "xy2": (0.58, 0.65),
    #  "style": "->", "color": "black", "lw": 1.2}
]


#plot results suite - not sure I really need this
def plot_element_compare_onepanel(
    df_meas,
    predA,
    predB=None,                          # optional second model (unused here but accepted)
    *,
    xcol="Ba_Zr_ratio",
    figsize=(7.6, 4.2),
    xlim=None,
    ylim=(10, 65),
    meas_grey="#7f7f7f",
    lw=2.0,

    # inline curve labels: dict like {"Ba": (x,y), "Zr": (x,y), "S": (x,y)}
    label_positions=None,
    # colors for model curves + labels
    label_size = 14,
    label_colors=None,                   # dict like {"Ba":"#2ca25f","Zr":"#3182bd","S":"#de2d26"}

    # small panel tag like "(b)"
    panel_tag=None,                      # e.g. "(b)"
    panel_tag_xy=(0.02, 0.96),          # axes fraction
    panel_tag_size=14,
    panel_tag_weight="normal",           # "normal" or "bold"
):
    """
    One-panel comparison: measured (open grey circles) vs predicted (lines).
    Accepts label_positions/label_colors/panel_tag to avoid using legend.
    Returns (fig, ax).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ticklabelsize = 14

    xlabel_text = label_size

    # defaults
    if label_colors is None:
        label_colors = {"Ba": "#2ca25f", "Zr": "#3182bd", "S": "#de2d26"}

    x = df_meas[xcol].to_numpy(float)
    o = np.argsort(x); x = x[o]

    Ba_m = df_meas["Ba_norm"].to_numpy(float)[o]
    Zr_m = df_meas["Zr_norm"].to_numpy(float)[o]
    S_m  = df_meas["S_norm"].to_numpy(float)[o]
    Ba_m *=100
    Zr_m *=100
    S_m *=100


    Ba_p, Zr_p, S_p = [np.asarray(v)[o] for v in predA]
    Ba_p *= 100
    Zr_p *= 100
    S_p *= 100


    fig, ax = plt.subplots(figsize=figsize)

    # measured (grey, open)
    mk = dict(marker="o", mfc="none", mec=meas_grey, ms=5, ls="None")
    ax.plot(x, Ba_m, **mk)
    ax.plot(x, Zr_m, **mk)
    ax.plot(x, S_m,  **mk)

    # model curves
    ax.plot(x, Ba_p, "-", lw=lw, color=label_colors["Ba"])
    ax.plot(x, Zr_p, "-", lw=lw, color=label_colors["Zr"])
    ax.plot(x, S_p,  "-", lw=lw, color=label_colors["S"])

    # inline labels (instead of legend)
    if label_positions:
        if "Ba" in label_positions:
            xb, yb = label_positions["Ba"]
            ax.text(xb, yb, "modeled Ba", color=label_colors["Ba"], fontsize=12, ha="left", va="center")
        if "Zr" in label_positions:
            xz, yz = label_positions["Zr"]
            ax.text(xz, yz, "modeled Zr", color=label_colors["Zr"], fontsize=12, ha="left", va="center")
        if "S" in label_positions:
            xs_, ys_ = label_positions["S"]
            ax.text(xs_, ys_, "modeled S",  color=label_colors["S"],  fontsize=12, ha="left", va="center")

    # axes, limits, grid
    if xlim is None:
        xlim = (float(x.min()), float(x.max()))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #ax.set_xlabel("[Ba] / ([Ba] + [Zr])")
    #ax.set_ylabel("Composition (at.%)")
    ax.set_xlabel("[Ba] / ([Ba] + [Zr])", fontsize=label_size, labelpad=10)
    ax.set_ylabel("Composition (at.%)", fontsize=label_size, labelpad=10)


    ax.grid(True, alpha=0.25, lw=0.6)

    # small panel tag like "(b)"
    if panel_tag:
        ax.annotate(panel_tag, xy=panel_tag_xy, xycoords="axes fraction",
                    ha="left", va="top", fontsize=panel_tag_size,
                    fontweight=panel_tag_weight, color="black")
    ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
    plt.tight_layout()
    return fig, ax


