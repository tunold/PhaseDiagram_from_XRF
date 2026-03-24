# =============================================================================
# Ba–Zr–S Phase Model from XRF Data
#
# Author:       Thomas Unold
# Affiliation:  Helmholtz-Zentrum Berlin für Materialien und Energie
# Year:        2025
# License:     MIT
#
# Description:
#   Streamlit app for modeling the phase distribution in Ba–Zr–S thin films
#   from XRF measurements. Phases: BaZrS3, Ba4Zr3S10, Ba3Zr2S7, ZrO2.
# ============================================================================

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


# ====== plotting helpers from your module ======
try:
    from stack_plotter_function import (
        plot_stacked_phases_with_grey_boundaries,
        plot_element_compare_onepanel,
    )
except Exception as e:
    st.error("Couldn't import `stack_plotter_function`. Make sure it sits next to this app.")
    raise

plt.rcParams["figure.dpi"] = 120
st.set_page_config(page_title="Ba–Zr–S Phasenmodell", layout="wide")

figsize_x = 4.9
figsize_y = 3.5

# ---- Generic annotation helper ----
def add_user_annotation(
    ax, text, xy, *, coord_system="data", fontsize=12, color="k", weight="normal",
    ha="left", va="bottom", bbox=None, arrow_to=None, arrow_kw=None
):
    if coord_system == "axes":
        trans = ax.transAxes
    else:
        trans = ax.transData

    if arrow_to is not None:
        ax.annotate(
            text,
            xy=arrow_to, xycoords=("data" if coord_system=="data" else "axes fraction"),
            xytext=xy, textcoords=(trans),
            fontsize=fontsize, color=color, weight=weight, ha=ha, va=va,
            bbox=bbox,
            arrowprops=(arrow_kw or dict(arrowstyle="-", lw=1.0, color=color)),
            annotation_clip=False,
        )
    else:
        ax.text(
            xy[0], xy[1], text, transform=trans,
            fontsize=fontsize, color=color, weight=weight, ha=ha, va=va,
            bbox=bbox, zorder=20,
        )


# ----------------------------- Utilities -------------------------------------
def _to_float(s):
    if isinstance(s, str):
        s = s.replace(",", ".")
    return pd.to_numeric(s, errors="coerce")

def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def r2_and_residuals(df_meas, Ba_p, Zr_p, S_p):
    x    = df_meas["Ba_Zr_ratio"].to_numpy(float)
    Ba_m = df_meas["Ba_norm"].to_numpy(float)
    Zr_m = df_meas["Zr_norm"].to_numpy(float)
    S_m  = df_meas["S_norm"].to_numpy(float)

    def r2(y, yhat):
        y = np.asarray(y, float); yhat = np.asarray(yhat, float)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-18
        return 1.0 - ss_res / ss_tot

    R2 = dict(Ba=r2(Ba_m, Ba_p), Zr=r2(Zr_m, Zr_p), S=r2(S_m, S_p))
    resid = dict(
        x=x,
        Ba_res=(Ba_m - np.asarray(Ba_p)),
        Zr_res=(Zr_m - np.asarray(Zr_p)),
        S_res =(S_m  - np.asarray(S_p)),
    )
    return R2, resid

def fig_to_png_bytes(fig, dpi=600):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# ----------------------- Stoichiometry & forward model ------------------------
A_STO = np.array([
    [1, 4, 3, 0],  # Ba
    [1, 3, 2, 1],  # Zr
    [3,10, 7, 0],  # S
    [0, 0, 0, 2],  # O (nur ZrO2)
], dtype=float)

def backproject_BaZrS(BZS, B4, B3, ZR):
    F = np.vstack([BZS, B4, B3, ZR])
    counts = (A_STO[:3, :] @ F)
    den = counts.sum(axis=0, keepdims=True); den[den == 0] = 1.0
    frac = counts / den
    return frac[0], frac[1], frac[2]

def piecewise_with_ZrO2_PCHIP(x, xmin, xmax, zL, z50, z54, z57, z60,
                                monotone_right=False, x_anchor=0.40):
    x = np.asarray(x, float)
    x1, x1b, x2, x3 = 0.50, 0.54, 4.0/7.0, 0.60

    def z_left(xv):
        t = (xv - x_anchor) / max(1e-12, (x1 - x_anchor))
        return (1.0 - t)*zL + t*z50

    kz = np.array([z50, z54, z57, z60], float)
    if monotone_right:
        for i in range(1, len(kz)):
            kz[i] = min(kz[i], kz[i-1])
    z_fun = PchipInterpolator([x1, x1b, x2, x3], kz, extrapolate=True)

    z = np.empty_like(x)
    mL = (x < x1)
    z[mL]  = np.clip([z_left(xv) for xv in x[mL]], 0, 1)
    z[~mL] = np.clip(z_fun(x[~mL]), 0, 1)

    scale = np.clip(1.0 - z, 0.0, 1.0)

    BZS = np.zeros_like(x); B4 = np.zeros_like(x); B3 = np.zeros_like(x)

    if mL.any():
        BZS[mL] = scale[mL]

    m2 = (x >= x1) & (x < x2)
    if m2.any():
        t = (x[m2] - x1) / max(1e-12, (x2 - x1))
        B4[m2]  = t       * scale[m2]
        BZS[m2] = (1 - t) * scale[m2]

    m3 = (x >= x2)
    if m3.any():
        t = (x[m3] - x2) / max(1e-12, (x3 - x2))
        t = np.clip(t, 0, 1)
        B3[m3] = t         * scale[m3]
        B4[m3] = (1.0 - t) * scale[m3]

    F = np.vstack([BZS, B4, B3, z])
    den = F.sum(axis=0, keepdims=True); den[den == 0] = 1.0
    F /= den
    return F[0], F[1], F[2], F[3]

# ---------------------- Globaler Fit ------------------------------------------
def fit_knots_global_rtot(
    df_meas,
    init=(0.35, 0.06, 0.06, 0.04, 0.02),
    bounds=((0.3, 1.0),(0, 0.30),(0, 0.30),(0, 0.30),(0, 0.30)),
    monotone_right=False,
    w_Ba=1.0, w_Zr=1.0, w_S=1.0,
    slope_match=True,
    slope_penalty_weight=1e-3,
):
    """Global fit: maximiert gewichtete R²-Summe (Ba, Zr, S)."""
    x    = df_meas["Ba_Zr_ratio"].to_numpy(float)
    xmin, xmax = float(x.min()), float(x.max())
    Ba_m = df_meas["Ba_norm"].to_numpy(float)
    Zr_m = df_meas["Zr_norm"].to_numpy(float)
    S_m  = df_meas["S_norm"].to_numpy(float)

    x_anchor = 0.40

    def ss_res(y, yhat):
        return float(np.sum((np.asarray(y, float) - np.asarray(yhat, float)) ** 2))

    def ss_tot(y):
        y = np.asarray(y, float)
        return float(np.sum((y - y.mean()) ** 2)) + 1e-18

    def slope_penalty(zL, z50, z54, z57, z60):
        if not slope_match:
            return 0.0
        sL = (z50 - zL) / max(1e-12, (0.50 - x_anchor))
        z_fun = PchipInterpolator(
            [0.50, 0.54, 4.0/7.0, 0.60],
            [z50,  z54,  z57,     z60],
            extrapolate=True
        )
        sR = float(z_fun.derivative()(0.50))
        return (sR - sL) ** 2

    def evaluate(p):
        zL, z50, z54, z57, z60 = p
        BZS, B4, B3, ZR = piecewise_with_ZrO2_PCHIP(
            x, xmin, xmax, zL, z50, z54, z57, z60, monotone_right=monotone_right
        )
        Ba_p, Zr_p, S_p = backproject_BaZrS(BZS, B4, B3, ZR)
        return BZS, B4, B3, ZR, Ba_p, Zr_p, S_p

    def objective(p):
        _, _, _, _, Ba_p, Zr_p, S_p = evaluate(p)
        r2_ba = 1.0 - ss_res(Ba_m, Ba_p) / ss_tot(Ba_m)
        r2_zr = 1.0 - ss_res(Zr_m, Zr_p) / ss_tot(Zr_m)
        r2_s  = 1.0 - ss_res(S_m,  S_p)  / ss_tot(S_m)
        return float(-(w_Ba * r2_ba + w_Zr * r2_zr + w_S * r2_s)
                     + slope_penalty_weight * slope_penalty(*p))

    res_global = differential_evolution(
        objective, bounds=bounds, seed=42, tol=1e-9, init='latinhypercube'
    )

    res = minimize(
        objective,
        x0=res_global.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 10000}
    )

    zL, z50, z54, z57, z60 = [float(v) for v in res.x]
    BZS, B4, B3, ZR, Ba_p, Zr_p, S_p = evaluate(res.x)
    R2, resid = r2_and_residuals(df_meas, Ba_p, Zr_p, S_p)
    r2_sum = R2["Ba"] + R2["Zr"] + R2["S"]

    return dict(
        params=dict(zL=zL, z50=z50, z54=z54, z57=z57, z60=z60),
        success=bool(res.success),
        message=str(res.message),
        objective_value=r2_sum,
        BZS=BZS, B4=B4, B3=B3, ZR=ZR,
        Ba_p=Ba_p, Zr_p=Zr_p, S_p=S_p,
        R2=R2, residuals=resid,
    )

# Backwards-compatible alias
fit_knots_least_squares = fit_knots_global_rtot

# -------------------------- Session state init --------------------------------
for key in ("model", "df_fit_grid", "df_fit_meas", "elem_panel", "last_fit_result"):
    if key not in st.session_state:
        st.session_state[key] = None

for key in ("model", "df_fit_grid", "df_fit_meas", "elem_panel", "last_fit_result", "knots_used"):
    if key not in st.session_state:
        st.session_state[key] = None

# trigger_calc: wird von fit_now gesetzt, damit do_calc im nächsten Rerun feuert
if "trigger_calc" not in st.session_state:
    st.session_state["trigger_calc"] = False


# ------------------------------- Daten laden ----------------------------------
st.sidebar.header("load data")
up = st.sidebar.file_uploader("choose CSV file", type=["csv"])

with st.sidebar.expander("Column names"):
    st.caption("Change only if your CSV uses different column names.")
    c_ba = st.text_input("Ba column [at%]", "Baat")
    c_zr = st.text_input("Zr column [at%]", "Zrat")
    c_s  = st.text_input("S  column [at%]", "Sat")

# NEU:
if up is not None:
    raw = pd.read_csv(up)
    try:
        df_meas = pd.DataFrame()
        df_meas["Ba_norm"]     = (_to_float(raw[c_ba]) / 100.0).astype(float)
        df_meas["Zr_norm"]     = (_to_float(raw[c_zr]) / 100.0).astype(float)
        df_meas["S_norm"]      = (_to_float(raw[c_s])  / 100.0).astype(float)
        df_meas["Ba_Zr_ratio"] = df_meas["Ba_norm"] / (df_meas["Ba_norm"] + df_meas["Zr_norm"])
    except KeyError as e:
        st.error(f"Column not found: {e}. Please check the column names.")
        st.stop()
else:
    try:
        df_meas_raw = pd.read_csv("analyzedXRF_3925-20.csv")
        df_meas = pd.DataFrame()
        df_meas["Ba_norm"] = (_to_float(df_meas_raw[c_ba]) / 100.0).astype(float)
        df_meas["Zr_norm"] = (_to_float(df_meas_raw[c_zr]) / 100.0).astype(float)
        df_meas["S_norm"] = (_to_float(df_meas_raw[c_s]) / 100.0).astype(float)
        df_meas["Ba_Zr_ratio"] = df_meas["Ba_norm"] / (df_meas["Ba_norm"] + df_meas["Zr_norm"])
        st.info("No CSV uploaded — using default dataset: analyzedXRF_3925-20.csv")
    except FileNotFoundError:
        st.warning("No CSV uploaded and default file not found — using synthetic demo data.")
        xx = np.linspace(0.42, 0.60, 80)
        Ba_demo = 0.22 + 0.28 * (xx - 0.42) / 0.18
        Zr_demo = 0.26 - 0.22 * (xx - 0.42) / 0.18
        S_demo = 1.0 - (Ba_demo + Zr_demo)
        df_meas = pd.DataFrame({
            "Ba_Zr_ratio": xx, "Ba_norm": Ba_demo,
            "Zr_norm": Zr_demo, "S_norm": S_demo
        })

for c in ("Ba_norm","Zr_norm","S_norm","Ba_Zr_ratio"):
    df_meas[c] = pd.to_numeric(df_meas[c], errors="coerce")
df_meas = df_meas.dropna().reset_index(drop=True)

sum_bzs = df_meas[["Ba_norm","Zr_norm","S_norm"]].sum(axis=1).replace(0, np.nan)
df_meas["Ba_norm"] = df_meas["Ba_norm"] / sum_bzs
df_meas["Zr_norm"] = df_meas["Zr_norm"] / sum_bzs
df_meas["S_norm"]  = df_meas["S_norm"]  / sum_bzs

# ---------------- BBZ window --------------------------------------------------
st.sidebar.header("BBZ window (model & plots)")
xlo_sel, xhi_sel = st.sidebar.slider(
    "Ba/(Ba+Zr) range", min_value=0.0, max_value=1.0,
    value=(0.40, 0.60), step=0.001, key="bbz_window"
)

df_win = df_meas.loc[df_meas["Ba_Zr_ratio"].between(xlo_sel, xhi_sel)].copy().reset_index(drop=True)

# ------------------------------- Sidebar UI ----------------------------------
with st.sidebar.expander("2) ZrO₂-knots"):
    zL  = st.number_input("zL  (ZrO₂ @ x_anchor≈0.40)",
                          min_value=0.0, max_value=1.00,
                          value=0.30, step=0.01, format="%.4f", key="knot_zL")
    z50 = st.number_input("z50 (ZrO₂ @ 0.50)",
                          min_value=0.00, max_value=0.30,
                          value=0.06, step=0.005, format="%.4f", key="knot_z50")
    z54 = st.number_input("z54 (ZrO₂ @ 0.54)",
                          min_value=0.00, max_value=0.30,
                          value=0.06, step=0.005, format="%.4f", key="knot_z54")
    z57 = st.number_input("z57 (ZrO₂ @ 4/7)",
                          min_value=0.00, max_value=0.30,
                          value=0.04, step=0.005, format="%.4f", key="knot_z57")
    z60 = st.number_input("z60 (ZrO₂ @ 0.60)",
                          min_value=0.00, max_value=0.30,
                          value=0.02, step=0.005, format="%.4f", key="knot_z60")

monotone_right   = st.sidebar.checkbox("ZrO₂ monotone", False, key="monotone_right")
#use_fitted_knots = st.sidebar.checkbox("Use fitted knots for model calculation", False, key="use_fitted_knots")
do_calc          = st.sidebar.button("manual Fit", key="do_calc_btn")

#st.sidebar.markdown("---")
#st.sidebar.markdown("**Globaler Fit (R²)**")

with st.sidebar.expander("Tweaks"):
    slope_match = st.checkbox("Steigungen bei 0.50 weich anpassen (nur Fit)", True,
                               key="tweak_slope_match")
    zL_min = st.number_input("zL Minimum (untere Bound für Global Fit)",
                              min_value=0.00, max_value=1.00,
                              value=0.25, step=0.01, format="%.3f",
                              key="tweak_zl_min")
    w_Ba   = st.number_input("Gewicht Ba", min_value=0.1, max_value=5.0,
                              value=1.0, step=0.1, format="%.1f", key="tweak_wba")
    w_Zr   = st.number_input("Gewicht Zr", min_value=0.1, max_value=5.0,
                              value=1.0, step=0.1, format="%.1f", key="tweak_wzr")
    w_S    = st.number_input("Gewicht S",  min_value=0.1, max_value=5.0,
                              value=1.0, step=0.1, format="%.1f", key="tweak_ws")
    zb_hi  = st.number_input("Max. ZrO₂ rechts (z50..z60)",
                              min_value=0.00, max_value=0.30,
                              value=0.20, step=0.01, format="%.3f", key="tweak_zbhi")

bounds  = ((zL_min, 1.0), (0, zb_hi), (0, zb_hi), (0, zb_hi), (0, zb_hi))
fit_now = st.sidebar.button("Global Fit (R²)", key="fit_rtot_btn")

#
#st.sidebar.markdown("---")


#st.sidebar.markdown("---")
with st.sidebar.expander("Phase diagram appearance"):
    decimate       = st.number_input("Decimate (every n-th marker)", 1, 50, 4, 1)
    boundary_size  = st.slider("Boundary marker size", 20, 120, 50, 2)
    boundary_alpha = st.slider("Boundary alpha", 0.1, 1.0, 0.7, 0.05)
    grey_level     = st.slider("Marker grey level", 0.0, 1.0, 0.72, 0.01)
    b3_grey_level  = st.slider("B3 grey level (optional)", 0.0, 1.0, 0.72, 0.01)
    show_guides    = st.checkbox("Show guide lines (0.50, 4/7, 0.60)", value=True)

with st.sidebar.expander("Pastel fill colors"):
    col_zro2 = st.color_picker("ZrO₂ fill",       "#ffb3a8")
    col_bzs  = st.color_picker("BaZrS₃ fill",     "#bfe6bf")
    col_b4   = st.color_picker("Ba₄Zr₃S₁₀ fill",  "#b7d5f5")
    col_b3   = st.color_picker("Ba₃Zr₂S₇ fill",   "#d8c6f6")

#st.sidebar.header("Figure annotations")

# --- Literature point (deaktiviert) ---
use_lit    = False
lit_x      = 0.407
lit_y      = 0.68
lit_label  = ""
lit_labx   = 0.407
lit_laby   = 0.62
lit_marker = "D"
lit_size   = 80
lit_edge   = "#2ca25f"
lit_face   = "none"

# --- Figure annotations (deaktiviert) ---
annA_on = False; annA_text = ""; annA_sys = "data"
annA_x = 0.0;   annA_y = 0.0;   annA_fs = 12
annA_col = "#000000"; annA_wt = "normal"; annA_arrow = False
annA_x2 = 0.0;  annA_y2 = 0.0

annB_on = False; annB_text = ""; annB_sys = "data"
annB_x = 0.0;   annB_y = 0.0;   annB_fs = 12
annB_col = "#000000"; annB_wt = "normal"; annB_arrow = False
annB_x2 = 0.0;  annB_y2 = 0.0

annC_extra_on = True
extra_cfg = []

# ------------------------------ Title & meta ----------------------------------
st.write("Ba–Zr–S (including ZrO₂) from XRF data")

with st.expander("📖 Help / Documentation", expanded=False):
    st.markdown("""
## Phase Model Ba–Zr–S with ZrO₂

This app models the phase distribution in Ba–Zr–S thin films from XRF measurements
(Ba, Zr, S as normalized atomic fractions) as a function of the composition variable
**BBZ = Ba/(Ba+Zr)**.

---

### Physical Model

The film consists of up to four phases:

| Phase | Formula | Stoichiometry (Ba:Zr:S) |
|---|---|---|
| BaZrS₃ | Perovskite | 1:1:3 |
| Ba₄Zr₃S₁₀ | Ruddlesden-Popper n=3 | 4:3:10 |
| Ba₃Zr₂S₇ | Ruddlesden-Popper n=2 | 3:2:7 |
| ZrO₂ | Oxide (secondary phase) | 0:1:0 (with O) |

The sulfide phases are arranged along **tie-lines**:
- BBZ < 0.50 → BaZrS₃ only
- 0.50 ≤ BBZ < 4/7 → BaZrS₃ ↔ Ba₄Zr₃S₁₀
- BBZ ≥ 4/7 → Ba₄Zr₃S₁₀ ↔ Ba₃Zr₂S₇

The ZrO₂ fraction is modeled separately and scales the sulfide phases by the factor
`(1 − ZrO₂)`.

---

### ZrO₂ Curve: Control Points

The ZrO₂ profile over BBZ is defined piecewise:

- **Left branch** (BBZ < 0.50): linear interpolation between the fixed anchor point
  `x_anchor = 0.40` with value **zL** and the point BBZ = 0.50 with value **z50**
- **Right branch** (BBZ ≥ 0.50): PCHIP interpolation through four knots:
  **z50** (at 0.50), **z54** (at 0.54), **z57** (at 4/7 ≈ 0.571), **z60** (at 0.60)

PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) produces a smooth curve
that allows local extrema between knots — this is physically relevant since ZrO₂
can exhibit a maximum near BBZ ≈ 0.54.

**ZrO₂ monotone**: forces the ZrO₂ curve in the right branch to never increase
(z54 ≤ z50, z57 ≤ z54, etc.). For most datasets this option should remain **off**.

---

### Back-Projection

The modeled phase fractions are back-projected into normalized elemental fractions
(Ba, Zr, S) via the stoichiometry matrix and compared against the XRF measurements.
The quality of the fit is evaluated using **R²** for each channel independently.

---

### Workflow

#### Manual Fit
1. Upload a CSV file (or use the built-in demo data)
2. Set the BBZ window
3. Adjust the ZrO₂ knots (zL, z50, z54, z57, z60) manually
4. Click **manual Fit** → plots and R² values appear
5. Iteratively refine the knots to maximize the total R² sum

#### Global Fit
1. Under **Tweaks**: set zL minimum, channel weights, and max. ZrO₂
2. Click **global Fit (global R²)**
3. The optimizer maximizes `w_Ba·R²(Ba) + w_Zr·R²(Zr) + w_S·R²(S)` automatically
4. The result is displayed directly as the current model
5. The optimized knot values are shown in the sidebar —
   they can be transferred manually into the knot input fields if needed

---

### Tweaks (global fit only)

| Parameter | Description |
|---|---|
| **zL minimum** | Lower bound for zL in the fit — prevents unrealistic ZrO₂ values at the left edge |
| **Weight Ba/Zr/S** | Relative importance of the three channels in the objective function |
| **Max. ZrO₂ right** | Upper bound for z50–z60 in the fit |
| **Smooth slopes at 0.50** | Penalty term for a kink at the junction BBZ = 0.50 |

---

### Plots

| Plot | Content |
|---|---|
| **(a) Elements** | XRF meas. (symbols) vs. model (lines) for Ba, Zr, S |
| **(b) Residuals** | Difference measurement − model per element |
| **(c) Phase distribution** | Stacked phase fractions over BBZ with phase boundaries |

---

### CSV Format

The CSV file must contain the following columns (column names are configurable under *column names*):

| Column | Description | Unit |
|---|---|---|
| `Baat` | Ba content | at.% |
| `Zrat` | Zr content | at.% |
| `Sat`  | S content  | at.% |

Ba/(Ba+Zr) is calculated automatically from Ba and Zr — no separate column needed.

Values are normalized internally to Ba+Zr+S = 1.
""")

st.write("XRF Data:", len(df_meas))

# ------------------------------ Global Fit ------------------------------------
if fit_now:
    if len(df_win) < 2:
        st.sidebar.warning("Zu wenige Punkte im gewählten Fenster für den Fit.")
    else:
        bounds_arr = np.array(bounds)
        init_arr   = np.clip(
            [zL, z50, z54, z57, z60],
            bounds_arr[:, 0] + 1e-6,
            bounds_arr[:, 1] - 1e-6
        )
        with st.spinner("Globaler Fit läuft..."):
            fit_out = fit_knots_global_rtot(
                df_win,
                init=tuple(init_arr),
                bounds=bounds,
                monotone_right=monotone_right,
                w_Ba=w_Ba, w_Zr=w_Zr, w_S=w_S,
                slope_match=slope_match,
            )

        # ERST alles speichern, DANN rerun
        st.session_state["last_fit_result"] = fit_out
        st.session_state["trigger_calc"]    = True

        p = fit_out["params"]
        st.sidebar.success(f"Optimized R²-Summe = {fit_out['objective_value']:.4f}")
        st.sidebar.info(
            f"zL={p['zL']:.4f}  z50={p['z50']:.4f}\n"
            f"z54={p['z54']:.4f}  z57={p['z57']:.4f}  z60={p['z60']:.4f}"
        )

        st.rerun()  # ← ganz am Ende

# ------------------------------ Compute on click ------------------------------
# do_calc: manueller Button
# trigger_calc: gesetzt nach globalem Fit → wird einmalig verbraucht
trigger = st.session_state.get("trigger_calc", False)
if trigger:
    st.session_state["trigger_calc"] = False  # sofort zurücksetzen

if do_calc or trigger:
    df_win = df_meas.loc[df_meas["Ba_Zr_ratio"].between(xlo_sel, xhi_sel)].copy().reset_index(drop=True)
    x_win  = df_win["Ba_Zr_ratio"].to_numpy(float)

    if len(df_win) < 2:
        st.warning("Zu wenige Punkte im gewählten Fenster.")
    else:
        if trigger:
            # Global Fit → gefittete Knoten
            out    = st.session_state["last_fit_result"]
            params = out["params"]
            zL_use, z50_use, z54_use, z57_use, z60_use = (
                params["zL"], params["z50"], params["z54"], params["z57"], params["z60"]
            )
            M = out.copy()
            M["message"] = f"Global fit | R²-Summe = {out.get('objective_value', np.nan):.4f}"
        else:
            # Manuell → number_input Knoten
            zL_use, z50_use, z54_use, z57_use, z60_use = zL, z50, z54, z57, z60
            M = dict(
                params={"zL": zL_use, "z50": z50_use, "z54": z54_use,
                        "z57": z57_use, "z60": z60_use},
                success=True, message="Manual knot evaluation"
            )

        xg_lo = min(xlo_sel, lit_x) if use_lit else xlo_sel
        xg_hi = max(xhi_sel, lit_x) if use_lit else xhi_sel
        xg_lo = max(0.0, xg_lo); xg_hi = min(1.0, xg_hi)
        x_grid = np.linspace(xg_lo, xg_hi, 801)

        BZS_g, B4_g, B3_g, ZR_g = piecewise_with_ZrO2_PCHIP(
            x_grid, xlo_sel, xhi_sel,
            zL_use, z50_use, z54_use, z57_use, z60_use,
            monotone_right=monotone_right
        )
        Ba_g, Zr_g, S_g = backproject_BaZrS(BZS_g, B4_g, B3_g, ZR_g)

        df_fit_grid = pd.DataFrame({
            "Ba_Zr_ratio": x_grid,
            "BaZrS3": BZS_g, "Ba4Zr3S10": B4_g, "Ba3Zr2S7": B3_g, "ZrO2": ZR_g,
            "Ba_pred": Ba_g, "Zr_pred": Zr_g, "S_pred": S_g
        })
        st.session_state["df_fit_grid"] = df_fit_grid

        order_meas = np.argsort(x_win)
        xs_meas = x_win[order_meas]
        Ba_m    = df_win["Ba_norm"].to_numpy(float)[order_meas]
        Zr_m    = df_win["Zr_norm"].to_numpy(float)[order_meas]
        S_m     = df_win["S_norm"].to_numpy(float)[order_meas]

        Ba_p_s = np.interp(xs_meas, x_grid, Ba_g)
        Zr_p_s = np.interp(xs_meas, x_grid, Zr_g)
        S_p_s  = np.interp(xs_meas, x_grid, S_g)

        R2, resid = r2_and_residuals(
            pd.DataFrame({"Ba_Zr_ratio": xs_meas, "Ba_norm": Ba_m, "Zr_norm": Zr_m, "S_norm": S_m}),
            Ba_p_s, Zr_p_s, S_p_s
        )
        M.update(R2=R2, residuals=resid, success=True)
        st.session_state["model"] = M

        BZS_m, B4_m, B3_m, ZR_m = piecewise_with_ZrO2_PCHIP(
            xs_meas, xlo_sel, xhi_sel,
            zL_use, z50_use, z54_use, z57_use, z60_use,
            monotone_right=monotone_right
        )
        df_fit_meas = pd.DataFrame({
            "Ba_Zr_ratio": xs_meas,
            "BaZrS3": BZS_m, "Ba4Zr3S10": B4_m, "Ba3Zr2S7": B3_m, "ZrO2": ZR_m
        })
        st.session_state["df_fit_meas"] = df_fit_meas

        st.session_state["elem_panel"] = dict(
            xs=xs_meas, Ba_m=Ba_m, Zr_m=Zr_m, S_m=S_m,
            Ba_p=Ba_p_s, Zr_p=Zr_p_s, S_p=S_p_s
        )
        # am Ende von if do_calc or trigger:, nach den anderen session_state Zuweisungen:
        st.session_state["knots_used"] = dict(
            zL=zL_use, z50=z50_use, z54=z54_use, z57=z57_use, z60=z60_use
        )

# ------------------------------ Plots & KPIs ----------------------------------
if any(st.session_state[k] is None for k in ("model", "df_fit_grid", "df_fit_meas", "elem_panel")):
    st.info("Bitte **Modell berechnen** oder **Optimize knots** klicken.")
else:
    M  = st.session_state["model"]
    FG = st.session_state["df_fit_grid"]
    FM = st.session_state["df_fit_meas"]
    E  = st.session_state["elem_panel"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (Ba)", f"{M['R2']['Ba']:.3f}")
    c2.metric("R² (Zr)", f"{M['R2']['Zr']:.3f}")
    c3.metric("R² (S)",  f"{M['R2']['S']:.3f}")
    c4.metric("R² tot",  f"{M['R2']['Ba']+M['R2']['Zr']+M['R2']['S']:.3f}")

    # ---------- (a) Elemente ----------
    st.markdown("### (a) elements: measurement vs. model")
    yl = st.slider("y-Limits (Elemente)", 0., 100., (10., 65.), 0.01, key="ylim_elem_panel")

    df_panel = pd.DataFrame({
        "Ba_Zr_ratio": E["xs"],
        "Ba_norm": E["Ba_m"], "Zr_norm": E["Zr_m"], "S_norm": E["S_m"],
    })
    xlim_panel = (float(FG["Ba_Zr_ratio"].min()), float(FG["Ba_Zr_ratio"].max()))

    figA, axA = plot_element_compare_onepanel(
        df_panel,
        predA=(E["Ba_p"], E["Zr_p"], E["S_p"]),
        predB=None, ylim=yl, xlim=xlim_panel,
        figsize=(figsize_x, figsize_y), meas_grey="#7f7f7f",
        label_positions={
            "Ba": (xlim_panel[0] + 0.08*(xlim_panel[1]-xlim_panel[0]), yl[0] + 0.10*(yl[1]-yl[0])),
            "Zr": (xlim_panel[0] + 0.08*(xlim_panel[1]-xlim_panel[0]), yl[0] + 0.35*(yl[1]-yl[0])),
            "S":  (xlim_panel[0] + 0.08*(xlim_panel[1]-xlim_panel[0]), yl[0] + 0.75*(yl[1]-yl[0])),
        },
        label_colors={"Ba":"#2ca25f","Zr":"#3182bd","S":"#de2d26"},
        label_size=14, panel_tag="",
        panel_tag_xy=(0.015, 0.96), panel_tag_size=16, panel_tag_weight="normal"
    )
    if annA_on and annA_text.strip():
        add_user_annotation(axA, annA_text, (annA_x, annA_y),
            coord_system=annA_sys, fontsize=annA_fs, color=annA_col, weight=annA_wt,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            arrow_to=((annA_x2, annA_y2) if annA_arrow else None),
            arrow_kw=dict(arrowstyle="-", lw=1.0, color=annA_col))
    st.pyplot(figA, use_container_width=True)
    st.download_button("Download (a) PNG (600 dpi)", data=fig_to_png_bytes(figA, dpi=600),
                       file_name="Fig4a_elements_compare.png", mime="image/png")

    # ---------- (b) Residuen ----------
    st.markdown("### (b) residuals (measurement − model)")
    res_ylim = st.slider("y-Limits (Residuen)", -0.2, 0.2, (-0.06, 0.06), 0.01, key="ylim_res_panel")
    res = M["residuals"]
    figB, axB = plt.subplots(figsize=(figsize_x, figsize_y))
    axB.axhline(0, color='k', lw=1)
    axB.plot(res["x"], res["Ba_res"], "o", mfc="none", mec="#2ca25f", ms=5, label="Ba resid")
    axB.plot(res["x"], res["Zr_res"], "o", mfc="none", mec="#3182bd", ms=5, label="Zr resid")
    axB.plot(res["x"], res["S_res"],  "o", mfc="none", mec="#de2d26", ms=5, label="S resid")
    axB.set_xlim(*xlim_panel); axB.set_ylim(*res_ylim)
    axB.set_xlabel("Ba / (Ba + Zr)"); axB.set_ylabel("Residual (at. fraction)")
    axB.grid(True, alpha=0.25, lw=0.6); axB.legend(frameon=False, ncol=3)
    if annB_on and annB_text.strip():
        add_user_annotation(axB, annB_text, (annB_x, annB_y),
            coord_system=annB_sys, fontsize=annB_fs, color=annB_col, weight=annB_wt,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            arrow_to=((annB_x2, annB_y2) if annB_arrow else None),
            arrow_kw=dict(arrowstyle="-", lw=1.0, color=annB_col))
    st.pyplot(figB, use_container_width=True)
    st.download_button("Download (b) PNG (600 dpi)", data=fig_to_png_bytes(figB, dpi=600),
                       file_name="Fig4b_residuals.png", mime="image/png")

    # ---------- (c) Phasenfeld ----------
    st.markdown("### (c) phase diagram")
    figC, axC = plot_stacked_phases_with_grey_boundaries(
        FG, df_mark=FM, figsize=(figsize_x, figsize_y),
        xlim=(float(FG["Ba_Zr_ratio"].min()), float(FG["Ba_Zr_ratio"].max())),
        ylim=(0.0, 1.0),
        fill_colors=dict(ZrO2=col_zro2, BZS=col_bzs, B4=col_b4, B3=col_b3),
        grey_level=grey_level, b3_grey_level=b3_grey_level,
        boundary_alpha=boundary_alpha, boundary_size=boundary_size,
        decimate=decimate, dashed_all_boundaries=True,
        dashed_lw=1.0, dashed_alpha=0.9, show_guides=show_guides,
    )
    # Phase labels direkt auf axC schreiben
    xlim_c = (float(FG["Ba_Zr_ratio"].min()), float(FG["Ba_Zr_ratio"].max()))
    x_mid = (xlim_c[0] + xlim_c[1]) / 2

    # Mittelpunkte der Phasenbänder an einem repräsentativen x-Wert berechnen
    x_rep = np.array([x_mid])

    if st.session_state.get("knots_used") is not None:

        # Phasenbreiten über den gesamten Grid
        FG_s1 = FG["BaZrS3"]
        FG_s2 = FG["BaZrS3"] + FG["Ba4Zr3S10"]
        FG_s3 = FG["BaZrS3"] + FG["Ba4Zr3S10"] + FG["Ba3Zr2S7"]
        FG_x = FG["Ba_Zr_ratio"]

        label_kw = dict(fontsize=11, ha="center", va="center", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

        # BaZrS3
        width_BZS = FG["BaZrS3"]
        if width_BZS.max() > 0.04:
            ix = width_BZS.argmax()
            axC.text(float(FG_x.iloc[ix])-0.02, float(FG_s1.iloc[ix]) / 2,
                     "BaZrS₃", **label_kw)

        # Ba4Zr3S10
        width_B4 = FG["Ba4Zr3S10"]
        if width_B4.max() > 0.04:
            ix = width_B4.argmax()
            axC.text(float(FG_x.iloc[ix])-0.02,
                     float(FG_s1.iloc[ix]) + float(width_B4.iloc[ix]) / 2+0.22,
                     "Ba₄Zr₃S₁₀", **label_kw)

        # Ba3Zr2S7: ha="right" + y nach oben
        width_B3 = FG["Ba3Zr2S7"]
        if width_B3.max() > 0.04:
            ix = width_B3.argmax()
            axC.text(float(FG_x.iloc[ix]),
                     float(FG_s2.iloc[ix]) + float(width_B3.iloc[ix]) / 2 + 0.1,
                     "Ba₃Zr₂S₇",
                     **{**label_kw, "ha": "right"})

        # ZrO2: ha="left"
        width_ZR = FG["ZrO2"]
        if width_ZR.max() > 0.04:
            ix = width_ZR.argmax()
            axC.text(float(FG_x.iloc[ix])+0.01,
                     float(FG_s3.iloc[ix]) + float(width_ZR.iloc[ix]) / 2,
                     "ZrO₂",
                     **{**label_kw, "ha": "left"})


    if use_lit:
        axC.scatter([lit_x], [lit_y], marker=lit_marker, s=lit_size,
                    facecolors=("none" if lit_face == "none" else lit_face),
                    edgecolors=lit_edge, linewidths=1.8, zorder=10, label=lit_label)
        axC.text(lit_labx, lit_laby, lit_label, fontsize=9, color=lit_edge,
                 ha="center", va="top", zorder=11)
    if annC_extra_on:
        for cfg in extra_cfg:
            if cfg["text"].strip():
                add_user_annotation(axC, cfg["text"], cfg["xy"],
                    coord_system=cfg["coord"], fontsize=cfg["fontsize"],
                    color=cfg["color"], weight=cfg["weight"],
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                    arrow_to=cfg["arrow_to"],
                    arrow_kw=dict(arrowstyle="-", lw=1.0, color=cfg["color"]))
    st.pyplot(figC, use_container_width=True)
    st.download_button("Download (c) PNG (600 dpi)", data=fig_to_png_bytes(figC, dpi=600),
                       file_name="Fig4c_phase_field.png", mime="image/png")

    # ---------- Export ----------
    with st.expander("Details / Export"):
        st.write("Knoten:", M["params"])
        st.write("R²:", M["R2"])
        if "objective_value" in M:
            st.write("R²-Summe:", M["objective_value"])
        if "message" in M:
            st.caption(M["message"])
        out_df = FG.merge(
            pd.DataFrame({
                "Ba_Zr_ratio": E["xs"],
                "Ba_meas": E["Ba_m"], "Zr_meas": E["Zr_m"], "S_meas": E["S_m"]
            }),
            on="Ba_Zr_ratio", how="left"
        )
        st.download_button("Ergebnisse als CSV",
                           out_df.to_csv(index=False).encode("utf-8"),
                           file_name="phase_model_results.csv", mime="text/csv")