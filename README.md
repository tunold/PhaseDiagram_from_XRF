# Ba–Zr–S Phase Model from XRF Data

A Streamlit app for modeling the phase distribution in Ba–Zr–S thin films from
X-ray fluorescence (XRF) measurements. The model decomposes elemental compositions
(Ba, Zr, S) into contributions from up to four phases — BaZrS₃, Ba₄Zr₃S₁₀,
Ba₃Zr₂S₇, and ZrO₂ — as a function of the composition variable BBZ = Ba/(Ba+Zr).

---

## Features

- **Manual fit**: interactively adjust ZrO₂ knot values and instantly visualize the phase model
- **Global fit**: automatic optimization of knot values by maximizing the weighted R² sum across all three elemental channels (Ba, Zr, S)
- **Back-projection**: modeled phase fractions are converted back to elemental fractions via the stoichiometry matrix and compared against XRF measurements
- **Three output plots**:
  - (a) Measured vs. modeled elemental fractions
  - (b) Residuals per element
  - (c) Stacked phase distribution with phase boundaries
- **Export**: download plots as 600 dpi PNG and results as CSV

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

The app will open in your browser. Upload a CSV file with XRF data or use the
built-in demo data to explore the interface.

---

## CSV Input Format

The CSV file must contain at least three columns with elemental compositions in at.%:

| Column | Description | Unit |
|--------|-------------|------|
| `Baat` | Ba content  | at.% |
| `Zrat` | Zr content  | at.% |
| `Sat`  | S content   | at.% |

The BBZ ratio (Ba/(Ba+Zr)) is calculated automatically. Column names are
configurable in the sidebar under *Column names*.

---

## Physical Model

The film is modeled as a mixture of four phases along tie-lines in the Ba–Zr–S
phase diagram:

| Phase | Stoichiometry | Region |
|-------|--------------|--------|
| BaZrS₃ | 1:1:3 | BBZ < 0.50 |
| Ba₄Zr₃S₁₀ | 4:3:10 | 0.50 ≤ BBZ < 4/7 |
| Ba₃Zr₂S₇ | 3:2:7 | BBZ ≥ 4/7 |
| ZrO₂ | secondary oxide | all regions |

The ZrO₂ fraction is described by a piecewise function:
- **Left branch** (BBZ < 0.50): linear interpolation from anchor point x = 0.40
- **Right branch** (BBZ ≥ 0.50): PCHIP interpolation through four knots at BBZ = 0.50, 0.54, 4/7, 0.60

The sulfide phase fractions are scaled by `(1 − ZrO₂)` at each composition point.

---

## File Structure

```
.
├── app.py                      # Main Streamlit application
├── stack_plotter_function.py   # Plotting helpers (phase field + element comparison)
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** `stack_plotter_function.py` must be present in the same directory as `app.py`.
> It provides `plot_stacked_phases_with_grey_boundaries` and `plot_element_compare_onepanel`.

---

## Dependencies

- [Streamlit](https://streamlit.io/) — web app framework
- [NumPy](https://numpy.org/) — numerical computations
- [pandas](https://pandas.pydata.org/) — data handling
- [Matplotlib](https://matplotlib.org/) — plotting
- [SciPy](https://scipy.org/) — PCHIP interpolation, optimization (`differential_evolution`, `minimize`)

---

## License

MIT License — Copyright (c) 2025 Thomas Unold

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
