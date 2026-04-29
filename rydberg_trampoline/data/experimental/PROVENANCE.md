# Experimental data provenance

This directory holds digitised data points from the figures of

> Y.-X. Chao, P. Ge, Z.-X. Hua, C. Jia, X. Wang, X. Liang, Z. Yue, R. Lu,
> M. K. Tey, L. You, *Probing False Vacuum Decay and Bubble Nucleation in a
> Rydberg Atom Array*, Phys. Rev. Lett. **136**, 120407 (2026);
> arXiv:[2512.04637](https://arxiv.org/abs/2512.04637).

Until the corresponding author shares raw data, every CSV in this directory
is **digitised** from the published figure (or arXiv preprint figure) using
[WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/). Each CSV has a
sibling YAML sidecar describing:

* the source figure (paper figure and panel)
* the extraction tool and its version
* the date of extraction
* a stated accuracy claim (typically ~3–5 % pixel-level error)
* any axis scaling assumptions (linear vs log, units)

Plots that overlay these points should label them as *"digitised from
Fig. X of arXiv:2512.04637"* and never claim quantitative agreement
tighter than the digitisation accuracy.

## Adding a new digitisation

1. Open the paper figure (PDF) in WebPlotDigitizer.
2. Calibrate the axes (left, right, bottom, top reference points).
3. Mask out background gridlines, then run automatic point extraction.
4. Manually verify each extracted point against the rendered figure.
5. Export as CSV and place at `<stem>.csv` in this directory.
6. Save a sidecar `<stem>.yaml`:
   ```yaml
   source_paper: arXiv:2512.04637
   source_figure: "Figure 3 (panel a)"
   extraction_tool: WebPlotDigitizer 4.7
   extraction_date: 2026-04-28
   accuracy_pct: 5
   x_axis: "1/Delta_l (1/MHz)"
   y_axis: "Gamma (1/us)"
   notes: "log-linear plot; extracted in linear y, recomputed log on import"
   ```
7. Verify the overlay in the relevant `fig_*.py` script.

## Status

- [ ] `fig2_decay.csv` — M_AFM_res(t) traces for several Δ_l
- [ ] `fig3_gamma.csv` — Γ vs 1/Δ_l
- [ ] `fig4_bubble_hist.csv` — bubble-length distribution on/off resonance

Place TODO entries above as `[x]` once digitisation is in.
