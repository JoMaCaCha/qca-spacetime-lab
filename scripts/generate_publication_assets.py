#!/usr/bin/env python3
"""
Generate publication-ready multi-panel vector figures (PDF, SVG, 300-DPI PNG) and
ready-to-compile LaTeX tables from the numerical datasets exported by `qsu_cli`
in `datos_investigacion/`.

Outputs in `figuras_publicacion/`:
  - fig_via_a.{pdf,svg,png} : Pathway A (3+1D QCA n=4, 4+1D Kaplan-Sen chiral slab, Yee-4 S4 real-space isotropy & birefringence)
  - fig_via_b.{pdf,svg,png} : Pathway B (2D Causal Sets 20-rep adaptive FSS N=20..64, Wang-Landau 1st-order barrier, mass & large-Nf sweep)
  - fig_via_c.{pdf,svg,png} : Pathway C (2-Flavor Schwinger continuum trajectory, Calabrese-Cardy c:2->1 up to L=64, 2-site TDVP & BKT Neel phase diagram)
  - tablas_latex.tex        : Ready-to-include booktabs LaTeX tables for all three manuscripts.
"""

from __future__ import annotations
import csv
import json
import math
import pathlib
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "datos_investigacion"
OUT = ROOT / "figuras_publicacion"

QCA_LABEL_MAP: dict[str, str] = {
    "Fermion-IsoSCFCC-S2": "Ferm S2",
    "Fermion-IsoSCFCC-S4": "Ferm S4\n(n=4)",
    "Mlodinow-Brun-2025": "MB 2025",
    "Bisio-DAriano-Perinotti-2016": "Bisio 16",
    "Eon-et-al-Fijo-2023": "Eon Fijo",
    "Eon-et-al-Palindromico-2023": "Eon Pal",
    "Yee4-S4-Nuevo": "Yee-4 S4\n(n=4)",
}

CHIRAL_LABEL_MAP: dict[str, str] = {
    "SC-Kaplan-Sen": "SC\nestandar",
    "Iso-SC+FCC-Kaplan-Sen": "Iso-SC+FCC\n(isotropo)",
    "Symanzik4-Kaplan-Sen": "Symanzik-4\n(n=4 Weyl)",
}

_LATEX_SPECIAL_RE = re.compile(r"[\\{}%&#$_~^]")
_LATEX_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape all LaTeX special characters in a single pass."""
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_ESCAPES[m.group(0)], text)


def load_csv(name: str) -> list[dict[str, str]]:
    p = DATA / name
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset {p}. Run `cargo run --release --bin qsu_cli -- phase18/phase19/phase20` first.")
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(name: str) -> dict:
    p = DATA / name
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset {p}.")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")


def make_fig_via_a() -> dict:
    qca_rows = load_csv("via_a_qca_n4.csv")
    chiral_rows = load_csv("via_a_chiral_kaplan_sen_n4.csv")
    yee4 = load_json("via_a_coupled_yee4.json")
    if not qca_rows or not chiral_rows:
        raise ValueError("Empty Pathway A dataset(s) in datos_investigacion/")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), constrained_layout=True)
    try:
        # Panel (a): Unified audit of QCA / photon models mapped by row key
        ax = axes[0]
        short_labels = [QCA_LABEL_MAP.get(r["modelo"], r["modelo"]) for r in qca_rows]
        x = np.arange(len(qca_rows))
        cpt = [max(float(r["cpt_asym"]), 1e-16) for r in qca_rows]
        spr = [max(float(r["a2_spread"]), 1e-16) for r in qca_rows]
        w = 0.36
        ax.bar(x - w / 2, np.log10(cpt), w, label=r"$\log_{10}$ asim. CPT ($n=1$)", color="#2b6cb0")
        ax.bar(x + w / 2, np.log10(spr), w, label=r"$\log_{10}$ spread $A_2$ ($n=2$)", color="#dd6b20")
        ax.axhline(-12, color="#2b6cb0", ls=":", lw=1.0, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=7.6)
        ax.set_ylabel(r"$\log_{10}(\mathrm{error\ espectral})$")
        ax.set_title("(a) Auditoria 3+1D: Fermiones y Fotones")
        ax.set_ylim(-16.5, 0.5)
        ax.legend(loc="lower center", framealpha=0.95)

        # Panel (b): 4+1D Kaplan-Sen chiral slab mapped by row key
        ax = axes[1]
        c_labels = [CHIRAL_LABEL_MAP.get(r["modelo"], r["modelo"]) for r in chiral_rows]
        xc = np.arange(len(chiral_rows))
        c_spr = [max(float(r["a2_spread"]), 1e-8) for r in chiral_rows]
        c_mean = [max(abs(float(r["a2_mean"])), 1e-8) for r in chiral_rows]
        ax.bar(xc - 0.18, np.log10(c_spr), 0.34, label=r"$\log_{10}\,\mathrm{spread}(A_2)$", color="#dd6b20")
        ax.bar(xc + 0.18, np.log10(c_mean), 0.34, label=r"$\log_{10}\,|\langle A_2\rangle|$", color="#2f855a")
        ax.set_xticks(xc)
        ax.set_xticklabels(c_labels, fontsize=8.3)
        ax.set_ylabel(r"$\log_{10}(\mathrm{error\ de\ cono\ Weyl\ }O(k^2))$")
        ax.set_title("(b) Losa quiral 4+1D Kaplan-Sen")
        ax.set_ylim(-6.0, 0.2)
        ax.legend(loc="lower left", framealpha=0.92)
        for i, r in enumerate(chiral_rows):
            ax.text(i, -0.35, f"gap={float(r['corner_gap']):.2f}", ha="center", fontsize=7.6, color="#333")

        # Panel (c): Real-space 16^3 Yee-2 vs Yee-4 S4 isotropy & 45-deg birefringence vs Mlodinow-Brun
        ax = axes[2]
        aniso_y2 = abs(float(yee4["c_dg_yee2"]) - float(yee4["c_ax_yee2"]))
        aniso_y4 = abs(float(yee4["c_dg_yee4"]) - float(yee4["c_ax_yee4"]))
        biref_mb = float(yee4["pol_rot_mlodinow_brun"])
        biref_y4 = float(yee4["pol_leak_yee4_45deg"])
        cats = ["Anisotropia\n|c_dg - c_ax|", "Birrefringencia\n45 deg (T=10)"]
        x3 = np.arange(2)
        ax.bar(x3 - 0.18, [max(aniso_y2, 1e-12), max(biref_mb, 1e-12)], 0.34, label="Yee-2 / MB 2025", color="#c53030")
        ax.bar(x3 + 0.18, [max(aniso_y4, 1e-12), max(biref_y4, 1e-12)], 0.34, label="Nuevo Yee-4 S4 (real 16^3)", color="#2f855a")
        ax.set_yscale("log")
        ax.set_xticks(x3)
        ax.set_xticklabels(cats, fontsize=8.4)
        ax.set_ylabel("Error en espacio real (escala log)")
        ax.set_title("(c) Propagacion 3D real 16^3: Yee-4 S4")
        ax.legend(loc="upper left", framealpha=0.92)
        ax.text(0, max(aniso_y2, 1e-4) * 1.6, f"{float(yee4['aniso_improvement_yee2_over_yee4']):.0f}x menor", ha="center", fontsize=8.0, fontweight="bold", color="#22543d")
        ax.text(1, max(biref_mb, 1e-4) * 1.6, f"{float(yee4['biref_suppression_ratio']):.0f}x menor", ha="center", fontsize=8.0, fontweight="bold", color="#22543d")
        ax.set_ylim(2e-7, 6.0)

        save_fig(fig, "fig_via_a")
    finally:
        plt.close(fig)
    return {"qca": qca_rows, "chiral": chiral_rows, "yee4": yee4}


def make_fig_via_b() -> dict:
    fss_sum = load_csv("via_b_fss_20rep_summary.csv")
    fss_fits = load_json("via_b_fss_fits.json")
    wl_rows = load_csv("via_b_wl_multicanonical.csv")
    sw_rows = load_csv("via_b_matter_mass_nf.csv")

    fits_by_mode = {f["mode"]: f for f in fss_fits.get("fits", [])}

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    try:
        # Panel (a): Finite-size scaling of d_MM(beta=0) vs 1/N (N = 20..64)
        ax = axes[0]
        styles = {
            "Puro(Nf=0)": ("#2b6cb0", "o", "-"),
            "SJ(Nf=1)": ("#2f855a", "s", "-"),
            "SJ(Nf=2)": ("#dd6b20", "^", "-"),
            "Ising(J=0.45)": ("#805ad5", "D", "--"),
        }
        for mode, (col, mk, ls) in styles.items():
            rows_m = [r for r in fss_sum if r["matter_mode"] == mode]
            if not rows_m:
                raise ValueError(f"Missing FSS summary rows for matter_mode={mode!r}")
            fit = fits_by_mode.get(mode)
            if fit is None:
                raise KeyError(f"Missing FSS fit entry for mode={mode!r} in via_b_fss_fits.json")
            inv_n = np.array([1.0 / float(r["N"]) for r in rows_m])
            dmm = np.array([float(r["dmm_b0"]) for r in rows_m])
            err = np.array([float(r["dmm_b0_err"]) for r in rows_m])
            ax.errorbar(inv_n, dmm, yerr=err, fmt=mk, color=col, capsize=3, ms=5.5,
                        label=f"{mode}: $d_\\infty={fit['dmm_inf']:.3f}$")
            xs = np.linspace(0.0, 0.055, 40)
            ax.plot(xs, fit["dmm_inf"] + fit["dmm_slope"] * xs, color=col, ls=ls, lw=1.3, alpha=0.85)
        ax.axhline(2.0, color="#111", ls=":", lw=1.0)
        ax.set_xlabel(r"$1 / N$  ($N \in \{20, 28, 36, 48, 64\}$)")
        ax.set_ylabel(r"Dimension Myrheim-Meyer $d_{\mathrm{MM}}(\beta=0)$")
        ax.set_title("(a) FSS adaptativo 20 replicas ($N \\to \\infty$)")
        ax.set_xlim(0.0, 0.055)
        ax.legend(loc="lower left", framealpha=0.92, fontsize=7.8)

        # Panel (b): Wang-Landau multicanonical chi_peak & free-energy barrier Delta F vs N
        ax = axes[1]
        for nf, col, mk, lbl in [(0, "#2b6cb0", "o", "Puro (Nf=0)"), (1, "#2f855a", "s", "SJ (Nf=1)")]:
            sub = sorted([r for r in wl_rows if int(r["Nf"]) == nf], key=lambda r: int(r["N"]))
            if not sub:
                raise ValueError(f"Missing Wang-Landau rows for Nf={nf}")
            ns = [int(r["N"]) for r in sub]
            chi = [float(r["chi_peak_wl"]) for r in sub]
            ax.plot(ns, chi, marker=mk, color=col, lw=1.8, ms=6, label=f"{lbl}: $\\chi_{{\\mathrm{{pico}}}}$ (WL)")
        ax.set_xlabel("Tamano del conjunto causal $N$")
        ax.set_ylabel(r"Susceptibilidad $\chi_S(\beta_c)$ (Wang-Landau)")
        ax.set_title("(b) Barrera 1.er orden (Wang-Landau)")
        ax2 = ax.twinx()
        for nf, col, ls, lbl in [(0, "#c53030", "--", "Puro: $\\Delta F$"), (1, "#dd6b20", ":", "SJ1: $\\Delta F$")]:
            sub = sorted([r for r in wl_rows if int(r["Nf"]) == nf], key=lambda r: int(r["N"]))
            ns = [int(r["N"]) for r in sub]
            df = [float(r["barrier_df"]) for r in sub]
            ax2.plot(ns, df, color=col, ls=ls, marker="^", ms=5, lw=1.4, label=lbl)
        ax2.set_ylabel(r"Barrera energia libre $\Delta F(\beta_c)$", color="#9b2c2c")
        ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.92, fontsize=7.6)

        # Panel (c): Mass m sweep (Nf=1) & Large-Nf sweep (m=0.80) at beta=0
        ax = axes[2]
        nf_sub = sorted([r for r in sw_rows if abs(float(r["m_scalar"]) - 0.80) < 1e-6], key=lambda r: int(r["Nf"]))
        if not nf_sub:
            raise ValueError("Missing m_scalar=0.80 sweep rows in via_b_matter_mass_nf.csv")
        nfs = [int(r["Nf"]) for r in nf_sub]
        d_nf = [float(r["dmm_mean"]) for r in nf_sub]
        e_nf = [float(r["dmm_err"]) for r in nf_sub]
        ax.errorbar(nfs, d_nf, yerr=e_nf, fmt="o-", color="#2f855a", capsize=3.5, lw=1.8, ms=6,
                    label=r"Barrido $N_f$ ($m=0{,}80$, $N=36$)")
        # Exclude (Nf=1, m=0.80) from scatter overlay so it is not plotted twice
        m_all = [r for r in sw_rows if int(r["Nf"]) == 1]
        m_extra = [r for r in m_all if abs(float(r["m_scalar"]) - 0.80) >= 1e-6]
        if m_extra:
            m_all_vals = [float(r["m_scalar"]) for r in m_all]
            d_m = [float(r["dmm_mean"]) for r in m_extra]
            ax.scatter([1.0] * len(d_m), d_m, color="#dd6b20", marker="x", s=45, zorder=5,
                       label=f"Otras masas $m \\in [{min(m_all_vals):.2f},{max(m_all_vals):.2f}]$ ($N_f=1$)")
        ax.axhline(2.0, color="#111", ls=":", lw=1.0, label="Variedad plana 2D ($d=2$)")
        ax.set_xlabel(r"Numero de campos cuanticos SJ $N_f$ (en $\beta=0$)")
        ax.set_ylabel(r"Dimension efectiva $d_{\mathrm{MM}}$")
        ax.set_title(r"(c) Transicion a gran $N_f$ y limite $m \to 0$")
        ax.legend(loc="upper left", framealpha=0.92, fontsize=7.8)

        save_fig(fig, "fig_via_b")
    finally:
        plt.close(fig)
    return {"fss_sum": fss_sum, "fss_fits": fss_fits, "wl": wl_rows, "sweep": sw_rows}


def make_fig_via_c() -> dict:
    cont_rows = load_csv("via_c_continuum_traj.csv")
    cft_rows = load_csv("via_c_cft_scaling.csv")
    bkt_rows = load_csv("via_c_tdvp_bkt.csv")
    tdvp = load_json("via_c_tdvp_resumen.json")
    sc_res = load_json("via_c_scaling_resumen.json")
    if len(cont_rows) < 2 or not cft_rows or not bkt_rows:
        raise ValueError("Incomplete Pathway C dataset(s) in datos_investigacion/")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    try:
        # Panel (a): Continuum trajectory v_IR(N_r -> inf) -> sqrt(v_A v_B) via exact OLS fit
        ax = axes[0]
        inv_nr = np.array([1.0 / float(r["N_r"]) for r in cont_rows], dtype=np.float64)
        v_ir = np.array([0.5 * (float(r["vA_IR"]) + float(r["vB_IR"])) for r in cont_rows], dtype=np.float64)
        v_geom = float(sc_res["v_geom"])
        ax.plot(inv_nr, v_ir, "o-", color="#2b6cb0", ms=6.5, lw=1.8, label=r"Lanczos exacto $v_A^{\mathrm{IR}} = v_B^{\mathrm{IR}}$")
        xs = np.linspace(0.0, 0.18, 50)
        slope, intercept = np.polyfit(inv_nr, v_ir, 1)
        assert abs(intercept - float(sc_res["v_inf_lin"])) < 1e-5, (
            f"OLS intercept {intercept:.6f} disagrees with via_c_scaling_resumen.json v_inf_lin={sc_res['v_inf_lin']:.6f}"
        )
        ax.plot(xs, intercept + slope * xs, "--", color="#2b6cb0", lw=1.2, alpha=0.8,
                label=f"Extrapolacion OLS $v_\\infty = {intercept:.4f}$")
        ax.axhline(v_geom, color="#2f855a", ls=":", lw=1.6, label=rf"Media geometrica $\sqrt{{v_A v_B}} = {v_geom:.4f}$")
        ax.scatter([0.0], [v_geom], color="#2f855a", s=55, zorder=5)
        ax.set_xlabel(r"$1 / N_r$  ($M_S L = \mathrm{const},\ J = 8/N_r^2$)")
        ax.set_ylabel(r"Velocidad infrarroja sincronizada $v_{\mathrm{IR}}$")
        ax.set_title("(a) Limite de continuo ($U(1)_A \\times U(1)_B$ exacto)")
        ax.set_xlim(-0.008, 0.185)
        ax.set_ylim(0.50, 0.70)
        ax.legend(loc="lower left", framealpha=0.92, fontsize=7.8)

        # Panel (b): Calabrese-Cardy multi-size scaling up to L=64 (with explicit finite-chi marker at L=64)
        ax = axes[1]
        lns = np.array([math.log(float(r["L_sites"])) for r in cft_rows], dtype=np.float64)
        s0 = np.array([float(r["S_J0_exact"]) for r in cft_rows], dtype=np.float64)
        s25_sym = np.array([float(r["S_J25_sym_c1"]) for r in cft_rows], dtype=np.float64)
        s1_asym = np.array([float(r["S_J1_asym"]) for r in cft_rows], dtype=np.float64)
        s25_neel = np.array([float(r["S_J25_asym_neel"]) for r in cft_rows], dtype=np.float64)
        ax.plot(lns, s0, "o-", color="#2b6cb0", ms=5.5, lw=1.8, label=r"$J=0$ exacto ($c \to 2$)")
        ax.plot(lns, s1_asym, "s-", color="#2f855a", ms=5.5, lw=1.7, label=r"$J=1{,}0,\ v_B/v_A=0{,}45$ ($c \to 1$)")
        ax.plot(lns[:4], s25_sym[:4], "^--", color="#dd6b20", ms=5.5, lw=1.6,
                label=r"$J=2{,}5,\ \Delta_{\mathrm{ef}}=1$ ($L \leq 48$, $c \to 1$)")
        if len(lns) > 4:
            ax.plot([lns[4]], [s25_sym[4]], marker="^", mfc="none", mec="#dd6b20", mew=1.6, ms=7.2, ls="none",
                    label=r"$L=64$ ($\Delta_{\mathrm{ef}}=1$, saturacion $\chi$)")
            ax.annotate(
                r"$L=64$ ($\chi$ finita)",
                xy=(lns[4], s25_sym[4]),
                xytext=(lns[4] - 0.58, s25_sym[4] - 0.14),
                fontsize=7.0,
                color="#9c4221",
                arrowprops=dict(arrowstyle="->", color="#dd6b20", lw=0.9),
            )
        ax.plot(lns, s25_neel, "D:", color="#c53030", ms=5.2, lw=1.6, label=r"$J=2{,}5,\ \Delta_{\mathrm{ef}}=1{,}34$ (BKT)")
        ax.set_xlabel(r"$\ln L$  ($L = 2 N_r \in \{16, 24, 32, 48, 64\}$)")
        ax.set_ylabel(r"Entropia central $S_{\mathrm{vN}}(L/2)$")
        ax.set_title("(b) Escalado Calabrese-Cardy U(1)-DMRG")
        ax.legend(loc="upper left", framealpha=0.92, fontsize=7.2)

        # Panel (c): BKT Neel phase diagram C_Neel(d=4) & c_eff vs Delta_eff + TDVP inset summary
        ax = axes[2]
        for jg, col, mk, ls in [(1.0, "#2b6cb0", "o", "--"), (2.5, "#c53030", "s", "-")]:
            sub = sorted([r for r in bkt_rows if abs(float(r["J"]) - jg) < 1e-4], key=lambda r: float(r["Delta_eff"]))
            if not sub:
                raise ValueError(f"Missing BKT phase diagram rows for J={jg}")
            de = [float(r["Delta_eff"]) for r in sub]
            cn = [float(r["C_neel_d4"]) for r in sub]
            ax.plot(de, cn, marker=mk, color=col, ls=ls, lw=1.8, ms=6, label=rf"$C_{{\mathrm{{Neel}}}}(d=4),\ J={jg:.1f}$")
        ax.set_xlabel(r"Anisotropia Schrieffer-Wolff $\Delta_{\mathrm{ef}} = \frac{v_A^2+v_B^2}{2 v_A v_B}$")
        ax.set_ylabel(r"Correlador alternado $C_{\mathrm{Neel}}(d=4)$")
        ax.set_title("(c) Transicion BKT de Neel + 2-site TDVP")
        ax.axvline(1.0, color="#111", ls=":", lw=1.0)
        tdvp_txt = (
            f"2-site U(1)-TDVP (L={int(tdvp['l_sites'])}, 2^{int(tdvp['l_sites'])})\n"
            f"J=0: mis={tdvp['mismatch_LC_J0']:.2f}  ->  J=0.5: mis={tdvp['mismatch_LC_J050']:.2f}\n"
            f"drift E = {tdvp['max_energy_drift']:.1e}"
        )
        ax.text(0.97, 0.06, tdvp_txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.4, bbox=dict(boxstyle="round,pad=0.35", facecolor="#edf2f7", edgecolor="#a0aec0"))
        ax.legend(loc="upper left", framealpha=0.92, fontsize=7.8)

        save_fig(fig, "fig_via_c")
    finally:
        plt.close(fig)
    return {"cont": cont_rows, "cft": cft_rows, "bkt": bkt_rows, "tdvp": tdvp, "scaling": sc_res}


def write_latex_tables(da: dict, db: dict, dc: dict) -> None:
    lines: list[str] = [
        "% Tablas LaTeX generadas automaticamente por scripts/generate_publication_assets.py",
        "% Requiere \\usepackage{booktabs,amsmath} en el preambulo.",
        "",
        "% ======================== TABLA VIA A ========================",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Auditoria espectral unificada 3+1D (fermiones QCA locales $S_2$ vs $S_4$, fotones publicados y nuevo foton gauge-invariante Yee-4 $S_4$) y extension quiral 4+1D de Kaplan--Sen.}",
        "\\label{tab:via_a_audit}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Modelo & Asim.\\ CPT ($n=1$) & $\\mathrm{spread}(A_2)$ & $\\langle A_2\\rangle$ & Hueco BZ & $n=4$ \\\\",
        "\\midrule",
    ]
    for r in da["qca"]:
        nm = escape_latex(r["modelo"])
        p4 = "Si" if int(float(r["pass_n4"])) == 1 else "No"
        lines.append(
            f"{nm} & ${float(r['cpt_asym']):.2e}$ & ${float(r['a2_spread']):.2e}$ & ${float(r['a2_mean']):+.2e}$ & ${float(r['corner_gap']):.3f}$ & {p4} \\\\"
        )
    lines.extend([
        "\\midrule",
        "\\multicolumn{6}{l}{\\textit{Extension quiral 4+1D en losa abierta Kaplan--Sen ($L_s=14,\\ m=0.55$, suma $\\langle\\gamma^5\\rangle_L+\\langle\\gamma^5\\rangle_R \\approx 0$):}} \\\\",
    ])
    tier_labels = {0: "No", 1: "Iso", 2: "Si ($n=4$)"}
    for r in da["chiral"]:
        nm = escape_latex(r["modelo"])
        tier_code = int(float(r["tier"]))
        if tier_code not in tier_labels:
            raise ValueError(f"Unknown chiral tier code {tier_code!r} for model {r['modelo']!r}")
        tier = tier_labels[tier_code]
        lines.append(
            f"{nm} & ${abs(float(r['net_chiral'])):.2e}$ & ${float(r['a2_spread']):.2e}$ & ${float(r['a2_mean']):+.2e}$ & ${float(r['corner_gap']):.3f}$ & {tier} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% ======================== TABLA VIA B ========================",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Conjuntos causales 2D acoplados a materia: extrapolacion FSS adaptativa ($20$ replicas, $N=20\\text{--}64$) y caracterizacion multicanonica Wang--Landau de la transicion de primer orden de Surya.}",
        "\\label{tab:via_b_wl_fss}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Modelo / Corrida WL & $N$ & $\\beta_c$ & $\\chi_{\\mathrm{pico}}$ & $\\Delta s$ & $\\Delta F$ & $d_{\\mathrm{MM}}(\\beta=0)$ \\\\",
        "\\midrule",
    ])
    for r in db["wl"]:
        lbl = "Puro ($N_f=0$)" if int(r["Nf"]) == 0 else f"SJ ($N_f={int(r['Nf'])}$)"
        lines.append(
            f"{lbl} & {int(r['N'])} & ${float(r['beta_c_wl']):.3f}$ & ${float(r['chi_peak_wl']):.1f}$ & ${float(r['latent_heat']):.2f}$ & ${float(r['barrier_df']):.2f}$ & ${float(r['dmm_b0']):.3f}$ \\\\"
        )
    lines.extend([
        "\\midrule",
        "\\multicolumn{7}{l}{\\textit{Ajustes FSS adaptativos $N \\in \\{20,28,36,48,64\\}$ ($d_{\\mathrm{MM}}(N) = d_\\infty + b/N$):}} \\\\",
    ])
    for f in db["fss_fits"]["fits"]:
        nm = escape_latex(f["mode"])
        lines.append(
            f"{nm} & $\\infty$ & ${f['beta_c_inf']:.4f}$ & $\\gamma={f['gamma_chi']:.2f}$ & -- & $B_{{4,\\min}}={f['binder_min_64']:+.2f}$ & ${f['dmm_inf']:.4f}$ \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% ======================== TABLA VIA C ========================",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Modelo de Schwinger 1+1D de dos sabores ($v_A=1, v_B\\neq v_A$, $U(1)_A\\times U(1)_B$ exacto): diagrama de fases BKT de Neel ($N_r=16\\to 24$) y evolucion en tiempo real 2-site $U(1)$-TDVP ($L=32$, $2^{32}$ estados).}",
        "\\label{tab:via_c_bkt_tdvp}",
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "$v_B / v_A$ & $J$ & $\\Delta_{\\mathrm{ef}}$ & $S_{\\mathrm{vN}}(N_r=16)$ & $c_{\\mathrm{eff}}(16\\to 24)$ & $C_{\\mathrm{Neel}}(d=4)$ \\\\",
        "\\midrule",
    ])
    for r in dc["bkt"]:
        lines.append(
            f"${float(r['vB_over_vA']):.2f}$ & ${float(r['J']):.2f}$ & ${float(r['Delta_eff']):.3f}$ & ${float(r['S16']):.4f}$ & ${float(r['c_eff_16_24']):+.3f}$ & ${float(r['C_neel_d4']):+.4f}$ \\\\"
        )
    td = dc["tdvp"]
    lines.extend([
        "\\midrule",
        f"\\multicolumn{{6}}{{l}}{{2-site $U(1)$-TDVP ($L={int(td['l_sites'])}$, $\\chi={int(td['chi_tdvp'])}$): $J=0\\ (v_A={td['vA_LC_J0']:.3f},\\ v_B={td['vB_LC_J0']:.3f},\\ \\mathrm{{mis}}={td['mismatch_LC_J0']:.3f}) \\to J=0.50\\ (v_A={td['vA_LC_J050']:.3f},\\ v_B={td['vB_LC_J050']:.3f},\\ \\mathrm{{mis}}={td['mismatch_LC_J050']:.3f})$, $\\delta E={td['max_energy_drift']:.2e}$.}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tablas_latex.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    set_pub_style()
    da = make_fig_via_a()
    db = make_fig_via_b()
    dc = make_fig_via_c()
    write_latex_tables(da, db, dc)
    print(f"Generated publication figures (PDF, SVG, PNG) and tablas_latex.tex in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
