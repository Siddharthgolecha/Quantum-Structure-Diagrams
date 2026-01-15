from contextlib import nullcontext

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle, Wedge

from .core import build_qsd_lattice
from .metrics import compute_qsd_metrics

# -------------------- Paper style (only used when style="paper") --------------------

PAPER_RC = {
    # Serif stack with a fallback that includes U+2011 (DejaVu Serif)
    "font.family": "serif",
    "font.serif": [
        "CMU Serif",
        "DejaVu Serif",
        "Times New Roman",
        "Latin Modern Roman",
    ],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 140,
    "savefig.dpi": 300,
}

_HUE_OFFSET = 2.0 / 3.0  # 0.666..., HSV hue for blue


def _rc_context(style: str):
    return (
        mpl.rc_context(PAPER_RC) if (style or "").lower() == "paper" else nullcontext()
    )


# -------------------- Text normalization (fixes U+2011 warnings) --------------------


def _normalize_text(s):
    """Replace non-breaking hyphen and long dashes with a regular hyphen for font safety."""
    if not s:
        return s
    return (
        s.replace("\u2011", "-")  # non-breaking hyphen
        .replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
    )


# ---------------- Optional Column Trimming (Preserve original indices) ----------------
def _trim_empty_columns(rows, eps=1e-12):
    """
    Remove columns with no nonzero amplitudes, preserving original column indices.
    Returns: (new_rows, kept_indices, keep_mask)
    """
    if not rows:
        return rows, [], []

    ncols = len(rows[0])
    keep_mask = []
    for c in range(ncols):
        col_has = False
        for row in rows:
            cell = row[c]
            if cell is not None and abs(cell) > eps:
                col_has = True
                break
        keep_mask.append(col_has)

    if not any(keep_mask):
        return rows, list(range(ncols)), keep_mask

    new_rows = [[cell for cell, k in zip(row, keep_mask) if k] for row in rows]
    kept_indices = [i for i, k in enumerate(keep_mask) if k]
    return new_rows, kept_indices, keep_mask


def _idx_to_ket(index: int, dims):
    total = np.prod(dims)
    if index >= total:
        return "|?⟩"  # fallback
    digits = []
    n = index
    for base in reversed(dims):
        digits.append(n % base)
        n //= base
    digits.reverse()
    return "|" + ",".join(str(d) for d in digits) + "⟩"


# ---------------------------------------------------------------------------
# Legend Helpers
# ---------------------------------------------------------------------------


def _add_phase_legend_outside(
    fig,
    ax,
    theme="dark",
    wheel_diameter_in=1.1,  # inches
    right_margin_in=0.35,  # inches
    center_text="φ",
    label_fs=9,
    center_fs=12,
):
    txt = "white" if theme == "dark" else "black"

    fig_w, _ = fig.get_size_inches()
    wheel_frac = wheel_diameter_in / fig_w
    margin_frac = right_margin_in / fig_w

    pos = ax.get_position()
    center_y = (pos.y0 + pos.y1) / 2
    wheel_left = 1.0 - wheel_frac - margin_frac
    wheel_bottom = center_y - wheel_frac / 2

    axl = fig.add_axes([wheel_left, wheel_bottom, wheel_frac, wheel_frac])
    axl.set_axis_off()
    axl.set_aspect("equal", adjustable="box")

    sat = 0.85 if (fig.get_facecolor()[0] > 0.9) else 1.0
    for deg in range(360):
        hue = (deg / 360.0 + _HUE_OFFSET) % 1.0
        color = hsv_to_rgb([hue, sat, 1.0])
        axl.add_patch(
            Wedge(
                (0.5, 0.5),
                0.48,
                deg,
                deg + 1,
                width=0.18,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
            )
        )

    axl.text(
        0.5, 0.5, center_text, color=txt, ha="center", va="center", fontsize=center_fs
    )
    axl.text(0.97, 0.50, "0", color=txt, ha="left", va="center", fontsize=label_fs)
    axl.text(
        0.50, 0.97, r"$\pi/2$", color=txt, ha="center", va="bottom", fontsize=label_fs
    )
    axl.text(
        0.03, 0.50, r"$\pi$", color=txt, ha="right", va="center", fontsize=label_fs
    )
    axl.text(
        0.50, 0.03, r"$3\pi/2$", color=txt, ha="center", va="top", fontsize=label_fs
    )
    return axl


def _add_magnitude_legend_below(fig, ring_ax, theme="dark"):
    txt = "white" if theme == "dark" else "black"
    rb = ring_ax.get_position()

    bar_w = rb.width * 0.26
    bar_h = rb.height * 0.80
    left = rb.x0 + (rb.width - bar_w) / 2.0
    bottom = rb.y0 - bar_h - 0.02
    axm = fig.add_axes([left, bottom, bar_w, bar_h])
    axm.set_axis_off()

    grad = np.linspace(0, 1, 256).reshape(256, 1)
    img = np.dstack([grad, grad, grad])
    axm.imshow(img, origin="lower", extent=[0, 1, 0, 1])

    axm.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=txt, lw=0.8)
    axm.text(1.07, 1.00, "1", color=txt, ha="left", va="center", fontsize=8)
    axm.text(1.07, 0.00, "0", color=txt, ha="left", va="center", fontsize=8)
    axm.text(0.5, -0.12, r"$|\psi|$", color=txt, ha="center", va="top", fontsize=9)


# ---------------------------------------------------------------------------
# Main Plot Function
# ---------------------------------------------------------------------------


def plot_qsd(
    psi,
    dims,
    grouping="hamming",
    ordering="lex",
    show_metrics=True,
    theme="dark",
    save_path=None,
    caption=None,
    show_legend=True,
    legend_kind="phase",
    min_height_in=3.0,
    style: str = "default",
    trim_empty: bool = None,
    annotate_basis: bool = True,
    annotate_threshold: float = 0.04,
    phase_gauge: bool = True,
    render_eps: float = 1e-12,
):

    with _rc_context(style):
        # ---------------- Lattice construction & pruning ------------------------
        EPS = float(render_eps)

        try:
            ordered, rows, index_rows, _, _ = build_qsd_lattice(
                psi,
                dims,
                grouping,
                ordering,
                return_indices=True,
                normalize_state=True,  # safe: plot is invariant to global scaling (you use gmax)
                phase_gauge=phase_gauge,  # aligns hue convention across figures
            )
            row_labels_full = list(ordered.keys())  # IMPORTANT: true group labels
        except TypeError:
            # Legacy core.py: (ordered, rows) only, no phase_gauge support
            ordered, rows = build_qsd_lattice(psi, dims, grouping, ordering)
            index_rows = None
            row_labels_full = list(ordered.keys())

        if trim_empty is None:
            trim_empty = len(dims) > 2

        # keep only rows with any visible amplitude
        keep_row_mask = [
            any((amp is not None) and (abs(amp) > EPS) for amp in row) for row in rows
        ]
        kept_row_labels = [
            row_labels_full[i] for i, keep in enumerate(keep_row_mask) if keep
        ]
        rows = [row for row, keep in zip(rows, keep_row_mask) if keep]

        if index_rows is None:
            index_rows = None
        else:
            index_rows = [ir for ir, keep in zip(index_rows, keep_row_mask) if keep]

        if not rows:
            rows = [[]]
            kept_row_labels = [0]
            index_rows = [[]] if index_rows is not None else None

        max_len = max(len(r) for r in rows)
        rows = [r + [None] * (max_len - len(r)) for r in rows]

        if index_rows is None:
            index_rows = [[None] * max_len for _ in range(len(rows))]
        else:
            index_rows = [ir + [None] * (max_len - len(ir)) for ir in index_rows]

        if trim_empty:
            rows, kept_indices, keep_mask = _trim_empty_columns(rows, EPS)
            index_rows = [
                [cell for cell, k in zip(ir, keep_mask) if k] for ir in index_rows
            ]
        else:
            kept_indices = list(range(max_len))
            keep_mask = [True] * max_len

        nrows = len(rows)
        ncols = len(rows[0]) if rows else 0

        # ---------------- Base Figure Setup -------------------------------------
        fig_w = max(ncols * 0.8 + 1.4, 3.8)
        fig_h = max(nrows * 0.8 + 1.6, min_height_in)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)

        # Label first, because we will measure label height later:
        ax.set_xlabel("Basis index within group")
        ax.set_ylabel("Group label (layer)")

        label_color = "white" if theme == "dark" else "black"
        bg_color = "black" if theme == "dark" else "white"
        grid_color = "white" if theme == "dark" else "black"
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=label_color)
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
        for s in ax.spines.values():
            s.set_color(label_color)

        # ---------------- Draw Cells -------------------------------------------
        # gmax per LaTeX: max_j |c_j| (within-figure normalization)
        flat_amps = [amp for row in rows for amp in row if amp is not None]
        gmax = float(np.max(np.abs(flat_amps))) if flat_amps else 1.0
        if gmax <= 0:
            gmax = 1.0
        for r, row in enumerate(rows):
            for c, amp in enumerate(row):
                if amp is None or abs(amp) <= EPS:
                    continue

                mag = abs(amp) / gmax
                phi = np.angle(amp)
                hue = ((phi / (2 * np.pi)) + _HUE_OFFSET) % 1.0

                # Adjust saturation/value for paper style
                if (style or "").lower() == "paper":
                    rgb = hsv_to_rgb([hue, mag * 0.85, mag * 0.95])
                else:
                    rgb = hsv_to_rgb([hue, mag, mag])

                ax.add_patch(Rectangle((c, r), 1, 1, color=rgb, linewidth=0))

                # --- Annotate basis label (if indices available & amplitude big enough)
                if annotate_basis and index_rows and index_rows[r][c] is not None:
                    if mag >= annotate_threshold:
                        basis_idx = index_rows[r][c]
                        label = _idx_to_ket(basis_idx, dims)
                        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                        txt_color = "black" if lum > 0.55 else "white"

                        fontsize = 9 if (style or "").lower() == "paper" else 10
                        ax.text(
                            c + 0.5,
                            r + 0.5,
                            label,
                            ha="center",
                            va="center",
                            color=txt_color,
                            fontsize=fontsize,
                            fontweight="semibold",
                            alpha=0.92 if (style or "").lower() == "paper" else 1.0,
                            clip_on=True,
                            # Add a thin outline for robustness on saturated colors
                            path_effects=[
                                pe.withStroke(
                                    linewidth=2.2,
                                    foreground=(
                                        "black" if txt_color == "white" else "white"
                                    ),
                                    alpha=0.75,
                                )
                            ],
                        )

        ax.set_aspect("equal")
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_xticks(np.arange(ncols) + 0.5)
        ax.set_yticks(np.arange(nrows) + 0.5)
        ax.set_xticklabels([str(i) for i in kept_indices])
        # y labels: preserve the ORIGINAL row indices (e.g., 0 and 3 for GHZ(3))
        ax.set_yticklabels([str(lbl) for lbl in kept_row_labels])

        ax.set_xticks(np.arange(ncols + 1), minor=True)
        ax.set_yticks(np.arange(nrows + 1), minor=True)
        ax.grid(which="minor", color=grid_color, alpha=0.08, linewidth=0.4)
        ax.tick_params(which="minor", length=0)

        # ---------------- Reserve Space Layout ----------------------------------
        legend_needed_in = 1.2
        current_fig_w = fig.get_size_inches()[0]

        if show_legend and legend_kind != "none":
            if current_fig_w - (fig_w - legend_needed_in) < legend_needed_in:
                new_fig_w = fig_w + legend_needed_in
                fig.set_size_inches(new_fig_w, fig.get_size_inches()[1])
                fig.canvas.draw()

        title_band = 0.12
        caption_band = 0.16 if caption else 0.00

        legend_width_in = 1.4
        fig_w, fig_h = fig.get_size_inches()
        legend_frac = legend_width_in / fig_w
        plot_right = 1.0 - legend_frac

        fig.subplots_adjust(
            left=0.16,
            right=plot_right,
            top=1.0 - title_band,
            bottom=0.14 + caption_band,
        )

        # ---------------- Top Title Band (Metrics) ------------------------------
        title_ax = fig.add_axes(
            [
                ax.get_position().x0,
                1.0 - title_band + 0.01,
                ax.get_position().width,
                title_band - 0.03,
            ]
        )
        title_ax.set_axis_off()

        if show_metrics:
            try:
                m = compute_qsd_metrics(psi, dims, grouping=grouping, ordering=ordering)
                D = m["row_delocalization"]
                E = m["bipartite_entanglement_linear"]["value"]
                rank1 = m.get("aligned_separability_test", {}).get("rank_one", None)

                if rank1 is None:
                    title_text = f"D_row: {D:.3f} |  E_lin(A:B): {E:.3f}"
                else:
                    title_text = f"D_row: {D:.3f} |  E_lin(A:B): {E:.3f} |  rank-1(A:B): {bool(rank1)}"
            except Exception:
                title_text = "Metrics unavailable for this configuration"

            # normalize to avoid U+2011 warnings
            title_text = _normalize_text(title_text)
            title_ax.text(
                0.5,
                0.5,
                title_text,
                ha="center",
                va="center",
                fontsize=11,
                color=label_color,
            )

        # ---------------- Caption Band (bottom anchored) ------------------------
        if caption:
            fig.canvas.draw()
            axpos = ax.get_position()
            cap_ax = fig.add_axes(
                [axpos.x0, 0.04, axpos.width, caption_band - 0.04]  # bottom anchored
            )
            cap_ax.set_axis_off()
            cap_ax.text(
                0.5,
                0.5,
                _normalize_text(caption),
                ha="center",
                va="center",
                fontsize=10,
                color=label_color,
            )

        # ---------------- Auto-resize for y-label clearance ---------------------
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ylab = ax.yaxis.label.get_window_extent(renderer=renderer)
        ylab_in = ylab.height / fig.dpi + 0.25

        usable_axes_height = (
            1.0 - title_band - (0.14 + caption_band)
        ) * fig.get_size_inches()[1]
        if usable_axes_height < ylab_in:
            scale = ylab_in / usable_axes_height
            new_h = fig.get_size_inches()[1] * scale
            fig.set_size_inches(fig.get_size_inches()[0], new_h)
            fig.canvas.draw()

        # ---------------- Legend Outside-Right ----------------------------------
        if show_legend and legend_kind in {"phase", "both"}:
            ring_ax = _add_phase_legend_outside(
                fig,
                ax,
                theme=theme,
                wheel_diameter_in=1.1,
                right_margin_in=0.1,
            )
            if legend_kind == "both":
                _add_magnitude_legend_below(fig, ring_ax, theme=theme)

        # ---------------- Save or Show -----------------------------------------
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=bg_color)
            plt.close(fig)
        else:
            plt.show()


def analyze_and_plot_qsd(psi, dims, **kwargs):
    plot_qsd(psi, dims, **kwargs)
    grouping = kwargs.get("grouping", "hamming")
    ordering = kwargs.get("ordering", "lex")
    return compute_qsd_metrics(psi, dims, grouping=grouping, ordering=ordering)
