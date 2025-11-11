from contextlib import nullcontext

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle, Wedge

from .core import build_hld_lattice
from .metrics import compute_hld_metrics

# -------------------- Paper style (only used when style="paper") --------------------

PAPER_RC = {
    # Serif stack with a fallback that includes U+2011 (DejaVu Serif)
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman", "Latin Modern Roman"],
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

def _rc_context(style: str):
    return mpl.rc_context(PAPER_RC) if (style or "").lower() == "paper" else nullcontext()


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
    Remove columns with no nonzero amplitudes, but preserve the *original*
    column indices in the tick labels.
    """
    ncols = len(rows[0])
    # Determine which columns have any amplitude
    keep = [
        any((cell is not None) and (abs(cell) > eps) for row in rows for cell in [row[c]])
        for c in range(ncols)
    ]

    # If all empty or already dense, return unchanged
    if not any(keep):
        return rows, list(range(ncols))

    # Filter rows
    new_rows = [[cell for cell, k in zip(row, keep) if k] for row in rows]

    # Preserve original indices for axis tick labels
    kept_indices = [i for i, k in enumerate(keep) if k]
    return new_rows, kept_indices


# ---------------------------------------------------------------------------
# Legend Helpers
# ---------------------------------------------------------------------------

def _add_phase_legend_outside(
    fig,
    ax,
    theme="dark",
    wheel_diameter_in=1.1,   # inches
    right_margin_in=0.35,    # inches
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

    for deg in range(360):
        color = hsv_to_rgb([deg / 360.0, 1.0, 1.0])
        axl.add_patch(Wedge((0.5, 0.5), 0.48, deg, deg + 1,
                            width=0.18, facecolor=color, edgecolor=color, linewidth=0))

    axl.text(0.5, 0.5, center_text, color=txt, ha="center", va="center", fontsize=center_fs)
    axl.text(0.97, 0.50, "0",        color=txt, ha="left",  va="center", fontsize=label_fs)
    axl.text(0.50, 0.97, r"$\pi/2$", color=txt, ha="center", va="bottom", fontsize=label_fs)
    axl.text(0.03, 0.50, r"$\pi$",   color=txt, ha="right",  va="center", fontsize=label_fs)
    axl.text(0.50, 0.03, r"$3\pi/2$",color=txt, ha="center", va="top",    fontsize=label_fs)
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

    axm.plot([0,1,1,0,0], [0,0,1,1,0], color=txt, lw=0.8)
    axm.text(1.07, 1.00, "1", color=txt, ha="left", va="center", fontsize=8)
    axm.text(1.07, 0.00, "0", color=txt, ha="left", va="center", fontsize=8)
    axm.text(0.5, -0.12, r"$|\psi|$", color=txt, ha="center", va="top", fontsize=9)


# ---------------------------------------------------------------------------
# Main Plot Function
# ---------------------------------------------------------------------------

def plot_hld(
    psi,
    dims,
    grouping="hamming",
    ordering="lex",
    show_metrics=True,
    theme="dark",
    save_path=None,
    caption=None,
    show_legend=True,
    legend_kind="phase",   # 'phase' | 'both' | 'none'
    min_height_in=3.0,
    style: str = "default",
    trim_empty: bool = None,
):
    with _rc_context(style):
        # ---------------- Lattice construction & pruning ------------------------
        _, rows = build_hld_lattice(psi, dims, grouping, ordering)
        EPS = 1e-12
        rows = [row for row in rows if any((amp is not None) and (abs(amp) > EPS) for amp in row)]
        if not rows:
            rows = [[]]

        max_len = max(len(r) for r in rows)
        rows = [r + [None]*(max_len - len(r)) for r in rows]

        # Apply trimming if enabled
        if trim_empty:
            rows, kept_indices = _trim_empty_columns(rows, EPS)
        else:
            kept_indices = list(range(max_len))

        nrows = len(rows)
        ncols = len(rows[0]) if rows else 1

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
        for s in ax.spines.values(): s.set_color(label_color)

        # ---------------- Draw Cells -------------------------------------------
        gmax = float(np.max(np.abs(psi))) if np.max(np.abs(psi)) > 0 else 1.0
        for r, row in enumerate(rows):
            for c, amp in enumerate(row):
                if amp is None or abs(amp) <= EPS:
                    continue
                mag = abs(amp) / gmax
                hue = (np.angle(amp) + np.pi) / (2 * np.pi)
                # For paper style, you can soften saturation/value slightly; keep vivid otherwise.
                if (style or "").lower() == "paper":
                    rgb = hsv_to_rgb([hue, mag * 0.85, mag * 0.95])
                else:
                    rgb = hsv_to_rgb([hue, mag, mag])
                ax.add_patch(Rectangle((c, r), 1, 1, color=rgb, linewidth=0))

        ax.set_aspect("equal")
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_xticks(np.arange(ncols)+0.5)
        ax.set_yticks(np.arange(nrows)+0.5)
        ax.set_xticklabels([str(i) for i in kept_indices])
        ax.set_yticklabels(list(map(str, range(nrows))))

        ax.set_xticks(np.arange(ncols+1), minor=True)
        ax.set_yticks(np.arange(nrows+1), minor=True)
        ax.grid(which="minor", color=grid_color, alpha=0.08, linewidth=0.5)
        ax.tick_params(which="minor", length=0)

        # ---------------- Reserve Space Layout ----------------------------------
        legend_needed_in = 1.2
        current_fig_w = fig.get_size_inches()[0]

        if show_legend and legend_kind!="none":
            if current_fig_w - (fig_w - legend_needed_in) < legend_needed_in:
                new_fig_w = fig_w + legend_needed_in
                fig.set_size_inches(new_fig_w, fig.get_size_inches()[1])
                fig.canvas.draw()

        title_band   = 0.12
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
        title_ax = fig.add_axes([
            ax.get_position().x0,
            1.0 - title_band + 0.01,
            ax.get_position().width,
            title_band - 0.03
        ])
        title_ax.set_axis_off()

        if show_metrics:
            m = compute_hld_metrics(psi, dims)
            title_text = (
                f"Global coherence: {m['global_coherence_spectrum']:.3f} |  "
                f"Entanglement visibility: {m['entanglement_visibility_index']:.3f}"
            )
            # normalize to avoid U+2011 warnings
            title_text = _normalize_text(title_text)
            title_ax.text(0.5, 0.5, title_text, ha="center", va="center",
                          fontsize=11, color=label_color)

        # ---------------- Caption Band (bottom anchored) ------------------------
        if caption:
            fig.canvas.draw()
            axpos = ax.get_position()
            cap_ax = fig.add_axes([
                axpos.x0,
                0.04,                   # bottom anchored
                axpos.width,
                caption_band - 0.04
            ])
            cap_ax.set_axis_off()
            cap_ax.text(0.5, 0.5, _normalize_text(caption), ha="center", va="center",
                        fontsize=10, color=label_color)

        # ---------------- Auto-resize for y-label clearance ---------------------
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ylab = ax.yaxis.label.get_window_extent(renderer=renderer)
        ylab_in = ylab.height/fig.dpi + 0.25

        usable_axes_height = (1.0 - title_band - (0.14 + caption_band)) * fig.get_size_inches()[1]
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


def analyze_and_plot_hld(psi, dims, **kwargs):
    plot_hld(psi, dims, **kwargs)
    return compute_hld_metrics(psi, dims)
