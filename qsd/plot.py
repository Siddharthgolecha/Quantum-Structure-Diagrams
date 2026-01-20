from contextlib import nullcontext

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle, Wedge

from .core import build_qsd_lattice
from .dimensions import resolve_dims
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
    """Return a matplotlib rc_context for the requested style.

    Args:
        style: "paper" enables PAPER_RC; anything else is a no-op context.

    Example:
        >>> ctx = _rc_context("paper")
        >>> hasattr(ctx, "__enter__")
        True
    """
    return (
        mpl.rc_context(PAPER_RC) if (style or "").lower() == "paper" else nullcontext()
    )


# -------------------- Text normalization (fixes U+2011 warnings) --------------------


def _normalize_text(s):
    """Replace non-breaking hyphen and long dashes with a safe ASCII hyphen.

    Example:
        >>> _normalize_text("A\u2011B\u2013C\u2014D")
        'A-B-C-D'
    """
    if not s:
        return s
    return (
        s.replace("\u2011", "-")  # non-breaking hyphen
        .replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
    )


# ---------------- Optional Column Trimming (Preserve original indices) ----------------
def _trim_empty_columns(rows, eps=1e-12):
    """Remove empty columns while preserving original column indices.

    Args:
        rows: Ragged rows of complex amplitudes.
        eps: Threshold below which entries are treated as zero.

    Returns:
        (new_rows, kept_indices, keep_mask)

    Example:
        >>> rows = [[0+0j, 1+0j], [0+0j, 0+0j]]
        >>> _trim_empty_columns(rows)[1]
        [1]
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
    """Convert a flat index to a ket string using mixed-radix digits.

    Args:
        index: Flat basis index.
        dims: Per-axis dimensions.

    Returns:
        A ket string like "|0,1,0>".

    Example:
        >>> _idx_to_ket(3, [2, 2])
        '|1,1>'
    """
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
    """Add a phase wheel legend outside the main axes.

    Args:
        fig: Matplotlib figure.
        ax: Main axes used for placement reference.
        theme: "dark" or "light" theme to set text colors.
        wheel_diameter_in: Diameter of the wheel in inches.
        right_margin_in: Right margin in inches.
        center_text: Center label text.
        label_fs: Font size for the labels.
        center_fs: Font size for the center label.

    Returns:
        The legend axes instance.

    Example:
        >>> fig, ax = plt.subplots()
        >>> leg = _add_phase_legend_outside(fig, ax, theme="light")
        >>> hasattr(leg, "set_axis_off")
        True
    """
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
    """Add a magnitude legend below a phase wheel axes.

    Args:
        fig: Matplotlib figure.
        ring_ax: Axes returned by _add_phase_legend_outside.
        theme: "dark" or "light" theme to set text colors.

    Returns:
        None. Adds an axes to the figure.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ring = _add_phase_legend_outside(fig, ax, theme="light")
        >>> _add_magnitude_legend_below(fig, ring, theme="light")
    """
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
    show: bool = True,
    return_fig: bool = False,
):
    """Render a QSD plot for a statevector.

    Args:
        psi: Statevector as a flat complex array.
        grouping: Row grouping strategy.
        ordering: Within-row ordering strategy.
        show_metrics: If True, render the metrics band at the top.
        theme: "dark" or "light" plot theme.
        save_path: Optional path to save the figure as an image.
        caption: Optional caption in the bottom band.
        show_legend: Whether to draw the phase/magnitude legend.
        legend_kind: "phase", "both", or "none".
        min_height_in: Minimum figure height in inches.
        style: "paper" for paper-friendly rcParams.
        trim_empty: If True, remove empty columns.
        annotate_basis: If True, annotate basis kets inside cells.
        annotate_threshold: Minimum magnitude to annotate.
        phase_gauge: Phase gauge strategy for consistent hue.
        render_eps: Render threshold for small amplitudes.
        show: If True, call plt.show() when not saving.
        return_fig: If True, return the figure instead of showing it.

    Returns:
        Matplotlib figure if return_fig=True, otherwise None.

    Example:
        >>> import numpy as np
        >>> psi = np.array([1, 0, 0, 0], dtype=complex)
        >>> fig = plot_qsd(psi, grouping="hamming", ordering="lex", return_fig=True, show=False)
        >>> fig is not None
        True
    """
    dims = resolve_dims(psi)

    with _rc_context(style):
        # ---------------- Lattice construction & pruning ------------------------
        EPS = float(render_eps)

        ordered, rows, index_rows, _, _ = build_qsd_lattice(
            psi,
            grouping,
            ordering,
            return_indices=True,
            normalize_state=True,  # safe: plot is invariant to global scaling (you use gmax)
            phase_gauge=phase_gauge,  # aligns hue convention across figures
        )
        row_labels_full = list(ordered.keys())  # IMPORTANT: true group labels

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
                m = compute_qsd_metrics(psi, grouping=grouping, ordering=ordering)
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

        if return_fig:
            return fig

        if save_path:
            plt.close(fig)
        elif show:
            plt.show()


def analyze_and_plot_qsd(psi, **kwargs):
    """Plot a QSD and return metrics (and optionally the figure).

    This is a convenience wrapper around plot_qsd and compute_qsd_metrics.

    Args:
        psi: Statevector as a flat complex array.
        **kwargs: Passed through to plot_qsd. If return_fig=True, the figure
            is returned alongside metrics.

    Returns:
        If return_fig=False: metrics dict.
        If return_fig=True: (fig, metrics).

    Example:
        >>> import numpy as np
        >>> psi = np.array([1, 0, 0, 0], dtype=complex)
        >>> fig, m = analyze_and_plot_qsd(psi, return_fig=True, show=False)
        >>> 'row_delocalization' in m
        True
    """
    fig = plot_qsd(psi, **kwargs)
    grouping = kwargs.get("grouping", "hamming")
    ordering = kwargs.get("ordering", "lex")
    metrics = compute_qsd_metrics(psi, grouping=grouping, ordering=ordering)
    if kwargs.get("return_fig", False):
        return fig, metrics
    return metrics
