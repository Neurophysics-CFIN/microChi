# Plotting functions for QSM / muQSM reconstruction results.
#
#   plot_components  : 4 x 3 figure -- GT, reconstruction, |error| and central profile
#                      for several components (e.g. chi, chi^c, chi^s) of one method
#   plot_comparison  : 3 x 3 figure -- one component, GT vs. several methods
#                      (e.g. QSM and muQSM)
#
# All plotting parameters are identical to the original scripts. As there, the style is
# set globally (plt.rcParams), so it is in effect whenever the figure is drawn or saved.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# LaTeX subscripts for the methods, used to build the labels
GT    = r"\mathrm{GT}"
QSM   = r"\mathrm{QSM}"
MUQSM = r"\mu\mathrm{QSM}"


# ── Shared helpers ───────────────────────────────────────────────────────────
def _to_numpy(a):
    """Accept NumPy or CuPy arrays."""
    return np.asarray(a.get() if hasattr(a, "get") and not isinstance(a, np.ndarray) else a)


def chi_label(component, method):
    """
    chi_label("c", MUQSM) -> r"$\\chi^c_{\\mu\\mathrm{QSM}}$"
    chi_label("",  GT)    -> r"$\\chi_{\\mathrm{GT}}$"
    """
    sup = f"^{component}" if component else ""
    return rf"$\chi{sup}_{{{method}}}$"


def _show_image(ax, image, cmap, limits, title, fontsize, color="black", pad=8):
    im = ax.imshow(
        image,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        origin="lower",
        interpolation="none"
    )
    ax.set_title(
        title,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        pad=pad
    )
    ax.set_axis_off()
    ax.set_box_aspect(1)
    return im


def _style_profile_axis(ax, FS, grid_color, minor_width):
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        color=grid_color,
        alpha=0.8
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=FS["tick"],
        length=7,
        width=1.2
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=4,
        width=minor_width
    )


# ── Fonts: Times New Roman for text, STIX (Times-style) for math text
def _pick_serif():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Liberation Serif"):   # Liberation Serif = metric clone
        if name in names:
            return name
    return "DejaVu Serif"


SERIF = _pick_serif()
FONT_RC = {
    "font.family": "serif",
    "font.serif": [SERIF, "DejaVu Serif"],
    # STIX: Times-style font made for math. Times New Roman itself has no math
    # metrics, which pushes super/subscripts far apart and into the panels above.
    "mathtext.fontset": "stix",
}


# ── Figure title ─────────────────────────────────────────────────────────────
TITLE_SPACE    = 0.6     # extra figure height [inch] for the title; panels keep their size
TITLE_GAP      = 0.3     # gap [inch] between the panel titles and the figure title
TITLE_FONTSIZE = 32


def _title_layout(height, title):
    """Figure height and (bottom, top) margins with room for a title on top."""
    extra = TITLE_SPACE if title else 0.0
    new_h = height + extra
    return new_h, 0.09 * height / new_h, 0.94 * height / new_h


def _add_title(fig, title, height):
    """Place the title centred over everything drawn and TITLE_GAP above its top."""
    if not title:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()]
    box = Bbox.union([b for b in boxes if b is not None])
    x = 0.5 * (box.x0 + box.x1) / fig.bbox.width
    y = box.y1 / fig.bbox.height + TITLE_GAP / height
    fig.suptitle(title, x=x, y=y, ha="center", va="bottom",
                 fontsize=TITLE_FONTSIZE, fontweight="bold")


def _method_name(method):
    """LaTeX subscript -> plain text for titles, e.g. MUQSM -> "μQSM" (stays bold)."""
    return method.replace(r"\mu", "μ").replace(r"\mathrm{", "").replace("}", "").replace(" ", "")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: components of one method (4 x 3)
# ═════════════════════════════════════════════════════════════════════════════
FS_COMPONENTS = {
    "title": 28,
    "rmse": 25,
    "label": 25,
    "tick": 22,
    "legend": 22,
    "cbar": 22,
    "cbar_label": 20,
}

RC_COMPONENTS = {
    **FONT_RC,
    "font.size": 18,
    "axes.labelsize": FS_COMPONENTS["label"],
    "xtick.labelsize": FS_COMPONENTS["tick"],
    "ytick.labelsize": FS_COMPONENTS["tick"],
    "figure.dpi": 400,
    "savefig.dpi": 800,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.2,
}


def plot_components(gt, rec, mask_roi, method=MUQSM, title=None, save_path=None, show=True):
    """
    GT, reconstruction, |error| and central x-profile for several components.

    Parameters
    ----------
    gt, rec : dict {component: 3D volume}
        Same keys in both, one column per key, in this order. The key is the
        superscript of chi: "" -> chi, "c" -> chi^c, "s" -> chi^s.
    mask_roi : 3D array
    method : str
        LaTeX subscript of the reconstruction method (MUQSM, QSM, ...).
    title : str or None
        Figure title. None -> "Ground truth vs. <method> reconstruction";
        "" -> no title (original figure size).
    save_path : str or None
        If given, the figure is saved there (800 dpi, tight bbox).
    show : bool
        Call plt.show() and return None. With show=False the figure is returned
        instead (for further editing or saving) and not shown.

    Example
    -------
    plot_components(
        gt ={"": chi_comp_gt,          "c": chi_comp_cyl_gt,          "s": chi_comp_sph_gt},
        rec={"": chi_comp_multi_muqsm, "c": chi_comp_cyl_multi_muqsm, "s": chi_comp_sph_multi_muqsm},
        mask_roi=mask_roi)
    """
    FS = FS_COMPONENTS
    keys = list(gt.keys())
    if list(rec.keys()) != keys:
        raise ValueError("gt and rec must have the same keys in the same order.")

    if title is None:
        title = f"Ground truth vs. {_method_name(method)} reconstruction"
    gt_titles  = {k: chi_label(k, GT) for k in keys}
    rec_titles = {k: chi_label(k, method) for k in keys}

    # Global style, as in the original scripts: fonts and math text are
    # resolved when the figure is drawn, which can happen after this function returns.
    plt.rcParams.update(RC_COMPONENTS)
    # ── Prepare volumes ──────────────────────────────────────────────────
    mask = np.asarray(_to_numpy(mask_roi), dtype=bool)

    def clean(vol):
        return np.nan_to_num(
            np.asarray(_to_numpy(vol), dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ) * mask

    gt_vols  = {k: clean(gt[k]) for k in keys}
    rec_vols = {k: clean(rec[k]) for k in keys}

    shape = gt_vols[keys[0]].shape
    cy, cz = shape[1] // 2, shape[2] // 2
    x_axis = np.arange(shape[0])
    mask_slice = mask[:, :, cz]

    all_values = np.concatenate([
        volume.ravel()
        for volumes in (gt_vols, rec_vols)
        for volume in volumes.values()
    ])
    chi_max = np.abs(all_values).max()
    chi_limits = (-chi_max, chi_max)              # symmetric: 0 is mid-grey

    # ── Maps, errors and profiles ────────────────────────────────────────
    results = {}
    for key in keys:
        g, r = gt_vols[key], rec_vols[key]
        difference = (r - g) * mask
        error = np.abs(difference[:, :, cz])
        results[key] = {
            "gt": g[:, :, cz],
            "rec": r[:, :, cz],
            "error": error,
            "gt_profile": g[:, cy, cz],
            "rec_profile": r[:, cy, cz],
            "rmse": np.sqrt(np.sum(difference**2) / mask.sum()),
            "error99": np.percentile(error[mask_slice], 99),
        }
    error_max = max(result["error99"] for result in results.values())

    # ── Figure ───────────────────────────────────────────────────────────
    ncols = len(keys)
    fig_h, bottom, top_margin = _title_layout(17, title)
    fig, axes = plt.subplots(
        4, ncols,
        figsize=(15 * ncols / 3, fig_h),
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.1, 1.05]},
        squeeze=False
    )
    fig.subplots_adjust(
        left=0.045,
        right=0.94,
        bottom=bottom,
        top=top_margin,
        wspace=0.30,          # room for the y tick labels of the profile plots
        hspace=0.24           # room for the panel titles (super/subscripts)
    )

    for col, key in enumerate(keys):
        result = results[key]

        # Ground truth
        im_chi = _show_image(axes[0, col], result["gt"], "gray", chi_limits,
                             gt_titles[key], FS["title"], pad=8)
        # Reconstruction
        _show_image(axes[1, col], result["rec"], "gray", chi_limits,
                    rec_titles[key], FS["title"], pad=8)
        # Absolute error
        im_err = _show_image(axes[2, col], result["error"], "inferno", (0, error_max),
                             f"RMSE = {result['rmse']:.4f}", FS["rmse"], "#8B0000", pad=8)

        # Central profile
        ax = axes[3, col]
        ax.plot(x_axis, result["gt_profile"], color="#1a1a2e", linewidth=2.5,
                label=gt_titles[key])
        ax.plot(x_axis, result["rec_profile"], "--", color="#c0392b", linewidth=2.3,
                label=rec_titles[key])
        ax.set_xlabel("Voxel index $x$", fontsize=FS["label"], labelpad=9)
        if col == 0:
            ax.set_ylabel(r"$\chi$", fontsize=FS["label"], labelpad=5)
        ax.legend(fontsize=FS["legend"], loc="upper left", frameon=True,
                  framealpha=0.9, edgecolor="0.7")
        _style_profile_axis(ax, FS, grid_color="0.75", minor_width=1)

    # ── Precisely positioned shared colorbars ────────────────────────────
    fig.canvas.draw()
    top    = axes[0, -1].get_position()
    middle = axes[1, -1].get_position()
    err    = axes[2, -1].get_position()

    # Susceptibility colorbar spanning the first two rows
    cax_chi = fig.add_axes([top.x1 + 0.012, middle.y0, 0.018, top.y1 - middle.y0])
    cb_chi = fig.colorbar(im_chi, cax=cax_chi)
    cb_chi.ax.tick_params(labelsize=FS["cbar"], direction="in", length=7, width=1.2)

    # Error colorbar
    cax_err = fig.add_axes([err.x1 + 0.012, err.y0, 0.018, err.height])
    cb_err = fig.colorbar(im_err, cax=cax_err)
    cb_err.ax.tick_params(labelsize=FS["cbar"], direction="in", length=7, width=1.2)

    _add_title(fig, title, fig_h)

    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
        return None     # not returned, so Jupyter does not display it a second time

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: one component, GT vs. several methods (3 x 3)
# ═════════════════════════════════════════════════════════════════════════════
FS_COMPARISON = {
    "title": 28,
    "rmse": 25,
    "label": 25,
    "tick": 22,
    "legend": 22,
    "cbar": 22,
    "cbar_label": 22,
}

RC_COMPARISON = {
    **FONT_RC,
    "font.size": 16,
    "axes.titlesize": FS_COMPARISON["title"],
    "axes.labelsize": FS_COMPARISON["label"],
    "xtick.labelsize": FS_COMPARISON["tick"],
    "ytick.labelsize": FS_COMPARISON["tick"],
    "figure.dpi": 400,
    "savefig.dpi": 800,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.2,
}


def plot_comparison(gt, cases, mask_roi, component="c", title=None, save_path=None, show=True):
    """
    One component: GT image, then for each method its reconstruction, |error| map
    and central x-profile against GT.

    Parameters
    ----------
    gt : 3D volume
        Ground truth of this component.
    cases : dict {method: 3D volume}
        method is the LaTeX subscript (QSM, MUQSM, or your own, e.g. r"\\mathrm{TKD}").
        One column per method, in this order.
    mask_roi : 3D array
    component : str
        Superscript of chi: "c", "s", or "" for total chi.
    title : str or None
        Figure title. None -> "<chi>: ground truth vs. <method 1> vs. <method 2> ...";
        "" -> no title (original figure size).
    save_path : str or None
        If given, the figure is saved there (800 dpi, tight bbox).
    show : bool
        Call plt.show() and return None. With show=False the figure is returned
        instead (for further editing or saving) and not shown.

    Example
    -------
    plot_comparison(chi_cyl_gt, {QSM: chi_cyl_multi_qsm, MUQSM: chi_cyl_multi_muqsm},
                    mask_roi, component="c")
    """
    FS = FS_COMPARISON
    CHI_GT = chi_label(component, GT)
    CHI = rf"$\chi^{component}$" if component else r"$\chi$"
    labels = {method: chi_label(component, method) for method in cases}
    if title is None:
        title = f"{CHI}: ground truth vs. " + " vs. ".join(_method_name(m) for m in cases)

    # Global style, as in the original scripts: fonts and math text are
    # resolved when the figure is drawn, which can happen after this function returns.
    plt.rcParams.update(RC_COMPARISON)
    gt = np.asarray(_to_numpy(gt))
    mask_roi = np.asarray(_to_numpy(mask_roi))

    nx, ny, nz = gt.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    mask_slice = mask_roi[:, :, cz].astype(bool)
    gt_slice = (gt * mask_roi)[:, :, cz]
    true_profile = gt[:, cy, cz]
    x_axis = np.arange(nx)

    chi_max = np.max(np.abs(gt))
    chi_limits = (-chi_max, chi_max)              # symmetric: 0 is mid-grey

    # ── Prepare reconstruction results ───────────────────────────────────
    results = {}
    for method, volume in cases.items():
        volume = np.asarray(_to_numpy(volume))
        difference = (volume - gt) * mask_roi
        error_slice = np.abs(difference[:, :, cz])
        results[labels[method]] = {
            "recon": (volume * mask_roi)[:, :, cz],
            "error": error_slice,
            "profile": volume[:, cy, cz],
            "rmse": np.sqrt(np.sum(difference**2) / np.sum(mask_roi)),
        }

    error_max = max(
        np.percentile(result["error"][mask_slice], 99)
        for result in results.values()
    )

    # ── Figure ───────────────────────────────────────────────────────────
    ncols = 1 + len(cases)
    fig_h, bottom, top_margin = _title_layout(14.5, title)
    fig, axes = plt.subplots(
        3, ncols,
        figsize=(14.5 * ncols / 3, fig_h),
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.05]},
        squeeze=False
    )
    fig.subplots_adjust(
        left=0.045,
        right=0.94,
        bottom=bottom,
        top=top_margin,
        wspace=0.18,
        hspace=0.24           # room for the panel titles (super/subscripts)
    )
    axes[1, 0].remove()
    # profiles share the y-axis: same GT in every column, directly comparable
    for col in range(2, ncols):
        axes[2, col].sharey(axes[2, 1])
    axes[2, 0].remove()

    def add_colorbar(ax, image, label=None):
        return _add_inset_colorbar(fig, ax, image, label, FS["cbar"], offset=1.04)

    # Ground truth
    im_gt = _show_image(axes[0, 0], gt_slice, "gray", chi_limits, CHI_GT,
                        FS["title"], pad=7)

    # Reconstructions, error maps, and profiles
    for col, (label, result) in enumerate(results.items(), start=1):
        _show_image(axes[0, col], result["recon"], "gray", chi_limits, label,
                    FS["title"], pad=7)
        im_err = _show_image(axes[1, col], result["error"], "inferno", (0, error_max),
                             f"RMSE = {result['rmse']:.4f}", FS["rmse"], "#8B0000", pad=7)

        ax = axes[2, col]
        ax.plot(x_axis, true_profile, color="#1a1a2e", linewidth=2.5, label=CHI_GT)
        ax.plot(x_axis, result["profile"], "--", color="#c0392b", linewidth=2.3,
                label=label)
        ax.set_xlabel("Voxel index $x$", fontsize=FS["label"], labelpad=7)
        if col == 1:
            ax.set_ylabel(f"{CHI}", fontsize=FS["label"], labelpad=3)
        ax.legend(fontsize=FS["legend"], loc="upper left", frameon=True,
                  framealpha=0.9, edgecolor="0.7")
        _style_profile_axis(ax, FS, grid_color="0.7", minor_width=1.0)
        if col > 1:
            ax.tick_params(which="both", labelleft=False)   # shared y-axis

    # Shared reconstruction and error colorbars
    add_colorbar(axes[0, -1], im_gt)
    add_colorbar(axes[1, -1], im_err)

    _add_title(fig, title, fig_h)

    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
        return None     # not returned, so Jupyter does not display it a second time

    return fig


#%% shared inset colorbar

def _add_inset_colorbar(fig, ax, image, label=None, fontsize=22, labelsize=22, offset=1.04):
    cax = inset_axes(
        ax, width="10%", height="100%", loc="lower left",
        bbox_to_anchor=(offset, 0, 1, 1),
        bbox_transform=ax.transAxes, borderpad=0,
    )
    cb = fig.colorbar(image, cax=cax)
    cb.ax.tick_params(labelsize=fontsize, length=6, width=1.2)
    cb.set_label(label, fontsize=labelsize, labelpad=9)
    return cb


#%% SNR comparison for L2 or TV reconstructions

FS_SNR = {
    "title": 26,
    "snr": 23,
    "rmse": 23,
    "label": 24,
    "tick": 24,
    "legend": 25,
    "cbar": 22,
    "cbar_label": 22,
}

RC_SNR = {
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 18,
    "axes.labelsize": FS_SNR["label"],
    "xtick.labelsize": FS_SNR["tick"],
    "ytick.labelsize": FS_SNR["tick"],
    "figure.dpi": 400,
    "savefig.dpi": 800,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.3,
}


def plot_snr_comparison(
    gt, recon, mask_roi,
    snr_levels=None,
    component="c",
    method=MUQSM,
    title=None,
    save_path=None,
    show=True,
):
    """Compare reconstructions across SNRs: maps, errors, and central profiles.

    recon uses string SNR keys, as in LoFd_cyl_noisy.npz. With snr_levels=None,
    all supplied levels are plotted in descending order, infinity first.
    title=None keeps the original untitled layout. With show=False, return
    the figure so it can be edited or saved without opening a window.
    """
    gt = np.asarray(_to_numpy(gt))
    mask_roi = np.asarray(_to_numpy(mask_roi), dtype=bool)
    if gt.ndim != 3 or mask_roi.shape != gt.shape or not np.any(mask_roi):
        raise ValueError("gt and a nonempty mask_roi must be matching 3D volumes.")
    if not np.all(np.isfinite(gt)):
        raise ValueError("gt must be finite.")
    recon_by_snr = {}
    for key, volume in recon.items():
        snr = float(key)
        if np.isnan(snr) or snr <= 0 or snr in recon_by_snr:
            raise ValueError("SNR keys must be distinct positive values or infinity.")
        volume = np.asarray(_to_numpy(volume))
        if volume.shape != gt.shape or not np.all(np.isfinite(volume)):
            raise ValueError("Each reconstruction must be finite and match gt.shape.")
        recon_by_snr[snr] = volume
    snr_levels = sorted(recon_by_snr, reverse=True) if snr_levels is None else [float(snr) for snr in snr_levels]
    if not snr_levels or any(float(snr) not in recon_by_snr for snr in snr_levels):
        raise ValueError("snr_levels must select available reconstructions.")
    if not np.any(mask_roi[:, :, gt.shape[2] // 2]):
        raise ValueError("mask_roi must include voxels in the central slice.")

    FS = FS_SNR


    plt.rcParams.update(RC_SNR)


    # ── Labels ───────────────────────────────────────────────────────────────────
    CHI_GT = chi_label(component, GT)
    CHI_MUQSM = chi_label(component, method)
    CHI = rf"$\chi^{component}$" if component else r"$\chi$"


    # ── Data and cropped display region ─────────────────────────────────────────
    nx, ny, nz = gt.shape
    cy, cz = ny // 2, nz // 2

    mask_slice = mask_roi[:, :, cz].astype(bool)
    full_gt = (gt * mask_roi)[:, :, cz]

    # Crop around the phantom
    CROP_PAD = 2
    rows, cols = np.where(mask_slice)

    row_slice = slice(
        max(rows.min() - CROP_PAD, 0),
        min(rows.max() + CROP_PAD + 1, nx)
    )

    col_slice = slice(
        max(cols.min() - CROP_PAD, 0),
        min(cols.max() + CROP_PAD + 1, ny)
    )

    crop = (row_slice, col_slice)

    gt_slice = full_gt[crop]
    true_profile = gt[:, cy, cz]
    x_axis = np.arange(nx)

    chi_limits = (
        np.min(gt),
        np.max(np.abs(gt))
    )


    # ── Prepare results ─────────────────────────────────────────────────────────
    results = {}

    for snr in snr_levels:
        volume = recon_by_snr[float(snr)]
        difference = (volume - gt) * mask_roi
        full_error = np.abs(difference[:, :, cz])

        results[snr] = {
            "recon": (volume * mask_roi)[:, :, cz][crop],
            "error": full_error[crop],
            "profile": volume[:, cy, cz],
            "rmse": np.sqrt(np.sum(difference**2) / np.sum(mask_roi)),
            "error99": np.percentile(full_error[mask_slice], 99)
        }

    error_max = max(result["error99"] for result in results.values())

    profile_values = np.concatenate(
        [true_profile] +
        [result["profile"] for result in results.values()]
    )

    profile_min, profile_max = profile_values.min(), profile_values.max()
    profile_pad = 0.05 * (profile_max - profile_min)

    profile_limits = (
        profile_min - profile_pad,
        profile_max + profile_pad
    )


    # ── Figure layout ────────────────────────────────────────────────────────────
    ncols = len(snr_levels) + 1

    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(22, 11.5),
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.05]}
    )

    fig.subplots_adjust(
        left=0.045,
        right=0.94,
        bottom=0.09,
        top=0.94,
        wspace=0.18,
        hspace=0.16
    )

    axes[1, 0].axis("off")
    axes[2, 0].axis("off")


    def add_colorbar(ax, image, label):
        return _add_inset_colorbar(
            fig, ax, image, label=label,
            fontsize=FS["cbar"], labelsize=FS["cbar_label"], offset=1.05,
        )


    # ── Ground truth ─────────────────────────────────────────────────────────────
    im_gt = _show_image(
        axes[0, 0],
        gt_slice,
        "gray",
        chi_limits,
        CHI_GT,
        FS["title"]
    )


    # ── Reconstructions, errors and profiles ─────────────────────────────────────
    legend_handles = None

    for col, snr in enumerate(snr_levels, start=1):

        result = results[snr]

        snr_title = (
            r"SNR $=\infty$"
            if np.isinf(snr)
            else f"SNR = {snr:g}"
        )

        _show_image(
            axes[0, col],
            result["recon"],
            "gray",
            chi_limits,
            snr_title,
            FS["snr"]
        )

        im_err = _show_image(
            axes[1, col],
            result["error"],
            "inferno",
            (0, error_max),
            f"RMSE = {result['rmse']:.4f}",
            FS["rmse"],
            "#8B0000"
        )

        ax = axes[2, col]

        line_gt, = ax.plot(
            x_axis,
            true_profile,
            color="#1a1a2e",
            linewidth=2.5,
            label=CHI_GT
        )

        line_rec, = ax.plot(
            x_axis,
            result["profile"],
            "--",
            color="#c0392b",
            linewidth=2.3,
            label=CHI_MUQSM
        )

        if legend_handles is None:
            legend_handles = [line_gt, line_rec]

        ax.set_xlim(x_axis[0], x_axis[-1])
        ax.set_ylim(profile_limits)
        ax.set_xlabel("Voxel index $x$", labelpad=8)

        if col == 1:
            ax.set_ylabel(f"{CHI}", labelpad=2)
        else:
            ax.tick_params(labelleft=False)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=FS["tick"],
            length=7,
            width=1.2
        )

        ax.tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=4,
            width=1
        )

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.8,
            color="0.75",
            alpha=0.8
        )


    # ── One shared legend and two shared colorbars ───────────────────────────────
    axes[2, 0].legend(
        legend_handles,
        [CHI_GT, CHI_MUQSM],
        loc="center left",
        fontsize=FS["legend"],
        frameon=True,
        edgecolor="0.7"
    )

    add_colorbar(
        axes[0, -1],
        im_gt,
        label=None
    )

    add_colorbar(
        axes[1, -1],
        im_err,
        label=None
    )

    if title:
        fig.suptitle(title, fontsize=FS["title"], fontweight="bold")
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
        return None
    return fig
