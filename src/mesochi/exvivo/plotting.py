"""The original ex vivo figures, with scoped styles and editable slices."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _slice_indices(shape, indices):
    if indices is None:
        indices = (max(0, shape[0]//2 - 20), shape[1]//2, shape[2]//2)
    if len(indices) != 3 or any(not isinstance(i, (int, np.integer)) or not 0 <= i < n
                                for i, n in zip(indices, shape)):
        raise ValueError('slice_indices must give one valid integer for each spatial axis.')
    return tuple(indices)


def _coronal_slices(mask, slice_xs):
    if slice_xs is None:
        if mask.shape == (170, 102, 102) and all(mask[x].any() for x in (65, 85, 105)):
            slice_xs = (65, 85, 105)
        else:
            active = np.flatnonzero(mask.any(axis=(1, 2)))
            if not len(active):
                raise ValueError('The plotting mask must contain at least one voxel.')
            slice_xs = active[np.round(np.linspace(0, len(active)-1, 5)).astype(int)[1:4]]
    if len(slice_xs) != 3 or any(not isinstance(x, (int, np.integer))
                                or not 0 <= x < mask.shape[0] or not mask[x].any()
                                for x in slice_xs):
        raise ValueError('slice_xs must contain three valid, nonempty coronal mask slices.')
    return tuple(int(x) for x in slice_xs)


def plot_slices(volume, *, slice_indices=None, figure_dpi=400, save_dpi=800):
    """Preserve the supplied orthogonal-slice rendering, including its colorbar."""
    slice_indices = _slice_indices(volume.shape, slice_indices)
    with plt.rc_context({'figure.dpi': figure_dpi, 'savefig.dpi': save_dpi}):
        plt_nme = '\u03c7_\u03bcQSM'
        plt_chi = volume

        cx, cy, cz = slice_indices

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        coronal_slice = plt_chi[cx, :, :]
        axes[0].imshow(np.flipud(coronal_slice), cmap='gray', origin='lower')
        axes[0].set_title(f'{plt_nme} Coronal Slice (X={cx})')
        axial_slice = plt_chi[:, cy, :]
        axes[1].imshow(axial_slice, cmap='gray', origin='lower')
        axes[1].set_title(f'{plt_nme} Axial Slice (Y={cy})')
        sagittal_slice = plt_chi[:, :, cz]
        axes[2].imshow(np.flipud(sagittal_slice), cmap='gray', origin='lower')
        axes[2].set_title(f'{plt_nme} Sagittal Slice (Z={cz})')
        plt.colorbar(axes[2].imshow(sagittal_slice, cmap='gray', origin='lower'), ax=axes[2], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        return fig







def plot_comparison(maps, mask_phase, *, slice_xs=None, figure_dpi=400, save_dpi=800):
    """Original three-row QSM/μQSM comparison, with ROI crops and four colorbars."""
    slice_xs = _coronal_slices(np.asarray(mask_phase, dtype=bool), slice_xs)
    chi_qsm_ref = maps['chi_qsm_ref']
    chi_muqsm_ref = maps['chi_muqsm_ref']
    chi_qsm_tv_ref = maps['chi_qsm_tv_ref']
    chi_muqsm_tv_ref = maps['chi_muqsm_tv_ref']
    with plt.rc_context():
        CROP_PAD = 4       # Reduce for stronger zoom
        CBAR_WIDTH = 0.78
        CBAR_DOWN = 0.025

        TITLE_FS = 25
        ROW_FS = 22
        CBAR_TICK_FS = 22
        CBAR_LABEL_FS = 24

        plt.rcParams.update({
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 20,
            "text.color": "white",
            "axes.titlecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "savefig.facecolor": "black",
            "figure.dpi": figure_dpi,
            "savefig.dpi": save_dpi,
            "savefig.bbox": "tight" 
        })


        # ── Data ─────────────────────────────────────────────────────────────────────
        mask = np.asarray(mask_phase, dtype=bool)


        def clean(vol):
            vol = np.nan_to_num(
                np.asarray(vol, dtype=np.float32),
                nan=0.0, posinf=0.0, neginf=0.0
            )
            return vol * mask


        qsm_l1 = clean(chi_qsm_tv_ref)
        muqsm_l1 = clean(chi_muqsm_tv_ref)
        qsm_l2 = clean(chi_qsm_ref)
        muqsm_l2 = clean(chi_muqsm_ref)

        maps = [
            qsm_l1,
            muqsm_l1,
            qsm_l1 - muqsm_l1,
            qsm_l2,
            muqsm_l2,
            qsm_l2 - muqsm_l2
        ]

        titles = [
            r"$\chi_{\mathrm{QSM}}$",
            r"$\chi_{\mu\mathrm{QSM}}$",
            r"$\chi_{\mathrm{QSM}}-\chi_{\mu\mathrm{QSM}}$",
            r"$\chi_{\mathrm{QSM}}$",
            r"$\chi_{\mu\mathrm{QSM}}$",
            r"$\chi_{\mathrm{QSM}}-\chi_{\mu\mathrm{QSM}}$"
        ]


        # ── Display limits ───────────────────────────────────────────────────────────
        def limits(indices):
            values = np.concatenate([
                maps[i][x][mask[x]]
                for i in indices
                for x in slice_xs
            ])
            return values.min(), values.max()


        main_lim = limits([0, 1, 3, 4])
        diff_lim = limits([2, 5])

        display_limits = [
            main_lim, main_lim, diff_lim,
            main_lim, main_lim, diff_lim
        ]


        # ── Crop each slice using its mask ───────────────────────────────────────────
        def crop_box(x, pad=CROP_PAD):
            rows, cols = np.where(mask[x])

            r0 = max(rows.min() - pad, 0)
            r1 = min(rows.max() + pad + 1, mask.shape[1])

            c0 = max(cols.min() - pad, 0)
            c1 = min(cols.max() + pad + 1, mask.shape[2])

            return slice(r0, r1), slice(c0, c1)


        crop_boxes = [crop_box(x) for x in slice_xs]


        # ── Layout ───────────────────────────────────────────────────────────────────
        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("black")

        fig = plt.figure(figsize=(18, 8.4), facecolor="black")

        gs = fig.add_gridspec(
            4, 7,
            width_ratios=[1, 1, 1, 0.16, 1, 1, 1],
            height_ratios=[1, 1, 1, 0.22],
            hspace=0.00,   # smaller gap between rows
            wspace=0.06
        )

        fig.subplots_adjust(
            left=0.06,
            right=0.995,
            top=0.94,
            bottom=0.14
        )

        grid_cols = [0, 1, 2, 4, 5, 6]

        axes = np.array([
            [fig.add_subplot(gs[r, c]) for c in grid_cols]
            for r in range(3)
        ])

        images = [None] * 6


        # ── Plot cropped maps ────────────────────────────────────────────────────────
        for r, (x, box) in enumerate(zip(slice_xs, crop_boxes)):
            rs, cs = box
            cropped_mask = mask[x, rs, cs]

            for c, (vol, clim) in enumerate(zip(maps, display_limits)):
                cropped_map = vol[x, rs, cs]

                image = np.ma.masked_where(
                    ~cropped_mask,
                    cropped_map
                )

                images[c] = axes[r, c].imshow(
                    image,
                    cmap=cmap,
                    vmin=clim[0],
                    vmax=clim[1],
                    origin="upper",
                    interpolation="none"
                )

                axes[r, c].axis("off")

            axes[r, 0].text(
                -0.22, 0.5,
                f"Coronal\n$x={x}$",
                transform=axes[r, 0].transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=ROW_FS,
                fontweight="bold",
                color="white",
                clip_on=False
            )


        for ax, title in zip(axes[0], titles):
            ax.set_title(
                title,
                fontsize=TITLE_FS,
                color="white",
                pad=12
            )


        # ── Colorbars ────────────────────────────────────────────────────────────────
        colorbar_info = [
            (images[0], gs[3, 0:2], r"$\chi$ (ppm)", 5),
            (images[2], gs[3, 2], r"$\Delta\chi$ (ppm)", 3),
            (images[3], gs[3, 4:6], r"$\chi$ (ppm)", 5),
            (images[5], gs[3, 6], r"$\Delta\chi$ (ppm)", 3)
        ]

        formatter = ticker.FuncFormatter(lambda value, _: f"{value:+.2f}")

        for image, location, label, nticks in colorbar_info:
            cax = fig.add_subplot(location)

            # Shorten and center colorbar
            pos = cax.get_position()
            width = pos.width * CBAR_WIDTH

            cax.set_position([
                pos.x0 + (pos.width - width) / 2,
                pos.y0 - CBAR_DOWN,   # move colorbar downward
                width,
                pos.height
            ])

            cb = fig.colorbar(image, cax=cax, orientation="horizontal")
            cb.set_ticks(np.linspace(*image.get_clim(), nticks))
            cb.ax.xaxis.set_major_formatter(formatter)

            cb.set_label(
                label,
                fontsize=CBAR_LABEL_FS,
                color="white",
                labelpad=12
            )

            cb.ax.tick_params(
                axis="x",
                labelsize=CBAR_TICK_FS,
                colors="white",
                direction="in",
                length=10,
                width=1.5,
                pad=9
            )

            cb.outline.set_edgecolor("white")
            cb.outline.set_linewidth(1.5)


        return fig


def save_reconstruction_figures(maps, mask, output_dir, *, show_plots=None,
                                figure_dpi=400, save_dpi=800,
                                slice_xs=None, slice_indices=None):
    """Save both figures and close only this call's figures when not displaying."""
    show = plt.get_backend().lower() != 'agg' if show_plots is None else show_plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Validate both sets before creating either figure.
    slice_xs = _coronal_slices(np.asarray(mask, dtype=bool), slice_xs)
    slice_indices = _slice_indices(mask.shape, slice_indices)
    saved = []
    for name in ('exvivo_slices', 'exvivo_comparison'):
        if name == 'exvivo_slices':
            fig = plot_slices(maps['chi_muqsm_ref'], slice_indices=slice_indices,
                              figure_dpi=figure_dpi, save_dpi=save_dpi)
        else:
            fig = plot_comparison(maps, mask, slice_xs=slice_xs,
                                  figure_dpi=figure_dpi, save_dpi=save_dpi)
        path = output_dir / f'{name}.png'
        try:
            fig.savefig(path, dpi=save_dpi,
                        bbox_inches='tight' if name == 'exvivo_comparison' else None,
                        facecolor=fig.get_facecolor())
            saved.append(str(path.resolve()))
            if show:
                plt.show()
        finally:
            if not show:
                plt.close(fig)
    return saved
