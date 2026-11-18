
from __future__ import annotations

from collections.abc import Iterable

HATCHES = ("///", r"\\", "xx", "..", "++", "oo", "--", "**")
FALLBACK_COLORS = (
    "#3569A8",
    "#D8832F",
    "#3D8B6D",
    "#B94D55",
    "#7A67A8",
    "#8A6B4F",
    "#4F8799",
    "#9A6574",
)

def _rgba_is_visible(color) -> bool:
    try:
        red, green, blue, alpha = color
    except (TypeError, ValueError):
        return False
    if alpha <= 0.05:
        return False
    return not (red > 0.96 and green > 0.96 and blue > 0.96)

def _bar_edge_color(patch, fallback):
    face = patch.get_facecolor()
    if _rgba_is_visible(face):
        return face
    edge = patch.get_edgecolor()
    if _rgba_is_visible(edge):
        return edge
    return fallback

def _style_patch(patch, color, hatch):
    patch.set_facecolor("white")
    patch.set_edgecolor(color)
    patch.set_linewidth(1.35)
    patch.set_hatch(hatch)
    patch.set_alpha(1.0)

def _bar_containers(axis) -> list:
    try:
        from matplotlib.container import BarContainer
    except ImportError:
        return []
    return [
        container
        for container in axis.containers
        if isinstance(container, BarContainer) and len(container.patches) > 0
    ]

def apply_hatched_bar_style(figure) -> int:

    styled_count = 0
    for axis in figure.axes:
        containers = _bar_containers(axis)
        for container_index, container in enumerate(containers):
            patches = list(container.patches)
            original_colors = [
                _bar_edge_color(
                    patch,
                    FALLBACK_COLORS[
                        (container_index + patch_index) % len(FALLBACK_COLORS)
                    ],
                )
                for patch_index, patch in enumerate(patches)
            ]
            distinct_colors = {
                tuple(round(float(channel), 5) for channel in color)
                for color in original_colors
            }
            per_bar_hatch = len(distinct_colors) > 1
            for patch_index, (patch, color) in enumerate(
                zip(patches, original_colors)
            ):
                if patch.get_hatch():
                    continue
                style_index = patch_index if per_bar_hatch else container_index
                _style_patch(
                    patch,
                    color,
                    HATCHES[style_index % len(HATCHES)],
                )
                styled_count += 1

        legend = axis.get_legend()
        if legend is None:
            continue
        for handle_index, patch in enumerate(legend.get_patches()):
            if patch.get_hatch():
                continue
            color = _bar_edge_color(
                patch,
                FALLBACK_COLORS[handle_index % len(FALLBACK_COLORS)],
            )
            _style_patch(
                patch,
                color,
                HATCHES[handle_index % len(HATCHES)],
            )

    try:
        import matplotlib as mpl

        mpl.rcParams["hatch.linewidth"] = 0.9
    except (ImportError, KeyError):
        pass
    return styled_count

def install_hatched_bar_save_hook():

    from matplotlib.figure import Figure

    if getattr(Figure, "_hatched_bar_save_hook_installed", False):
        return Figure.savefig

    original_savefig = Figure.savefig

    def hatched_savefig(figure, filename, *args, **kwargs):
        apply_hatched_bar_style(figure)
        return original_savefig(figure, filename, *args, **kwargs)

    Figure.savefig = hatched_savefig
    Figure._hatched_bar_save_hook_installed = True
    Figure._hatched_bar_original_savefig = original_savefig
    return original_savefig
