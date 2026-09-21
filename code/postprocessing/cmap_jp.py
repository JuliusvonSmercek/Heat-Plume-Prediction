import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# shifted hot colormap to better capture details
def new_cmap(colors, nodes, name: str = None):
    nodes = np.asarray(nodes, dtype=float).copy()
    nodes -= nodes[0]
    nodes /= nodes[-1]
    # log.info(nodes)

    if name:
        my_cmap = LinearSegmentedColormap.from_list(name, list(zip(nodes, colors, strict=True)))
        try:
            mpl.colormaps.register(cmap=my_cmap)
        except ValueError:
            # Already registered — replace so step2 ambient/spread updates take effect
            mpl.colormaps.unregister(name)
            mpl.colormaps.register(cmap=my_cmap)


def register_jp_temperature_cmaps(ambient_temperature_C: float, temperature_spread_C: float) -> None:
    """Register temperature colormaps using ambient / spread from step2 physical_parameters.

    Absolute node values are normalized to [0, 1] inside ``new_cmap``; plot color limits
    still come from ``DataToVisualize`` vmin/vmax. Intermediate stops keep the same
    relative spacing as the original 10.6 ± 5.0 design.
    """
    mid = ambient_temperature_C
    smax = temperature_spread_C
    # Original intermediate offsets (for smax=5): ±0.25, ±0.75, ±1.5 and heating/cooling knots
    scale = smax / 5.0

    # bidirectional: ambient ± spread
    s2 = 1.5 * scale
    s1 = 0.75 * scale
    s0 = 0.25 * scale
    nodes = np.array([mid - smax, mid - s2, mid - s1, mid - s0, mid, mid + s0, mid + s1, mid + s2, mid + smax])

    name = "jp_temperature_bidirectional"
    colors = ["#313695", "#4575B4", "#74ADD1", "#E0F3F8", "#FFFFFF", "#FEE090", "#FDAE61", "#F46D43", "#A50026"]
    new_cmap(colors, nodes, name)

    name = "jp_temperature_bidirectional_dark"
    colors = ["#80F3FF", "#00B4D8", "#0077B6", "#032030", "#000000", "#2E0505", "#C41E3A", "#FF6B00", "#FFD700"]
    new_cmap(colors, nodes, name)

    # heating: ambient → ambient + spread
    nodes = mid + scale * np.array([0.0, 1.1, 1.4, 2.9, 5.0])
    name = "jp_temperature_upperlinear"
    colors = ["white", "darkblue", "darkred", "orange", "yellow"]
    new_cmap(colors, nodes, name)

    name = "jp_temperature_upperlinear_dark"
    colors = ["black", "darkblue", "darkred", "orange", "yellow"]
    new_cmap(colors, nodes, name)

    # cooling: ambient - spread → ambient
    nodes = mid + scale * np.array([-5.0, -3.9, -3.6, -2.1, 0.0])
    name = "jp_temperature_lowerlinear"
    colors = ["yellow", "orange", "darkred", "darkblue", "white"]
    new_cmap(colors, nodes, name)

    name = "jp_temperature_lowerlinear_dark"
    colors = ["yellow", "orange", "darkred", "darkblue", "black"]
    new_cmap(colors, nodes, name)


# Register with unit spacing so names exist at import; visualization re-registers with step2 values.
register_jp_temperature_cmaps(ambient_temperature_C=0.0, temperature_spread_C=1.0)

if True:
    # shifted hot colormap to better capture details
    name = "jp_linear"
    colors = ["white", "darkblue", "darkred", "orange", "yellow"]
    nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    new_cmap(colors, nodes, name)

    # dark mode
    name = "jp_linear_dark"
    colors = ["black", "darkblue", "darkred", "orange", "yellow"]
    nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    new_cmap(colors, nodes, name)
