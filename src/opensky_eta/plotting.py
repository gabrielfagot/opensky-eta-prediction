"""Plot style shared by the notebooks."""
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PHASE_COLORS = {"climb": "#2a78d6", "est_climb": "#86b6ef", "cruise": "#eb6834",
                "short cruise": "#eda100", "descent": "#1baf7a", "est_descent": "#8fd8bb"}


def set_plot_style():
    """Matplotlib defaults used in every notebook."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=COLORS),
    })
