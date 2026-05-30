import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.examples.octadecane.nusselt_correlation import nusselt_correlation
from src.parameters.config import ExperimentConfig


mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
    }
)

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
nu_history = np.load("./data/nusselt.npz")["nu"]
exp_data = np.load("./data/exp_nusselt.npz")

times, nu_values = zip(*nu_history)

dim_times = [
    t * cfg.stefan_number * cfg.thermal_diffusivity_ref / cfg.l**2 for t in times
]
nu_pred = nusselt_correlation(dim_times, Ra=cfg.rayleigh_number)

mask = exp_data["x"] <= dim_times[-1]
exp_x = exp_data["x"][mask]
exp_y = exp_data["y"][mask]

plt.figure(figsize=(6.3, 3.9))
plt.plot(
    dim_times[1000:],
    nu_values[1000:],
    color="C0",
    linestyle="--",
    label="Present work",
)
plt.plot(
    dim_times[1000:],
    nu_pred[1000:],
    color="C1",
    linestyle="-",
    label="Jany & Bejan (1988)",
)
plt.plot(
    exp_x, exp_y, color="C2", linestyle="-", label="Okada (1984), exp."
)

plt.xlabel(r"$\tilde{t}$")
plt.ylabel(r"$Nu$")
plt.legend()
plt.ylim(5, 9)
plt.tight_layout()
plt.savefig("./graphs/nusselt_evolution_2.tiff")
