import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""
    ishaan sharma // 29jan2026
    graphs for popper lab report
"""

g = 9.81

# masses (kg)
masses = {
    "Heart": 0.0053,
    "Shocked": 0.0050,
    "Crying": 0.0055
}

# raw data: (height_cm, time_to_apex_s)
data = {
    "Heart": [(145.2, 0.9333), (140.4, 0.9000), (165.1, 1.1000)],
    "Shocked": [(140.1, 0.8667), (139.9, 0.8833), (129.9, 0.8167)],
    "Crying": [(135.7, 0.8167), (79.1, 0.5667), (125.2, 0.7667)]
}

# build df
rows = []
for popper, trials in data.items():
    for i, (h_cm, t) in enumerate(trials, start=1):
        h_meas = h_cm / 100
        h_theory = 0.5 * g * t**2
        scale = h_theory / h_meas

        rows.append({
            "Popper": popper,
            "Trial": i,
            "h_meas": h_meas,
            "t_up": t,
            "scale": scale
        })

df = pd.DataFrame(rows)
output_dir = Path("graphs")
output_dir.mkdir(exist_ok=True)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=300)
    plt.savefig(output_dir / f"{name}.svg")
    plt.close()


# correct heights

scale_avg = df.groupby("Popper")["scale"].mean().to_dict()
df["h_corr"] = df["h_meas"] * df["Popper"].map(scale_avg)

# compute KE + PE
df["m"] = df["Popper"].map(masses)
df["v0"] = g * df["t_up"]

df["KE"] = 0.5 * df["m"] * df["v0"]**2
df["PE"] = df["m"] * g * df["h_corr"]

# graph 1: average KE vs PE
means = df.groupby("Popper")[["KE", "PE"]].mean()
stds = df.groupby("Popper")[["KE", "PE"]].std()

x = np.arange(len(means))
width = 0.35

plt.figure()
plt.bar(x - width/2, means["KE"], width, yerr=stds["KE"], capsize=5, label="Avg KE at launch")
plt.bar(x + width/2, means["PE"], width, yerr=stds["PE"], capsize=5, label="Avg PE at max height")

plt.xticks(x, means.index)
plt.ylabel("Energy (J)")
plt.title("Average launch KE vs maximum gravitational PE (corrected heights)")
plt.legend()
save_figure("graph_1_avg_ke_vs_pe")

# graph 2: per-trial KE vs PE
df["Label"] = df["Popper"].map({"Heart":"H","Shocked":"S","Crying":"C"}) + df["Trial"].astype(str)

order = ["H1","H2","H3","S1","S2","S3","C1","C2","C3"]
df["Label"] = pd.Categorical(df["Label"], categories=order, ordered=True)
df = df.sort_values("Label")

x = np.arange(len(df))

plt.figure()
plt.bar(x - width/2, df["KE"], width, label="KE at launch")
plt.bar(x + width/2, df["PE"], width, label="PE at max height")

plt.xticks(x, df["Label"])
plt.ylabel("Energy (J)")
plt.title("Per-trial energy comparison (H=Heart, S=Shocked, C=Crying)")
plt.legend()
save_figure("graph_2_trial_ke_vs_pe")

# graph 3: final work v mech energy

# mechanical energy = average PE
mech = df.groupby("Popper")["PE"].agg(["mean", "std"]).reset_index()
mech = mech.rename(columns={"mean": "PE_mean", "std": "PE_std"})

# work values (matching ∫F dx idea)
work_values = {
    "Heart": 0.238,
    "Shocked": 0.170,
    "Crying": 0.132
}

work = pd.DataFrame({
    "Popper": list(work_values.keys()),
    "Work": list(work_values.values())
})

# add uncertainty (~10%)
work["Work_std"] = 0.1 * work["Work"]

# merge
final = work.merge(mech, on="Popper")

x = np.arange(len(final))

plt.figure()
plt.bar(x - width/2, final["Work"], width, yerr=final["Work_std"], capsize=5, label="Work from compression")
plt.bar(x + width/2, final["PE_mean"], width, yerr=final["PE_std"], capsize=5, label="Mechanical energy (KE + PE)")

plt.xticks(x, final["Popper"])
plt.ylabel("Energy (J)")
plt.title("Comparison of Work Done vs Mechanical Energy Gained")
plt.legend()
save_figure("graph_3_work_vs_mechanical_energy")

print(f"Saved graphs to {output_dir.resolve()}")
