import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex']=False # added by DrE for latex includesvg
plt.rcParams['svg.fonttype']='none' # added by DrE for latex includesvg
plt.rcParams['font.size']=8
from pathlib import Path

"""
Ishaan Sharma // 29 Jan 2026 (updated 27 Mar 2026)
Graphs for Popper Lab Report
Modified by Dr E // 3 Apr 2026 for figure formatting
"""

g = 9.81

output_dir = Path("graphs")
output_dir.mkdir(exist_ok=True)

def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close()

# GRAPH 1: FORCE vs DISPLACEMENT (FROM SCISSOR JACK DATA)

x_disp = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

heart_force = np.array([0, 235, 415, 480, 620, 800, 980, 850, 750, 630])/1000.*g
shocked_force = np.array([0, 230, 370, 448, 600, 740, 830, 710, 600, 520])/1000.*g
crying_force = np.array([0, 200, 340, 420, 580, 720, 810, 680, 580, 500])/1000.*g

plt.figure(figsize=(3.4167,2),dpi=600)

plt.plot(x_disp, heart_force, color='green', marker='s', linewidth=2, markersize=8, label='heart')
plt.plot(x_disp, shocked_force, color='purple', marker='o', linewidth=2, markersize=8, label='shocked')
plt.plot(x_disp, crying_force, color='orange', marker='^', linewidth=2, markersize=8, label='crying')

#plt.title("Displacement vs Force")
#plt.xlabel("Scissor Jack compression displacement (starting from 8.5) (mm)")
plt.xlabel(r"displacement, \unit{\milli\meter}")
#plt.ylabel("Force (grams)")
plt.ylabel(r"force, \unit{\newton}")

plt.xticks(x_disp)
#plt.yticks(np.arange(0, 1100, 200))
#plt.ylim(0, 1000)

plt.grid(True, linestyle='-', linewidth=0.5, color='black', alpha=0.3)

plt.legend() #loc='best', bbox_to_anchor=(1, 0.5))

#save_figure("graph_1_force_vs_displacement")
plt.tight_layout()
plt.savefig("graph_1_force_vs_displacement.png",dpi=600)
plt.savefig("graph_1_force_vs_displacement.svg")
plt.close()









# ENERGY CALCULATIONS

# masses (kg)
masses = {
    "heart": 0.0053,
    "shocked": 0.0057,
    "crying": 0.0055
}

# raw data: (height_cm, time_to_apex_s)
data = {
    "heart": [(145.2, 0.9333), (140.4, 0.9000), (165.1, 1.1000)],
    "shocked": [(140.1, 0.8667), (139.9, 0.8833), (129.9, 0.8167)],
    "crying": [(135.7, 0.8167), (79.1, 0.5667), (125.2, 0.7667)]
}

rows = []
for popper, trials in data.items():
    for i, (h_cm, t_up) in enumerate(trials, start=1):
        rows.append({
            "Popper": popper,
            "Trial": i,
            "h_cm": h_cm,
            "h_m": h_cm / 100.0,
            "t_up": t_up
        })

df = pd.DataFrame(rows)
df["m"] = df["Popper"].map(masses)
df["PE"] = df["m"] * g * df["h_m"]

# GRAPH 2.1: AVERAGE MAX PE

means = df.groupby("Popper")["PE"].mean().reindex(["heart", "shocked", "crying"])
stds = df.groupby("Popper")["PE"].std().reindex(["heart", "shocked", "crying"])

x = np.arange(len(means))

plt.figure(figsize=(3.4167, 2),dpi=600)
plt.bar(x, means.values, yerr=stds.values, capsize=4)
plt.xticks(x, means.index)
plt.ylabel(r"energy, \unit{\joule}")
#plt.title("Average Maximum Gravitational Potential Energy")
#save_figure("graph_2_1_avg_max_pe")
plt.tight_layout()
plt.savefig("graph_2_1_avg_max_pe.png",dpi=600)
plt.savefig("graph_2_1_avg_max_pe.svg")
plt.close()








# GRAPH 2.2: PER-TRIAL MAX PE

label_map = {"heart": "H", "shocked": "S", "crying": "C"}
df["Label"] = df["Popper"].map(label_map) + df["Trial"].astype(str)

order = ["H1", "H2", "H3", "S1", "S2", "S3", "C1", "C2", "C3"]
df["Label"] = pd.Categorical(df["Label"], categories=order, ordered=True)
df = df.sort_values("Label")

x = np.arange(len(df))

plt.figure(figsize=(3.4167, 2))
plt.bar(x, df["PE"])
plt.xticks(x, df["Label"])
plt.ylabel(r"energy, \unit{\joule}")
#plt.title("Per-Trial Maximum Gravitational Potential Energy")
#save_figure("graph_2_2_trial_max_pe")
plt.tight_layout()
plt.savefig("graph_2_2_trial_max_pe.png",dpi=600)
plt.savefig("graph_2_2_trial_max_pe.svg")
plt.close()










# =========================================================
# GRAPH 3: WORK vs PE
# =========================================================

mech = df.groupby("Popper")["PE"].agg(["mean", "std"]).reset_index()
mech = mech.rename(columns={"mean": "PE_mean", "std": "PE_std"})

work_values = {
    "heart": 0.238,
    "shocked": 0.170,
    "crying": 0.132
}

work = pd.DataFrame({
    "Popper": list(work_values.keys()),
    "Work": list(work_values.values())
})

work["Work_std"] = 0.1 * work["Work"]

final = work.merge(mech, on="Popper")
final["Popper"] = pd.Categorical(final["Popper"], categories=["heart", "shocked", "crying"], ordered=True)
final = final.sort_values("Popper")

x = np.arange(len(final))
width = 0.35

plt.figure(figsize=(3.4167, 2),dpi=600)
plt.bar(
    x - width / 2,
    final["Work"],
    width,
    yerr=final["Work_std"],
    capsize=4,
    label=r"$W$"
)
plt.bar(
    x + width / 2,
    final["PE_mean"],
    width,
    yerr=final["PE_std"],
    capsize=4,
    label=r"GPE"
)

plt.xticks(x, final["Popper"])
plt.ylabel(r"energy, \unit{\joule}")
#plt.title("Comparison of Compression Work and Launch Gravitational Potential Energy")
plt.legend() #pos='upper right'
#save_figure("graph_3_work_vs_launch_pe")
plt.tight_layout()
plt.savefig("graph_3_work_vs_launch_pe.png",dpi=600)
plt.savefig("graph_3_work_vs_launch_pe.svg")
plt.close()

summary = df.groupby("Popper")["PE"].agg(["mean", "std"]).reindex(["heart", "shocked", "crying"])
print("PE summary (J):")
print(summary.round(6))
#print(f"\nSaved graphs to {output_dir.resolve()}")
