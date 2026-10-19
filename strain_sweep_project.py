import csv
import matplotlib.pyplot as plt
import numpy as np
from pythtb import *


def build_topological_insulator(strain=0.0):
    """Constructs a 2D Kane-Mele topological insulator model using explicit 4-orbital basis (2 sublattices x 2 spins) to handle spin-orbit coupling.

    Includes uniaxial lattice strain along direction 1.
    """
    # 2D Hexagonal lattice
    lat = [[1.0, 0.0], [0.5, 0.866025]]

    # 4 orbitals: A_up (0), B_up (1), A_dn (2), B_dn (3)
    orb = [
        [1 / 3, 2 / 3],  # A_up
        [2 / 3, 1 / 3],  # B_up
        [1 / 3, 2 / 3],  # A_dn
        [2 / 3, 1 / 3],  # B_dn
    ]

    model = tb_model(2, 2, lat, orb)

    t0 = 1.0  # Base hopping (eV)
    t_strain = t0 * (1.0 - strain)  # Strained hopping

    # --- Nearest-Neighbor Hoppings (Same for Spin Up and Spin Down) ---
    for spin_offset in [0, 2]:  # 0 for spin-up pair, 2 for spin-down pair
        A = 0 + spin_offset
        B = 1 + spin_offset

        model.set_hop(-t_strain, A, B, [0, 0], mode="reset")
        model.set_hop(-t0, A, B, [-1, 0], mode="reset")
        model.set_hop(-t0, A, B, [0, -1], mode="reset")

    # --- Next-Nearest-Neighbor Spin-Orbit Coupling (SOC) ---
    soc = 0.15  # SOC parameter
    for spin_sign, spin_offset in [(1, 0), (-1, 2)]:  # Up = +1, Down = -1
        A = 0 + spin_offset
        B = 1 + spin_offset

        # Intra-sublattice hoppings producing topological mass gap
        model.set_hop(1j * soc * spin_sign, A, A, [1, 0], mode="add")
        model.set_hop(-1j * soc * spin_sign, B, B, [1, 0], mode="add")

    return model


def calculate_k_point_gap(model):
    """Calculates the energy gap between conduction and valence bands at the high-symmetry K-point [1/3, 2/3]."""
    k_K_point = [[1 / 3, 2 / 3]]
    evals = model.solve_all(k_K_point)  # Returns array of shape (4, 1)
    sorted_evals = np.sort(evals[:, 0])

    # In 4-state system: valence = indices 0,1; conduction = indices 2,3
    valence_max = sorted_evals[1]
    conduction_min = sorted_evals[2]
    return conduction_min - valence_max


# Run strain sweep from 0% to 10% in 0.5% steps
strains = np.linspace(0.0, 0.10, 21)
gaps = []

print("Running corrected strain sweep (0% to 10%)...")
for s in strains:
    model = build_topological_insulator(strain=s)
    gap = calculate_k_point_gap(model)
    gaps.append(gap)
    print(f"  Strain: {s*100:4.1f}%  -->  Topological Band Gap: {gap:.5f} eV")

# Export quantitative data to CSV
csv_filename = "topological_gap_vs_strain.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Strain_Percentage", "Strain_Fraction", "Band_Gap_eV"])
    for s, g in zip(strains, gaps):
        writer.writerow([f"{s*100:.1f}", f"{s:.4f}", f"{g:.6f}"])

print(f"\nData saved to '{csv_filename}'.")

# Plot Figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    strains * 100,
    gaps,
    marker="o",
    color="darkblue",
    linewidth=2,
    markersize=6,
    label="K-Point Gap",
)

ax.set_title(
    "Topological Band Gap Evolution Under Mechanical Strain",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("Uniaxial Strain (%)", fontsize=11, labelpad=8)
ax.set_ylabel("Band Gap $\Delta E$ (eV)", fontsize=11, labelpad=8)
ax.set_xlim(0, 10)
ax.set_ylim(0, max(gaps) * 1.15)
ax.grid(True, linestyle="--", alpha=0.5)

# Annotate initial gap
ax.annotate(
    f"0% Strain Gap: {gaps[0]:.3f} eV",
    xy=(0, gaps[0]),
    xytext=(1.2, gaps[0] * 0.85),
    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
    fontsize=10,
    fontweight="bold",
)

plt.tight_layout()
image_filename = "gap_vs_strain_plot.png"
plt.savefig(image_filename, dpi=300)
print(f"Plot saved as '{image_filename}'.")