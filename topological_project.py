import matplotlib.pyplot as plt 
import numpy as np
from pythtb import *

def build_topological_insulator(strain=0.0): 
    """This models a 2D Kane Mele topological insulator under mechanical strain. A 2D Kane-Mele topological insulator is basically a sheet of graphene (a 2D honeycomb lattice of carbon atoms) where electrons experience a special quantum mechanic force called spin orbit coupling.""" 

    """Insulators normally block electricity completely. But a topological insulator is special here. It acts as an insulator on the inside (bulk), but conducts electricity perfectly along its outer edges. 

    The strain parameter modifies the hopping along one lattice direction. Pulling or squeezing the crystal lattice (mechanical strain) stretches the bonds between atoms. This changes how esily electrons hop from atom to atom, which can alter its electrical properties. 

    EX: We inputed strain=0.0, if we we're to call build_topological_insulator(), then python automatically assumes strain = 0.0 (an unstretched material). However, if we call in build_topological_insulator(strain=0.05), it overrides the default and applies 5% mechanical stretch to the cystal lattice."""




    lat = [[1.0, 0.0], [0.5, 0.866025]]
    orb = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]] 




    """These define the geometry of a 2D honeycomb lattice like graphene. 'lat' defines the lattice vectors. Crystals are repearing patterns. Primitive lattice vectors tell you how to step from one repeating unit cell to the next across the material. [0.5, 0.866025] represent steps at a 60 degree angle (0.5 = cos(60 degrees) and 0.866025 = sin(60 degrees) or sqrt(3)/2. 60 degrees here represent the geometry of a hexagon. Graphene for example is built out of hexagons, which has 120 degrees internal angles. The Bravais lattice of a honeycomb is a triangular lattice. 

    In order to tile a 2D plane using a triangular grid we must have our primitive vector (a_1) point straight across the x-axis (1,0). Our second primitive vector (a_2) must be angled at 60 degrees relative to the first so that when we stack them they form equilateral triangles and hexagons. 

    The orb here represents the sublattice orbitals. A honeycomb lattice has two atom basis (A and B sublattices). And so, the coordinates here are in fractional units (fractions of the lattice vector). In each unit cell, these sites define where the carbon atoms actually reside to form the hexagonal ring pattern. 

    A honeycomb lattice isn't just a simple Bravais lattice. In tight binding code like PythTB, positions are often defined in fractional coodinates relative to a_1 and a_2. We chose 1/3 and 2/3 because it allows us to place the two atoms at exact geometric offsets. This is required to satisfy the carbon-carbon bond lenght (1.42 A_angstrom). This is done also to ensure that when the unit cell is repeated infinitely it can weave together into an unbroken symmetric honeycomb sheet without gaps or overlaps.""" 


    model = tb_model(2, 2, lat, orb, nspin=2) 

    """This is the container for all our physics. The first 2 here (dim_k = 2) tells the code that the reciprocal space is 2 dimensional. This means the crystal is fully periodic (repeating infinitely) in two directions. The second 2 (dim_r = 2) tells it that the real space is 2 dimensional. This means the honeycomb lives on a flat 2D plane. 

    lat and orb here just let's us know that we are feeding the lattice vectors and the sublattice positions that we defined earlier into the model so it knows where the atoms actually sit in space. 

    The reason spin=True here is because electrons carry spin. A model with 2 sublattices has 2 states per unit cell. By setting spin=True we are doubling the degrees of freedom to 4 states per unit cell (A-up, A-down, B-up, and B-down). We need spin=True for a Kane-Mele topological insulator because topological behavior is driven by spin orbit coupling which mixes up spin-up and spin-down electrons as they hop around the lattice.""" 



    t0 = 1.0
    t_strain = t0 * (1.0 - strain)


    """t0 = 1.0 represent the nearest neighbor hopping amplitude. In a tight binding model, hopping (t) represents the quantum mechanical probability or energy cost for an electron to tunnel from one atom to a neighboring atom. t_0 is your baseline hopping value for an unstretched material. Setting it to 1.0 sets the main energy scale for our entire simulation. All other energies like spin orbit coupling will be measured relative to this t_0 value. 

    t_strain = t0 * (1.0 - strain) represents the modified hopping under strain. When we apply mechanical strain to the crystal, we are physically pulling or pushing the atoms closer together or further apart. If we we're to stretch the material (tensile strain where strain >0), then the atoms move further apart. Because of this their electron clouds overlap less, which makes it harder for electrons to hop. The hopping parameter therefore decreases, which is also why a minus sign is used to scale down t_0 when strain is applied."""


    model.set_hop(-t_strain, 0, 1, [0, 0], mode="reset") 
    model.set_hop(-t0, 0, 1, [-1, 0], mode="reset")
    model.set_hop(-t0, 0, 1, [0, -1], mode="reset")

    """This is where we wire up the bonds between our atoms. In PythTB, the method model.set_hop() tells the code to allow an electron to tunnel from one atom to another with a specific energy strenght. The amplitude is -t_strain or -t0. The reason is negative is because in quantum mechanics and tight binding hamiltonians, kinetic energy/hopping is usually written with a minus sign. Its suppose to represent the lowering of energy when electrons are delocalized between atoms. 

    One bond uses -t_strain which changes if you stretch the material. The other two use the baseline -t0. So the format is: model.set_hop(amplitude, from_orbital, unit_cell_shift, mode). The orbitals (0, 1) refer to the two sublattices. 0 is Sublattice A and 1 is sublattice B. This means electrons are hopping from an A atom to a B atom. 

    The cell shift ([0, 0], [-1, 0], [0, -1]) represent the way atoms are connected with their neighboring unit cells. [0, 0] means the neighboring unit cell is inside that same unit cell. [-1, 0] and [0, -1] means the neighbor sits adjacent to the unit cells shifted backward along the lattice vectors. 

    The mode="reset" simply just tells the code to overwrite the already existing value with the new value instead of adding them together. A single carbon atom in a honeycomb for example, has three nearest neighboring unit cells branching from it. The first line of this section sets the hopping for the first bond using the modified strain affected value. This models uniaxial strain, where one specific bond gets stretched or compressed differently from the others. Line two and three sets the hopping for the other two bonds using the unstrained value t0.""" 


    soc = 0.15
    soc_matrix_0 = [[1j * soc, 0], [0, -1j * soc]]
    soc_matrix_1 = [[-1j * soc, 0], [0, 1j * soc]]

    model.set_hop(soc_matrix_0, 0, 0, [1, 0], mode="add")
    model.set_hop(soc_matrix_1, 1, 1, [1, 0], mode="add")

    return model 



"""This introduces our spin orbit coupling (SOC) which transforms regular graphene. It allows us to transform it from a semiconductor into a topological insulator. soc = 0.15 is the spin orbit strength. Spin orbit coupling is a relativistic effect where an electron's spin interacts with its motion around the atomic nucleus. 0.15 means the spin orbit interaction strength is set to 15% of our base hopping parameter t_0. This creates the energy gap that protects edge state from scattering.

This means it loops twice. Once for spin up (spin=0 where s_sign = 1) and once for spin down (spin =1 where s_sign = -1). For the Kane-Mele model, the spin up and spin down electrons feel opposite effective magnetic fields. By assigning an opposite sign(s_sign), the code ensures us that whatever spin up does the spin down does the opposite. This preserves global time reversal symmetry. 

For the model.set_hop parts, these lines hop from 0 to 0 and 1 and 1. This means electrons are skipping their immediate neighbors and hopping to the next nearest atoms on the same sublattice. model="add" tells python to add this new spin orbit effect on top of the standard hopping we define rather than overwriting it.

The 1j represents the imaginary unit. Real numbers in hopping represent simple tunneling. Imaginary numbers introduce a complex phase. This phase acts like a magnetic flux passing through the honeycomb ring. Because of 1j, electrons circulating clockwise pick up a different quantum phase than those going counter clockwise. This creates chiral edge currents that flow exclusively along the outer boundaries of the material.

the return model simply hands back the model variable containing all our lattice geometry."""


print("Calculating band structures for 0% and 5% strain..") 

path = [[0.0, 0.0], [1 / 3, 2 / 3], [0.5, 0.5], [0.0, 0.0]]


"""This sets up the momentum space path (k-space) so we can calculate and plot the material's band structure (energy levels of the electrons). Path is a sequence in reciprocal space. When we feed this path into a band plotting function it calculates the electron energy eigenvalues along the straight lines connecting points 1-4 in order. 

In a 2D crystal, the momentum space (brillouin zone) has geometric corners and edges with quantum phenomena. These fractions are coordinates in fractional reciprocal space. [0.0, 0.0] represents the center of the Brillouin zone where momentum is zero (k_x = 0, k_y = 0). It represents electrons moving with infinite wavelength. This means no phase shift between unit cells. 

[1/3, 2/3] is the K point. This is the dirac point or valley corner of the honeycomb lattice. In graphene this is where the valence and conduction bands touch conically. In our topological insulator, spin orbit coupling tears this point apart to open up a band gap. 

[0.5, 0.5] represents the middle of the edge of the hexagon brillouin zone. And the reason we go back to [0.0, 0.0] is because we must close the loop in order to get a continuous slice through the crystal's momentum space. 

All this will allow us to visually spot the topological band gap we built with our spin orbit coupling.""" 


model_0 = build_topological_insulator(strain=0.0)
(k_vec, k_dist, k_node) = model_0.k_path(path, 300) 
evals_0 = model_0.solve_all(k_vec)



"""Here we are calculating the electron energy bands for the unstrained material. We store the results in model_0. Setting strain = 0.0 ensures this model represents the unstretched honeycomb lattice. This will serve as our baseline reference point before we begin to compare it to a strained version. 

The second line prepares the momentum space (k_space) grid for our plot. path here is our high symmetry coordinates. The number 300 tells PythTB to break our total path down into 300 discrete momentum steps. This ensures that when we plot the bands that the curves look smooth instead of blocky. k_vec is the actual coordinates (k_x and k_y) for all 300 points and k_dist is the cumulative distance long the path. k_node here is the exact tick locations where our high symmetry points sit.

evals_0 feeds every momentum point in k_vec into our quantum mechanical hamiltonian matrix and solves it. In quantum mechanics this means solving an eigenvalue and that is what evals_0 stands for. Because our model has 2 sublattices and spin=True (which doubles the states to 4), solve_all calculates 4 seperate energy bands for every single one of our 300 momentum points."""


model_5 = build_topological_insulator(strain=0.05)
evals_5 = model_5.solve_all(k_vec)


"""Exact same calculation as before, but with mechanical strain now turned on.""" 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

"""plotting the results"""

for i in range(evals_0.shape[0]):
    ax1.plot(k_dist, evals_0[i], color="navy", lw=1.2)
    ax2.plot(k_dist, evals_5[i], color="crimson", lw=1.2)

"""evals_0.shape[0] gives us the total number of energy bands. This loop runs 4 times and letrs us plot each individual energy band one by one from i = 0 to i = 3. Because we added spin orbit coupling, we will see a topological band gap open up at the dirac points (K) in the unstrained plot. Then by looking at the strained plot, we should be able to see how pulling on the crystal lattice distorts that band gap. This could potentially drive a topological phase transition."""


for ax, title in zip(
    [ax1, ax2], ["0% Strain (Ideal Lattice)", "5% Strain (Deformed Lattice)"]
):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(k_node)
    ax.set_xticklabels(["Γ", "K", "M", "Γ"], fontsize=11)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

ax1.set_ylabel("Energy (eV)", fontsize=11)
plt.suptitle(
    "Topological insulator band structure under mechanical deformation", 
    fontsize=14, 
)
plt.tight_layout()

plt.savefig("topological_strain_analysis.png", dpi=300)
print(
    "\nSUCCESS! High resolution band structure plot saved as 'topological_strain_analysis.png'."
)

"""Instead of writing seperate formatting commands for both subplots we use a zip loop to update ax1 and ax2 at the same time."""
