from __future__ import annotations
from typing import List
import os, argparse, numpy as np, pickle
import matplotlib.pyplot as plt
from pathlib import Path

from simulation import Diagram, Simulation
from packsim_core import AreaEnergy, RadialALEnergy, RadialTEnergy

ENERGY_R_STR = {AreaEnergy: "Area", RadialALEnergy: "Radial[AL]", RadialTEnergy: "Radial[T]"}
ENERGY_I_STR = {AreaEnergy: "area", RadialALEnergy: "radial-al", RadialTEnergy: "radial-t"}
I_TO_R = {"area": "Area","radial-t": "Radial[AL]", "radial-t": "Radial[T]"}

def get_torus_config_energies(n: int, widths: np.ndarray, h: float, r: float,
								energy: str) -> Tuple:
	sim_file = SIM_FOLDER / f"{I_TO_R[energy]} - TorusConfigEnergy - N{n}.data"
	if sim_file.is_file():
		with open(sim_file, "rb") as data:
			return pickle.load(data)

	torus_min_energies, torus_max_energies = np.empty(widths.shape), np.empty(widths.shape)
	torus_min_configs, torus_max_configs = [None]*len(widths), [None]*len(widths)

	for i, w in enumerate(widths):
		sim = Simulation(n, w, h, r, energy)
		configs = []

		for j in range(1):
			for c in range(1,n):	# Ignore 0, tends to error.
				config = (1,c) if j == 0 else (c,1)
				sim.add_frame(torus=config)
				configs.append(config)

				# eigs = np.sort(np.linalg.eig(sim.frames[-1].hessian(10e-5))[0])
				# if eigs[0] > 1e-4:
				# 	del sim.frames[-1]
				# 	del config[-1]

				hashes = int(21*i/len(widths))
				print(f'Generating at width {w:.02f}... ' + \
						f'|{"#"*hashes}{" "*(20-hashes)}| {i+1}/{len(widths)}, ' + \
						f'{c + (n-1)*j}/{2*(n-1)} completed.', flush=True, end='\r')

		pair = list(zip(configs,[frame.energy for frame in sim.frames]))
		torus_min_configs[i], torus_min_energies[i] = min(pair, key=lambda x: x[1])
		torus_max_configs[i], torus_max_energies[i] = max(pair, key=lambda x: x[1])

	print(flush=True)

	out_tup = (torus_min_energies, torus_max_energies, torus_min_configs, torus_max_configs)
	with open(sim_file, "wb") as output:
		pickle.dump(out_tup, output)

	return out_tup


def get_equilibria_data(filepath: Path):
	if filepath.is_file():
		with open(filepath, "rb") as data:
			return pickle.load(data)

	sims = []
	files = list(Path(filepath).iterdir())

	for i, file in enumerate(files):
		sims.append(Simulation.load(file))

		hashes = int(21*i/len(files))
		print(f'Loading simulations... |{"#"*hashes}{" "*(20-hashes)}|' + \
				f' {i+1}/{len(files)} simulations loaded.', flush=True, end='\r')
	print(flush=True)
	sims.sort(key=lambda x: x.w)

	widths = np.asarray([sim.w for sim in sims])

	pairs_at_widths = []
	for sim in sims:
		pairs_at_widths.append([(frame.energy, np.var(frame.stats["avg_radius"]) <= 1e-8) \
										 for frame in sim.frames])

	# min_frames = [min(sim.frames, key=lambda x: x.energy) for sim in sims]
	# max_frames = [max(sim.frames, key=lambda x: x.energy) for sim in sims]

	# min_energies = np.asarray([frame.energy for frame in min_frames])
	# max_energies = np.asarray([frame.energy for frame in max_frames])

	# min_markers = [np.var(frame.stats["avg_radius"]) <= 1e-8 for frame in min_frames]
	# max_markers = [np.var(frame.stats["avg_radius"]) <= 1e-8 for frame in max_frames]

	n, h, r, energy = sims[0].n, sims[0].h, sims[0].r, sims[0].energy

	sim_file = SIM_FOLDER / f"{ENERGY_R_STR[energy]} - EquilibriaData - N{n}.data"
	out_tup = (widths, pairs_at_widths, n, h, r, ENERGY_I_STR[energy])
	with open(sim_file, "wb") as output:
		pickle.dump(out_tup, output)

	return out_tup

def axis_settings(ax, widths):
	ax.invert_xaxis()
	ax.grid(zorder=0)
	ax.set_xticks([round(w,2) for w in widths[::-2]])
	ax.set_xticklabels(ax.get_xticks(), rotation = 90)


def main():
	# Loading arguments.
	parser = argparse.ArgumentParser("Compiles the equilibriums for each width into a diagram.")
	parser.add_argument('sims_path', metavar='path/to/data',
						help="folder that contains simulation files, or cached data file.")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")

	args = parser.parse_args()

	widths, energy_shape_tups, n, h, r, energy = get_equilibria_data(Path(args.sims_path))

	torus_min_energies, torus_max_energies, _, _ = get_torus_config_energies(
		n, widths, h, r, energy
	)

	fig_folder = Path(f"figures/ShrinkEnergyComparison - N{n}")
	fig_folder.mkdir(exist_ok=True)

	# Torus minimum energies used as reference.
	plt.tight_layout()
	# Density of States diagram.
	fig, ax = plt.subplots(figsize=(16, 8))

	distinct_ordered, distinct_unordered = [], []
	for energy_shapes in energy_shape_tups:
		equal_shape = list([tup[1] for tup in energy_shapes])
		distinct_ordered.append(equal_shape.count(True))
		distinct_unordered.append(equal_shape.count(False))

	ax.plot(widths, distinct_unordered, label="Unordered Equilibria")
	ax.plot(widths, distinct_ordered, label="Ordered Equilibria")
	ax.legend()
	axis_settings(ax, widths)
	ax.title.set_text('Density of States')
	ax.set_xlabel("Width")
	ax.set_ylabel("Number of States")
	fig.savefig(fig_folder / "Density Of States.png")

	# Bifurcation diagram
	fig, ax = plt.subplots(figsize=(16, 8))

	ordered_energies, unordered_energies = [], []
	for energy_shapes in energy_shape_tups:
		order_width, unorder_width = [], []
		for pair in energy_shapes:
			if pair[1]:
				order_width.append(pair[0])
			else:
				unorder_width.append(pair[0])
		ordered_energies.append(order_width)
		unordered_energies.append(unorder_width)


	for i in range(len(torus_min_energies)):
		ordered_energies[i].append(torus_min_energies[i])
		ordered_energies[i].append(torus_max_energies[i])


	null_unorder = []
	for i, width in enumerate(unordered_energies):
		if len(width) == 0:
			null_unorder.append(i)
			for x in unordered_energies[i-1]:
				width.append(x)
			#width = unordered_energies[i-1]
			#width.append(max(unordered_energies[i-1]))

	min_order = np.asarray([min(width) for width in ordered_energies])
	max_order = np.asarray([max(width) for width in ordered_energies])
	min_unorder = np.asarray([min(width) for width in unordered_energies])
	max_unorder = np.asarray([max(width) for width in unordered_energies])

	ax.plot(widths, min_order - torus_min_energies, color='C1')
	#ax.plot(widths, max_order - torus_min_energies, color='C1', linestyle='dotted')
	ax.plot(widths, min_unorder - torus_min_energies, color='C0')
	ax.plot(widths, max_unorder - torus_min_energies, color='C0', linestyle='dotted')
	axis_settings(ax, widths)

	for i in null_unorder:
		ax.scatter(widths[i], min_unorder[i] - torus_min_energies[i],
			marker='X', color="blue", s=50, zorder=4)
		ax.scatter(widths[i], max_unorder[i] - torus_min_energies[i],
			marker='X', edgecolors="blue", facecolors='none', s=100, zorder=4)

	# for i, marker in enumerate(min_markers):
	# 	if marker:
	# 		ax.scatter(widths[i], min_energies[i]-torus_min_energies[i],
	# 			marker='H', color="orange", s=20, zorder=4)
	# 	else:
	# 		ax.scatter(widths[i], min_energies[i]-torus_min_energies[i],
	# 			marker='d', color="blue", s=20, zorder=4)

	# for i, marker in enumerate(max_markers):
	# 	if marker:
	# 		ax.scatter(widths[i], max_energies[i]-torus_min_energies[i],
	# 			marker='H', edgecolors="orange", s=20, facecolors='none', zorder=4)
	# 	else:
	# 		ax.scatter(widths[i], max_energies[i]-torus_min_energies[i],
	# 			marker='d', edgecolors="blue", s=20, facecolors='none', zorder=4)

	ax.title.set_text('Reduced Energy vs. Width')
	ax.set_xlabel("Width")
	ax.set_ylabel("Reduced Energy")

	fig.savefig(fig_folder / "Bifurcation.png")

if __name__ == "__main__":
	os.environ["QT_LOGGING_RULES"] = "*=false"
	SIM_FOLDER = Path(f"simulations/ShrinkEnergyComparison")
	SIM_FOLDER.mkdir(exist_ok=True)
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")