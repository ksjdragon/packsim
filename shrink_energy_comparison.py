from __future__ import annotations
from typing import List
import os, math, argparse, numpy as np, pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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

	data = {"all": {}, "distinct": {}}
	files = list(Path(filepath).iterdir())

	for i, file in enumerate(files):
		sim = Simulation.load(file)
		data["all"][sim.w] = []
		for frame in sim.frames:
			data["all"][sim.w].append([frame.energy, np.var(frame.stats["avg_radius"]) <= 1e-8])

		sim.get_distinct()
		data["distinct"][sim.w] = []
		for frame in sim.frames:
			data["distinct"][sim.w].append([frame.energy,
							np.var(frame.stats["avg_radius"]) <= 1e-8])

		hashes = int(21*i/len(files))
		print(f'Loading simulations... |{"#"*hashes}{" "*(20-hashes)}|' + \
				f' {i+1}/{len(files)} simulations loaded.', flush=True, end='\r')
	print(flush=True)

	widths = np.asarray(sorted(data["all"]))
	n, h, r, energy = sim.n, sim.h, sim.r, sim.energy

	sim_file = SIM_FOLDER / f"{ENERGY_R_STR[energy]} - EquilibriaData - N{n}.data"
	out_tup = (widths, data, n, h, r, ENERGY_I_STR[energy])
	with open(sim_file, "wb") as output:
		pickle.dump(out_tup, output)

	return out_tup

def axis_settings(ax, widths):
	ax.invert_xaxis()
	ax.grid(zorder=0)
	ax.set_xticks([round(w,2) for w in widths[::-2]])
	ax.set_xticklabels(ax.get_xticks(), rotation = 90)
	plt.subplots_adjust(.05, .12, .97, .9)


def main():
	# Loading arguments.
	parser = argparse.ArgumentParser("Compiles the equilibriums for each width into a diagram.")
	parser.add_argument('sims_path', metavar='path/to/data',
						help="folder that contains simulation files, or cached data file.")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")

	args = parser.parse_args()

	widths, data, n, h, r, energy = get_equilibria_data(Path(args.sims_path))

	torus_min_energies, torus_max_energies, _, _ = get_torus_config_energies(
		n, widths, h, r, energy
	)

	fig_folder = Path(f"figures/ShrinkEnergyComparison - N{n}")
	fig_folder.mkdir(exist_ok=True)

	# Torus minimum energies used as reference.

	# Basin of attraction diagram.
	fig, ax = plt.subplots(figsize=(16, 8))
	all_disorder_count = []
	for width in widths:
		equal_shape = list([c[1] for c in data["all"][width]])
		all_disorder_count.append(100*equal_shape.count(False)/len(data["all"][width]))

	ax.plot(widths, all_disorder_count)
	axis_settings(ax, widths)
	ax.yaxis.set_major_formatter(mtick.PercentFormatter())
	ax.title.set_text('Basin of Attraction')
	ax.set_xlabel("Width")
	ax.set_ylabel("Disordered Equilibria")
	ax.set_yticks(np.arange(0,105, 5))
	fig.savefig(fig_folder / "Basin of Attraction.png")


	# Density of States diagram.
	fig, ax = plt.subplots(figsize=(16, 8))
	distinct_ordered, distinct_unordered = [], []
	for width in widths:
		equal_shape = list([c[1] for c in data["distinct"][width]])
		distinct_ordered.append(equal_shape.count(True))
		distinct_unordered.append(equal_shape.count(False))

	ax.plot(widths, distinct_unordered, label="Unordered Equilibria")
	ax.plot(widths, distinct_ordered, label="Ordered Equilibria")
	ax.legend()
	axis_settings(ax, widths)
	ax.title.set_text('Density of States')
	ax.set_xlabel("Width")
	ax.set_ylabel("Number of States")
	dos_y_max = 1.05*max(distinct_ordered + distinct_unordered)
	ax.set_yticks(np.arange(0, dos_y_max, round(dos_y_max/200, 1)*10))
	fig.savefig(fig_folder / "Density Of States.png")


	# Bifurcation diagram
	fig, ax = plt.subplots(figsize=(16, 8))

	ordered_energies, unordered_energies = [], []
	for width in widths:
		ordered_energies.append([c[0] for c in data["distinct"][width] if c[1]])
		unordered_energies.append([c[0] for c in data["distinct"][width] if not c[1]])

	for i in range(len(torus_min_energies)):
		ordered_energies[i].append(torus_min_energies[i])
		ordered_energies[i].append(torus_max_energies[i])


	null_unorder = []
	for i, energies in enumerate(unordered_energies):
		if len(energies) == 0:
			null_unorder.append(i)
			energies.append(torus_min_energies[i])

	min_order = np.asarray([min(width) for width in ordered_energies])
	max_order = np.asarray([max(width) for width in ordered_energies])
	min_unorder = np.asarray([min(width) for width in unordered_energies])
	max_unorder = np.asarray([max(width) for width in unordered_energies])

	min_unorder_off = min_unorder - torus_min_energies
	max_unorder_off = max_unorder - torus_min_energies
	ax.plot(widths, min_order - torus_min_energies, color='C1')
	#ax.plot(widths, max_order - torus_min_energies, color='C1', linestyle='dotted')
	ax.plot(widths, min_unorder_off, color='C0')
	ax.plot(widths, max_unorder_off, color='C0', linestyle='dotted')
	axis_settings(ax, widths)

	for i in null_unorder:
		ax.scatter(widths[i], min_unorder[i] - torus_min_energies[i],
			marker='X', color="blue", s=50, zorder=4)
		# ax.scatter(widths[i], max_unorder[i] - torus_min_energies[i],
		# 	marker='X', edgecolors="blue", facecolors='none', s=100, zorder=4)

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
	bif_y_max = np.max(np.abs(np.concatenate((min_unorder_off, max_unorder_off))))
	ax.set_yticks(np.arange(-bif_y_max, bif_y_max, round(bif_y_max/20, \
											-math.floor(math.log10(bif_y_max/20)))))
	fig.savefig(fig_folder / "Bifurcation.png")

	print(f"Wrote to {fig_folder}.")

if __name__ == "__main__":
	os.environ["QT_LOGGING_RULES"] = "*=false"
	SIM_FOLDER = Path(f"simulations/ShrinkEnergyComparison")
	SIM_FOLDER.mkdir(exist_ok=True)
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")