from __future__ import annotations
from typing import List
from simulation import Diagram, Simulation
import os, argparse, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_torus_config_energies(n: int, widths: np.ndarray, h: float, r: float,
								energy: str) -> Tuple[np.ndarray, np.ndarray]:
	torus_min_energies, torus_max_energies = np.empty(widths.shape), np.empty(widths.shape)
	for i, w in enumerate(widths):
		sim = Simulation(n, w, h, r, energy)
		configs = []

		for j in range(1):
			for c in range(1,n):	# Ignore 0, tends to error.
				config = (1,c) if j == 0 else (c,1)
				sim.add_frame(torus=config)
				configs.append(config)

				eigs = np.sort(np.linalg.eig(sim.frames[-1].hessian(10e-5))[0])
				zero_eigs = np.count_nonzero(np.isclose(eigs, np.zeros((len(eigs),)), atol=1e-4))
				if zero_eigs != 2:
					del sim.frames[-1]
					del config[-1]

				hashes = int(21*i/len(widths))
				print(f'Generating at width {w:.02f}... ' + \
						f'|{"#"*hashes}{" "*(20-hashes)}| {i+1}/{len(widths)}, {2*c}/{2*(n-1)}' + \
						f' completed.', flush=True, end='\r')

		torus_min_energies[i] = min([frame.energy for frame in sim.frames])
		torus_max_energies[i] = max([frame.energy for frame in sim.frames])

	print(flush=True)
	return torus_min_energies, torus_max_energies


def main():
	# Loading arguments.
	parser = argparse.ArgumentParser("Compiles the equilibriums for each width into a diagram.")
	parser.add_argument('sims_path', metavar='path/to/folder', 
						help="folder that contains simulation files.")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")
	parser.add_argument('-o', '--output', dest='output_file')

	args = parser.parse_args()

	sims = []
	files = list(Path(args.sims_path).iterdir())
	
	for i, file in enumerate(files):
		sims.append(Simulation.load(file))

		hashes = int(21*i/len(files))
		print(f'Loading simulations... |{"#"*hashes}{" "*(20-hashes)}|' + \
				f' {i+1}/{len(files)} simulations loaded.', flush=True, end='\r')
	print(flush=True)
	sims.sort(key=lambda x: x.w)

	sim_folder = Path(f"simulations/ShrinkEnergyComparison")
	fig_folder = Path(f"figures/ShrinkEnergyComparison - N{sims[0].n}")
	sim_folder.mkdir(exist_ok=True)
	fig_folder.mkdir(exist_ok=True)

	widths = np.asarray([sim.w for sim in sims])

	min_frames = [min(sim.frames, key=lambda x: x.energy) for sim in sims]
	max_frames = [max(sim.frames, key=lambda x: x.energy) for sim in sims]

	min_energies = np.asarray([frame.energy for frame in min_frames])
	max_energies = np.asarray([frame.energy for frame in max_frames])

	torus_min_energies, torus_max_energies = get_torus_config_energies(
		sims[0].n, widths, sims[0].h, sims[0].r, sims[0].energy
	)

	min_markers = [np.var(frame.stats["avg_radius"]) <= 1e-8 for frame in min_frames]
	max_markers = [np.var(frame.stats["avg_radius"]) <= 1e-8 for frame in max_frames]

	# Torus minimum energies used as reference.

	fig, ax = plt.subplots(figsize=(16, 8))
	#ax.plot(widths, nums)
	ax.plot(widths, [len(sim.frames) for sim in sims])
	fig.savefig(folder / "")

	fig, ax = plt.subplots(figsize=(16, 8))
	ax.plot(widths, torus_min_energies - torus_min_energies, color='C1')
	ax.plot(widths, min_energies - torus_min_energies, color='C0')
	ax.plot(widths, max_energies - torus_min_energies, color='C0', linestyle='dotted')
	#ax.plot(widths, torus_max_energies - torus_min_energies, color='C1', linestyle='dotted')

	for i, marker in enumerate(min_markers):
		if marker:
			ax.scatter(widths[i], min_energies[i]-torus_min_energies[i],
				marker='H', color="orange", s=20, zorder=4)
		else:
			ax.scatter(widths[i], min_energies[i]-torus_min_energies[i],
				marker='d', color="blue", s=20, zorder=4)

	for i, marker in enumerate(max_markers):
		if marker:
			ax.scatter(widths[i], max_energies[i]-torus_min_energies[i],
				marker='H', edgecolors="orange", s=20, facecolors='none', zorder=4)
		else:
			ax.scatter(widths[i], max_energies[i]-torus_min_energies[i],
				marker='d', edgecolors="blue", s=20, facecolors='none', zorder=4)

	
	ax.invert_xaxis()
	ax.title.set_text('Reduced Energy vs. Width')
	ax.set_xlabel("Width")
	ax.set_ylabel("Reduced Energy")
	ax.grid(zorder=0)

	ax.set_xticks([round(w,2) for w in widths[::-2]])
	#ax.set_yticks(np.arange(-920, 1120, 40))
	ax.set_xticklabels(ax.get_xticks(), rotation = 90)

	plt.tight_layout()

	fig.savefig(f"figures/WidthsEnergyComparison - N{sims[0].n}.png")

if __name__ == "__main__":
	os.environ["QT_LOGGING_RULES"] = "*=false"
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")