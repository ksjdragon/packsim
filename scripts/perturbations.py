from __future__ import annotations
from typing import List
import argparse, pickle, numpy as np, os
from pathlib import Path
import matplotlib.pyplot as plt

from squish import Simulation
from squish.common import OUTPUT_DIR


def main():
	parser = argparse.ArgumentParser("Graphs perturbation graphs for a collection of simulations.")
	parser.add_argument('sims_path', metavar='path/to/data',
					help="folder that contains simulations of perturbations from an equilibrium.")
	parser.add_argument('end_path', metavar='path/to/equilbrium',
					help="NumPy binary (.npy) file that contains the equilibrium to compare to.")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")

	args = parser.parse_args()

	end = np.load(args.end_path)

	data = {}
	for file in Path(args.sims_path).iterdir():
		k = float(file.name.split('k')[-1])
		delta = 10**k

		sim, frames = Simulation.load(file / 'data.squish')
		data[delta] = {"norm": [], "time": [], "k": k}

		for i, frame in enumerate(frames):
			adjusted = frame["arr"] + (end[0] - frame["arr"][0])

			data[delta]["norm"].append(np.linalg.norm(adjusted - end))
			data[delta]["time"].append(sim.step_size * i)

	fig, ax = plt.subplots(figsize=(12, 8))
	plt.subplots_adjust(.07, .12, .97, .9)

	for delta in sorted(data):
		ax.plot(np.log10(np.array(data[delta]["time"])+1), np.log10(data[delta]["norm"]),
				label=f"k = {data[delta]['k']}")

	fig.suptitle("Equilibrium Perturbations")
	ax.grid(zorder=0)
	#ax.set_xlim([0, 5])
	ax.legend()
	ax.set_xlabel("Log Time")
	ax.set_ylabel("Log L2 Norm of Difference")

	fig.savefig(OUTPUT_DIR / "Equilibrium Perturbations.png")
	print(f"Wrote to {OUTPUT_DIR / 'Equilibrium Perturbations.png'}")


if __name__ == '__main__':
	os.environ["QT_LOGGING_RULES"] = "*=false"
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")