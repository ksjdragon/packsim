from __future__ import annotations
from typing import List
import argparse, pickle, numpy as np, os
from pathlib import Path
import matplotlib.pyplot as plt


def main():
	parser = argparse.ArgumentParser("Graphs convergence graphs for a collection of simulations.")
	parser.add_argument('sims_path', metavar='path/to/data',
						help="folder that contains simulation files at various step sizes.")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")

	args = parser.parse_args()
	data = {}

	for file in Path(args.sims_path).iterdir():
		with open(file, "rb") as f:
			all_info, _ = pickle.load(f)

		step = float(file.name[:-4].split("-")[1])
		data[step] = {"times": [], "values": [], "diffs": []}
		for i, frame_info in enumerate(all_info):
			data[step]["times"].append(step*i)
			data[step]["values"].append(np.linalg.norm(frame_info["arr"]))
			data[step]["diffs"].append(np.linalg.norm(all_info[-1]["arr"] - frame_info["arr"]))

	fig, ax = plt.subplots(1, 2, figsize=(16, 8))
	plt.subplots_adjust(.07, .12, .97, .9)

	for step, d in data.items():
		ax[0].plot(d["times"], d["values"], label=step)
		ax[1].plot(d["times"], d["diffs"], label=step)

	fig.suptitle("Equilibrium Convergence")
	ax[0].grid(zorder=0)
	ax[0].legend()
	ax[0].set_xlabel("Time")
	ax[0].set_ylabel("L2 Norm of Sites")

	ax[1].grid(zorder=0)
	ax[1].legend()
	ax[1].set_xlabel("Time")
	ax[1].set_ylabel("L2 Norm of Difference")

	fig.savefig("figures/Equilibrium Convergence.png")


if __name__ == '__main__':
	os.environ["QT_LOGGING_RULES"] = "*=false"
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")