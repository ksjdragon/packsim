from __future__ import annotations
from typing import List
import argparse, pickle, numpy as np, os
from pathlib import Path
import matplotlib.pyplot as plt

from squish import Simulation
from squish.common import OUTPUT_DIR


def main():
	parser = argparse.ArgumentParser("Graphs average defects at N.")
	parser.add_argument('sims_path', metavar='path/to/data',
						help="folder that contains simulation files at various Ns.")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")

	args = parser.parse_args()
	data = {}


	for file in Path(args.sims_path).iterdir():
		sim, frames = Simulation.load(file / "data.squish")
		avg_defects = 0
		count = 0

		for frame in frames:
			if np.var(frame["stats"]["avg_radius"]) > 1e-8:
				avg_defects += np.count_nonzero(frame["stats"]["site_edge_count"] != 6)
				count += 1


		avg_defects /= (1 if count == 0 else count)
		data[sim.domain.n] = avg_defects

	data = sorted(data.items())
	ns, defects = np.array([x[0] for x in data]), np.array([x[1] for x in data])

	corrected = []
	for i, x in enumerate(defects):
		if x == 0:
			corrected.append(defects[i+1])
		else:
			corrected.append(x)

	fig, ax = plt.subplots(1, 2, figsize=(16, 8))
	plt.subplots_adjust(.07, .12, .97, .9)

	fig.suptitle("Defects at N")

	m0, b0 = np.polyfit(ns, defects, 1)

	ax[0].plot(ns, defects)
	ax[0].plot(ns, m0*ns+b0, label=f"Slope: {m0:.5f}")
	ax[0].grid(zorder=0)
	ax[0].legend()
	ax[0].set_xlabel("N")
	ax[0].set_ylabel("Average Defects")

	x, y = np.log10(ns), np.log10(corrected)
	m, b = np.polyfit(x, y, 1)

	x2, y2 = x[14:], np.log10(defects[14:])
	m2, b2 = np.polyfit(x2, y2, 1)

	ax[1].plot(x, y, linestyle='dotted', color='C0')
	ax[1].plot(x, np.log10(defects))

	ax[1].plot(x, m*x+b, label=f"All N: {m:.5f}")
	ax[1].plot(x2, m2*x2+b2, label=f"N $\\geq$ 25: {m2:.5f}")
	ax[1].grid(zorder=0)
	ax[1].legend()
	ax[1].set_xlabel("log10 N")
	ax[1].set_ylabel("log10 Average Defects")




	fig.savefig(OUTPUT_DIR / "DefectsN.png")
	print(f"Wrote to {OUTPUT_DIR / 'DefectsN.png'}")


if __name__ == '__main__':
	os.environ["QT_log10GING_RULES"] = "*=false"
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")