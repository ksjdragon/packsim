from __future__ import annotations
from typing import List, Tuple, Dict
import argparse, numpy as np, os, pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from pathlib import Path

from squish import Simulation, DomainParams
from squish.common import OUTPUT_DIR


def eq_file_process(file: Path) -> Tuple[float, List[float], List[float]]:
	sim, frames = Simulation.load(file / 'data.squish')

	alls = []
	for frame_info in frames:
		alls.append([
			frame_info["energy"],
			np.var(frame_info["stats"]["avg_radius"]) <= 1e-8,
			np.count_nonzero(frame_info["stats"]["site_edge_count"] != 6)
		])

	sim, frames = Simulation.load(file / 'data.squish')
	sim.frames = list(frames)
	counts = sim.get_distinct()

	distincts = []
	for j, frame_info in enumerate(sim.frames):
		distincts.append([
			frame_info["energy"],
			np.var(frame_info["stats"]["avg_radius"]) <= 1e-8,
			np.count_nonzero(frame_info["stats"]["site_edge_count"] != 6),
			counts[j]
		])

	return sim.domain.w, alls, distincts


def get_equilibria_data(filepath: Path) -> Tuple[Dict, numpy.ndarray, DomainParams]:
	data = {"all": {}, "distinct": {}}
	files = list(Path(filepath).iterdir())
	sim, frames = Simulation.load(files[0] / 'data.squish')

	with Pool(cpu_count()) as pool:
		for i, res in enumerate(pool.imap_unordered(eq_file_process, files)):
			data["all"][res[0]] = res[1]
			data["distinct"][res[0]] = res[2]

			hashes = int(21*i/len(files))
			print(f'Loading simulations for N={sim.domain.n}... |{"#"*hashes}{" "*(20-hashes)}|' + \
				f' {i+1}/{len(files)} simulations loaded.', flush=True, end='\r')
		print(flush=True)

	widths = np.asarray(sorted(data["all"]))
	domain = DomainParams(sim.domain.n, widths[-1], sim.domain.h, sim.domain.r)
	return data, widths, domain


def main():
	# Loading arguments.
	parser = argparse.ArgumentParser("Outputs width search data into diagrams")
	parser.add_argument('sims_path', metavar='path/to/data',
						help="folder that contains simulation files of all searches for all N.")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")

	args = parser.parse_args()

	# with open("testing.pkl", "rb") as f:
	# 	disorder_dict = pickle.load(f)
	# 	widths = np.linspace(3.0, 10.0, 141)
	# 	min_n, max_n = 60, 80

	disorder_dict = {}
	for file in Path(args.sims_path).iterdir():
		sim_data, widths, domain = get_equilibria_data(file)

		disorder_count = []
		for width in widths:
			equal_shape = list([c[1] for c in sim_data["all"][width]])
			disorder_count.append(100*equal_shape.count(False)/len(sim_data["all"][width]))

		disorder_dict[domain.n] = disorder_count

	min_n, max_n = min(disorder_dict), max(disorder_dict)
	filepath = f"Disorder Heatmap N{min_n}-{max_n}"

	# with open("testing.pkl", "wb") as f:
	# 	pickle.dump(disorder_dict, f, pickle.HIGHEST_PROTOCOL)

	disorder_arr = np.zeros((max_n-min_n+1, len(widths)))
	for key, value in disorder_dict.items():
		disorder_arr[key-min_n] = np.asarray(value)

	fig, ax = plt.subplots(figsize=(12, 8))

	extent = [min(widths), max(widths), min_n, max_n+1]
	ax.imshow(disorder_arr, cmap='plasma', interpolation='nearest', aspect='auto', extent=extent)

	ax.invert_xaxis()
	ax.set_xticks([round(w,2) for w in widths[::-2]])
	ax.set_xticklabels(ax.get_xticks(), rotation = 90)
	ax.set_yticks(list(range(min_n, max_n+1)))
	plt.subplots_adjust(.07, .12, .97, .9)

	ax.title.set_text(filepath)
	ax.set_xlabel("Width")
	ax.set_ylabel("Number of Sites")
	fig.savefig(OUTPUT_DIR / filepath)

	print(f"Wrote to {OUTPUT_DIR / filepath}.")


if __name__ == '__main__':
	os.environ["QT_LOGGING_RULES"] = "*=false"
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")