from __future__ import annotations
from typing import List, Tuple, Dict
import argparse, math, numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from multiprocessing import Pool, cpu_count
from pathlib import Path

import squish.ordered as order
from squish import Simulation, DomainParams
from squish.common import OUTPUT_DIR


def order_process(domain: DomainParams) -> Tuple[float, float, float]:
	energies = []
	configs = order.configurations(domain)
	for config in configs:
		energies.append(2*domain.w*domain.h + \
			2*math.pi*domain.n*(domain.r**2 - 2*domain.r*order.avg_radius(domain, config)))

	return domain.w, min(energies), max(energies)


def get_ordered_energies(orig_domain: DomainParams, widths: np.ndarray) -> Dict:
	data = {}
	domains = [DomainParams(orig_domain.n, w, orig_domain.h, orig_domain.r) for w in widths]

	with Pool(cpu_count()) as pool:
		mins, maxes = {}, {}
		for i, res in enumerate(pool.imap_unordered(order_process, domains)):
			mins[res[0]] = res[1]
			maxes[res[0]] = res[2]

			hashes = int(21*i/len(widths))
			print(f'Generating at width {res[0]:.02f}... |{"#"*hashes}{" "*(20-hashes)}|' + \
				f' {i+1}/{len(widths)} completed.', flush=True, end='\r')

		print(flush=True)

		data["min"] = list([x[1] for x in sorted(mins.items())])
		data["max"] = list([x[1] for x in sorted(maxes.items())])

	return data


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

	with Pool(cpu_count()) as pool:
		for i, res in enumerate(pool.imap_unordered(eq_file_process, files)):
			data["all"][res[0]] = res[1]
			data["distinct"][res[0]] = res[2]

			hashes = int(21*i/len(files))
			print(f'Loading simulations... |{"#"*hashes}{" "*(20-hashes)}|' + \
				f' {i+1}/{len(files)} simulations loaded.', flush=True, end='\r')
		print(flush=True)

	sim, frames = Simulation.load(files[0] / 'data.squish')
	widths = np.asarray(sorted(data["all"]))
	domain = DomainParams(sim.domain.n, widths[-1], sim.domain.h, sim.domain.r)
	return data, widths, domain


def axis_settings(ax, widths):
	ax.invert_xaxis()
	ax.grid(zorder=0)
	ax.set_xticks([round(w,2) for w in widths[::-2]])
	ax.set_xticklabels(ax.get_xticks(), rotation = 90)
	plt.subplots_adjust(.07, .12, .97, .9)


def main():
	# Loading arguments.
	parser = argparse.ArgumentParser("Outputs width search data into diagrams")
	parser.add_argument('sims_path', metavar='path/to/data',
						help="folder that contains simulation files, or cached data file.")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")

	args = parser.parse_args()

	data, widths, domain = get_equilibria_data(Path(args.sims_path))
	order_data = get_ordered_energies(domain, widths)

	fig_folder = OUTPUT_DIR / Path(f"ShrinkEnergyComparison - N{domain.n}")
	fig_folder.mkdir(exist_ok=True)

	# Torus minimum energies used as reference.

	# Probability of disorder diagram.
	fig, ax = plt.subplots(figsize=(16, 8))
	all_disorder_count = []
	for width in widths:
		equal_shape = list([c[1] for c in data["all"][width]])
		all_disorder_count.append(100*equal_shape.count(False)/len(data["all"][width]))

	ax.plot(widths, all_disorder_count)
	axis_settings(ax, widths)
	ax.yaxis.set_major_formatter(mtick.PercentFormatter())
	ax.title.set_text(f"Probability of Disorder - N{domain.n}")
	ax.set_xlabel("Width")
	ax.set_ylabel("Disordered Equilibria")
	boa_y_min = round(min(all_disorder_count)/20)*20 - 5
	ax.set_yticks(np.arange(boa_y_min, 100.01, 2.5))
	fig.savefig(fig_folder / "Probability of Disorder.png")


	# Density of States diagram.
	fig, ax = plt.subplots(figsize=(16, 8))
	distinct_ordered, distinct_unordered = [], []
	for width in widths:
		equal_shape = list([c[1] for c in data["distinct"][width]])
		distinct_ordered.append(equal_shape.count(True))
		distinct_unordered.append(equal_shape.count(False))

	ax2 = ax.twinx()
	ax.plot(widths, distinct_unordered, label="Unordered Equilibria", color='C0')
	ax2.plot(widths, distinct_ordered, label="Ordered Equilibria", color='C1')
	axis_settings(ax, widths)
	ax.title.set_text(f"Density of States - N{domain.n}")
	ax.set_xlabel("Width")
	ax.set_ylabel("Number of States (Disordered)", color='C0')
	ax2.set_ylabel("Number of States (Ordered)", color='C1')

	dos_y_max_unorder = 1.05*max(distinct_unordered)
	dos_y_max_order = 1.05*max(distinct_ordered)
	ax.set_yticks(np.linspace(0, dos_y_max_unorder, 20).astype(int))
	#ax.set_yticks(np.arange(0, dos_y_max_unorder, round(dos_y_max_unorder/200, 1)*10))
	ax2.set_yticks(np.arange(0, dos_y_max_order))


	fig.savefig(fig_folder / "Density Of States.png")

	# Defect density diagram
	fig, ax = plt.subplots(figsize=(16, 8))

	defects = []
	for width in widths:
		defects.append(sum([c[2] for c in data["all"][width] if not c[1]])/len(data["all"][width]))

	ax.plot(widths, defects)
	axis_settings(ax, widths)
	ax.title.set_text(f"Average Defects - N{domain.n}")
	ax.set_xlabel("Width")
	ax.set_ylabel("Defects")
	ax.set_yticks(np.arange(0, 1+max(defects), 0.5))
	fig.savefig(fig_folder / "Defects.png")


	# Bifurcation diagram
	fig, ax = plt.subplots(figsize=(16, 8))

	ordered_energies, unordered_energies = [], []
	for width in widths:
		ordered_energies.append([c[0] for c in data["distinct"][width] if c[1]])
		unordered_energies.append([c[0] for c in data["distinct"][width] if not c[1]])

	for i in range(len(order_data["min"])):
		ordered_energies[i].append(order_data["min"][i])
		ordered_energies[i].append(order_data["max"][i])

	null_unorder = []
	for i, energies in enumerate(unordered_energies):
		if len(energies) == 0:
			null_unorder.append(i)
			energies.append(order_data["min"][i])

	min_order = np.asarray([min(width) for width in ordered_energies])
	max_order = np.asarray([max(width) for width in ordered_energies])
	min_unorder = np.asarray([min(width) for width in unordered_energies])
	max_unorder = np.asarray([max(width) for width in unordered_energies])


	min_unorder_off = min_unorder - min_order
	max_unorder_off = max_unorder - min_order
	ax.plot(widths, min_order - min_order, color='C1')
	#ax.plot(widths, max_order - offset, color='C1', linestyle='dotted')
	ax.plot(widths, min_unorder_off, color='C0')
	ax.plot(widths, max_unorder_off, color='C0', linestyle='dotted')
	axis_settings(ax, widths)

	for i in null_unorder:
		ax.scatter(widths[i], 0,
			marker='X', color="blue", s=50, zorder=4)
		# ax.scatter(widths[i], max_unorder[i] - offset[i],
		# 	marker='X', edgecolors="blue", facecolors='none', s=100, zorder=4)

	ax.title.set_text(f"Reduced Energy vs. Width - N{domain.n}")
	ax.set_xlabel("Width")
	ax.set_ylabel("Reduced Energy")
	bif_y_max = np.max(np.abs(np.concatenate((min_unorder_off, max_unorder_off))))
	bif_top = np.arange(0, bif_y_max, round(bif_y_max/20, -math.floor(math.log10(bif_y_max/20))))
	ax.set_yticks(np.concatenate((-bif_top[1:][::-1], bif_top)))
	fig.savefig(fig_folder / "Bifurcation.png")

	print(f"Wrote to {fig_folder}.")

if __name__ == '__main__':
	os.environ["QT_LOGGING_RULES"] = "*=false"
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")