from __future__ import annotations
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt, numpy as np, os
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.spatial import Voronoi, voronoi_plot_2d
from multiprocessing import Pool, cpu_count

from .common import DomainParams

SYMM = np.array([[0,0], [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])

class SimData:
	"""Stores diagram information for a simulation.

	Attributes:
		path (Path): path to output directory.
		domains (List[DomainParams]): domain parameters from simulation frames,
		energies (List[float]): energy from simulation frames.
		voronois (List[Voronoi]): voronoi information from scipy from simulation frames.
		stats (List[numpy.ndarray]): statistics from simulation frames.

	"""

	__slots__ = ['path', 'domains', 'energies', 'voronois', 'stats']


	def __init__(self, sim: Simulation) -> None:
		self.path = sim.path
		self.domains = list([DomainParams(s.n, s.w, s.h, s.r) for s in sim])
		self.energies = list([s.energy for s in sim])
		self.voronois = list([s.vor_data for s in sim])
		self.stats = list([s.stats for s in sim])


	def __len__(self) -> int:
		return len(self.domains)


	def hist(self, stat: str, i: int, bins: int = 10, bounds: Optional[Tuple[float, float]] = None,
				cumul: bool = False, avg: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]:
		"""Generates a histogram from the selected data.

		Arguments:
			stat (str): name of data to obtain
			i (int): which frame to select from
			bins (int): number of bins for the histogram.
			bounds (Optional[Tuple[float, float]]): upper and lower bounds of the histogram.
				this will automatically take the minimum and maximum value of not set.
			cumul (bool): aggregates all data up to frame i if True.
			avg (bool): will average the data based on number of frames if True.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]: the histogram and its bins.

		"""
		if cumul:
			values = np.concatenate([f[stat] for f in self.stats[:(i+1)]])
		else:
			values = self.stats[i][stat]

		if np.var(values) <= 1e-8:
			hist = np.zeros((bins,))
			val = np.average(values)
			hist[(bins+1) // 2 - 1] = len(values)
			bin_list = np.linspace(0, val, bins//2+1, endpoint=True)
			bin_list = np.concatenate((bin_list, (bin_list+val)[1:]))
			return hist, bin_list[not (bins%2):]

		hist, bin_edges = np.histogram(values, bins=bins, range=bounds)
		bin_list = np.array([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])

		if avg and cumul:
			return hist / (i+1), bin_list

		return hist, bin_list


class Diagram:
	"""Class for generating diagrams.

	Attributes:
		sim (SimData): the simulation data that contains all the frames and information.
		diagrams (numpy.ndarray): array that selects which diagrams to show.
		cumulative (bool): selects whether or not graph statistics are cumulative.

	"""

	__slots__ = ['sim', 'diagrams', 'cumulative']


	def __init__(self, sim: Simulation, diagrams: np.ndarray, cumulative: bool = False) -> None:
		self.sim = SimData(sim)
		self.diagrams = np.atleast_2d(diagrams)
		self.cumulative = cumulative


	def generate_frame(self, frame: int, mode: str, fol: str) -> None:
		if mode not in ["save", "open"]:
			raise ValueError("Not a valid mode for diagrams!")

		shape = self.diagrams.shape
		fig, axes = plt.subplots(*shape, figsize=(shape[1]*8, shape[0]*8))
		if self.diagrams.shape == (1,1):
			getattr(self, str(self.diagrams[0][0]) + '_plot')(frame, axes)
		else:
			axes = np.atleast_2d(axes)
			it = np.nditer(self.diagrams, flags=["multi_index"])
			for diagram in it:
				if diagram == "":
					continue
				getattr(self, str(diagram) + '_plot')(frame, axes[it.multi_index])

		plt.tight_layout()

		if mode == "save":
			plt.savefig(self.sim.path / fol / f"img{frame:05}.png")
			plt.close(fig)
		elif mode == "show":
			plt.show()


	def voronoi_plot(self, i: int, ax: AxesSubplot) -> None:
		domain = self.sim.domains[i]
		n, w, h = domain.n, domain.w, domain.h
		scale = 1.5
		area = n <= 60

		voronoi_plot_2d(self.sim.voronois[i], ax, show_vertices=False, point_size = 7-n/100)
		ax.plot([-w, 2*w], [0, 0], 'r')
		ax.plot([-w, 2*w], [h, h], 'r')
		ax.plot([0,0], [-h, 2*h], 'r')
		ax.plot([w, w], [-h, 2*h], 'r')
		ax.axis('equal')
		ax.set_xlim([(1-scale)*w/2, (1+scale)*w/2])
		ax.set_ylim([(1-scale)*h/2, (1+scale)*h/2])
		ax.title.set_text("Voronoi Visualization")

		props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

		defects = {5: {"x": [], "y": []}, 7: {"x": [], "y": []}}

		for j in range(n):
			for s in SYMM:
				vec = (self.sim.voronois[i].points[j] + s*self.sim.domains[i].dim)

				if area:
					txt = ax.text(*vec, str(round(self.sim.stats[i]["site_areas"][j], 3)))
					txt.set_clip_on(True)

				if self.sim.stats[i]["site_edge_count"][j] == 5:
					defects[5]["x"].append(vec[0])
					defects[5]["y"].append(vec[1])
				elif self.sim.stats[i]["site_edge_count"][j] == 7:
					defects[7]["x"].append(vec[0])
					defects[7]["y"].append(vec[1])

		ax.scatter(defects[5]["x"], defects[5]["y"], marker="p", color="red")
		ax.scatter(defects[7]["x"], defects[7]["y"], marker="*", color="red")


		ax.text(0.05, 0.95, f'Energy: {self.sim.energies[i]}', transform=ax.transAxes, fontsize=14,
				verticalalignment='top', bbox=props)


	def energy_plot(self, i: int, ax: AxesSubplot) -> None:
		ax.set_xlim([0, len(self.sim)])

		energies = self.sim.energies[:(i+1)]
		ax.plot(list(range(i+1)), energies)
		ax.title.set_text('Energy vs. Time')
		# max_value = round(self.sim[0].energy)
		# min_value = round(self.sim[-1].energy)
		#diff = max_value-min_value
		#ax.set_yticks(np.arange(int(min_value-diff/5), int(max_value+diff/5), diff/25))
		ax.set_xlabel("Iterations")
		ax.set_ylabel("Energy")
		ax.grid()


	def site_areas_plot(self, i: int, ax: AxesSubplot) -> None:
		y, x = self.sim.hist("site_areas", i, cumul=self.cumulative, avg=True)

		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Site Areas')
		ax.set_xlabel("Area")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		# for xtick, color in zip(ax.get_xticklabels(), areas_bar[2]):
		# 	if color != 'C0':
		# 		xtick.set_color(color)


	def site_edge_count_plot(self, i: int, ax: AxesSubplot) -> None:
		y, x = self.sim.hist("site_edge_count", i, bounds=(1, 11), cumul=self.cumulative, avg=True)

		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Edges per Site')
		ax.set_xlabel("Number of Edges")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.set_xticklabels([int(z) for z in x])
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))


	def site_isos_plot(self, i: int, ax: AxesSubplot) -> None:
		y, x = self.sim.hist("site_isos", i, bounds=(0,1), cumul=self.cumulative, avg=True)

		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Isoparametric Values')
		ax.set_xlabel("Isoparametric Value")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		# for xtick, color in zip(ax.get_xticklabels(), isoparam_bar[2]):
		# 	if color != 'C0':
		# 		xtick.set_color(color)


	def site_energies_plot(self, i: int, ax: AxesSubplot) -> None:
		y, x = self.sim.hist("site_energies", i, cumul=self.cumulative, avg=True)

		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Site Energies')
		ax.set_xlabel("Energy")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))


	def avg_radius_plot(self, i: int, ax: AxesSubplot) -> None:
		y, x = self.sim.hist("avg_radius", i, cumul=self.cumulative, avg=True)

		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Site Average Radii')
		ax.set_xlabel("Average Radius")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))


	def isoparam_avg_plot(self, i: int, ax: AxesSubplot) -> None:
		y, x = self.sim.hist("isoparam_avg", i, cumul=self.cumulative, avg=True)

		ax.bar(x,y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Site Isoperimetric Averages')
		ax.set_xlabel("Isoperimetric Average")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))


	def edge_lengths_plot(self, i: int, ax: AxesSubplot) -> None:
		y, x = self.sim.hist("edge_lengths", i, 30, cumul=self.cumulative, avg=True)

		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Edge Lengths')
		ax.set_xlabel("Length")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.set_xticklabels(ax.get_xticks(), rotation = 90)
		ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
		#ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		# for xtick, color in zip(ax.get_xticklabels(), lengths_bar[2]):
		# 	if color != 'C0':
		# 		xtick.set_color(color)


	def eigs_plot(self, i: int, ax: AxesSubplot) -> None:
		try:
			eigs = self.sim.stats[i]["eigs"]
			ax.plot(list(range(len(eigs))), eigs, marker='o', linestyle='dashed', color='C0')
			ax.plot([0,len(eigs)], [0, 0], color="red")
		except KeyError:
			ax.text(0.5, 0.5, "Not Computed")

		ax.title.set_text('Hessian Eigenvalues')
		ax.set_xlabel("")
		ax.set_ylabel("Value")


	def render_frames(self, frames: List[int], fol: str = 'frames') -> None:
		(self.sim.path / fol).mkdir(exist_ok=True)
		combo_list = []
		for i in range(cpu_count()):
			combo_list.append((self, frames[:int((i+1)*len(frames)/cpu_count())],
								fol, len(frames)))

		with Pool(cpu_count()) as pool:
			for _ in pool.imap_unordered(render_frame_range, combo_list):
				pass

		print(flush=True)
		print(f'Wrote to \"{self.sim.path / fol}\".', flush=True)

	def render_video(self, time: int, mode: str) -> None:
		if mode not in ["use_all", "sample"]:
			raise ValueError("Not a valid mode for videos!")

		if mode == "use_all":
			frames = list(range(len(self.sim)))
		elif mode == "sample":
			fps = 30
			if len(self.sim) < fps*time :
				frames = list(range(len(self.sim)))
				fps = len(self.sim)/time
			else:
				frames = list(np.round(np.linspace(0, len(self.sim), fps*time)).astype(int))
				print(frames)

		self.render_frames(frames, 'temp')
		path = self.sim.path / 'simulation.mp4'

		print("Assembling MP4...", flush=True)
		os.system(f'ffmpeg -hide_banner -loglevel error -r {fps} -i' + \
				f' \"{self.sim.path}/temp/img%05d.png\"' + \
				f' -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -vf' + \
				f' "scale=trunc(iw/2)*2:trunc(ih/2)*2" -f mp4 "{path}"')

		# Remove files.
		for i in frames:
			os.remove(self.sim.path / f"temp/img{i:05}.png")

		os.rmdir(self.sim.path / 'temp')
		print(f'Wrote to \"{path}\".', flush=True)


def render_frame_range(combo: Tuple[Diagram, List[int], str, int]) -> None:
	self, frames, fol, num_frames = combo
	for frame in frames:
		self.generate_frame(frame, "save", fol)
		i = len(list((self.sim.path / fol).iterdir()))
		hashes = int(21*i/num_frames)
		print(f'Generating frames... |{"#"*hashes}{" "*(20-hashes)}|' + \
				f' {i}/{num_frames} frames rendered.', flush=True, end='\r')
