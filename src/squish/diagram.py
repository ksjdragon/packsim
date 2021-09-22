from __future__ import annotations
from typing import Tuple, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import os, math, random, time, pickle, scipy, numpy as np
from timeit import default_timer as timer

INT = np.int64
FLOAT = np.float64

SYMM = np.array([[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])


class Diagram():
	"""
	Class for generating diagrams.
	:param sim: [Simulation] Simulation class containing dynamics.
	:param diagrams: [np.ndarray] selects which diagrams to show.
	"""

	__slots__ = ['sim', 'diagrams', 'cumulative']

	def __init__(self, sim: Simulation, diagrams: np.ndarray, cumulative: bool = True):
		self.sim = sim
		self.diagrams = np.atleast_2d(diagrams)
		self.cumulative = cumulative


	def generate_frame(self, frame: int):
		"""
		Generates one frame for the plot.
		:param frame: [int] frame index to draw.
		:param scale: [float] how much of the domain to draw.
		:param area: [bool] set to false to not label areas.
		:param only: [bool] set to True to only render diagram.
		"""
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


	def voronoi_plot(self, i: int, ax):
		n,w,h = self.sim[i].n, self.sim[i].w, self.sim[i].h
		scale = 1.5
		area = n <= 60

		scipy.spatial.voronoi_plot_2d(self.sim[i].vor_data, ax, show_vertices=False,
										point_size = 7-n/100)
		ax.plot([-w, 2*w], [0, 0], 'r')
		ax.plot([-w, 2*w], [h, h], 'r')
		ax.plot([0,0], [-h, 2*h], 'r')
		ax.plot([w, w], [-h, 2*h], 'r')
		ax.axis('equal')
		ax.set_xlim([(1-scale)*w/2, (1+scale)*w/2])
		ax.set_ylim([(1-scale)*h/2, (1+scale)*h/2])
		ax.title.set_text("Voronoi Visualization")

		props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

		# if area:
		# 	global SYMM
		# 	for site_index in range(n):
		# 		for s in np.concatenate(([[0,0]], SYMM)):
		# 			txt = ax.text(*(site.vec + s*self.sim[i].dim),
		# 							str(round(site.cache("area"), 3)))
		# 			txt.set_clip_on(True)

		ax.text(0.05, 0.95, f'Energy: {self.sim[i].energy}', transform=ax.transAxes, fontsize=14,
				verticalalignment='top', bbox=props)


	def energy_plot(self, i: int, ax):
		ax.set_xlim([0, len(self.sim)])
		try:
			ax.plot([0, len(self.sim)], [self.sim[i].minimum, self.sim[i].minimum], 'red')
		except AttributeError:
			pass

		energies = [self.sim[j].energy for j in range(i+1)]
		ax.plot(list(range(i+1)), energies)
		ax.title.set_text('Energy vs. Time')
		max_value = round(self.sim[0].energy)
		min_value = round(self.sim[-1].energy)
		#diff = max_value-min_value
		#ax.set_yticks(np.arange(int(min_value-diff/5), int(max_value+diff/5), diff/25))
		ax.set_xlabel("Iterations")
		ax.set_ylabel("Energy")
		ax.grid()


	def site_areas_plot(self, i: int, ax):
		regular_area = self.sim[i].w*self.sim[i].h/self.sim[i].n
		y, x = self.sim.generate_bar_info("site_areas", i, self.cumulative,
										 	avg=True, reg=regular_area)

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


	def site_edge_count_plot(self, i: int, ax):
		y, x = self.sim.generate_bar_info("site_edge_count", i, self.cumulative,
											bounds=(1, 11), avg=True)

		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Edges per Site')
		ax.set_xlabel("Number of Edges")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.set_xticklabels([int(z) for z in x])
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))


	def site_isos_plot(self, i: int, ax):
		regular_area = self.sim[i].w*self.sim[i].h/self.sim[i].n
		regular_edge = math.sqrt(2*regular_area/(3*math.sqrt(3)))
		regular_isoparam = 4*math.pi*regular_area/(6*regular_edge)**2

		y, x = self.sim.generate_bar_info("site_isos", i, self.cumulative, bounds=(0,1),
											avg=True, reg=regular_isoparam)

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


	def site_energies_plot(self, i: int, ax):
		y, x = self.sim.generate_bar_info("site_energies", i, self.cumulative, avg=True)

		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Site Energies')
		ax.set_xlabel("Energy")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))


	def avg_radius_plot(self, i: int, ax):
		y, x = self.sim.generate_bar_info("avg_radius", i, self.cumulative, avg=True)
		ax.bar(x, y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Site Average Radii')
		ax.set_xlabel("Average Radius")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))


	def isoparam_avg_plot(self, i: int, ax):
		y, x = self.sim.generate_bar_info("isoparam_avg", i, self.cumulative, avg=True)

		ax.bar(x,y, width=0.8*(x[1]-x[0]))
		ax.title.set_text('Site Isoperimetric Averages')
		ax.set_xlabel("Isoperimetric Average")
		ax.set_ylabel("Average Occurances")
		ax.set_xticks(x)
		ax.ticklabel_format(useOffset=False)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))


	def edge_lengths_plot(self, i: int, ax):
		regular_area = self.sim[i].w*self.sim[i].h/self.sim[i].n
		regular_edge = math.sqrt(2*regular_area/(3*math.sqrt(3)))
		y, x = self.sim.generate_bar_info("edge_lengths", i, self.cumulative,
											30, avg=True, reg=regular_edge)

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


	def eigs_plot(self, i: int, ax):
		eigs = self.sim[i].stats["eigs"]
		ax.plot(list(range(len(eigs))), eigs, marker='o', linestyle='dashed', color='C0')
		ax.plot([0,len(eigs)], [0, 0], color="red")
		ax.title.set_text('Hessian Eigenvalues')
		ax.set_xlabel("")
		ax.set_ylabel("Value")


	def render_static(self, i: int, j: int = None, filename = None):
		"""
		Renders single frames.
		:param filename: [str] name of file.
		:param i: [int] index of frame to start rendering.
		:param j: [j] index of frame to stop rendering.
		:param only: [bool] set to True to only render diagram.
		"""
		if j is None:
			j = len(self.sim)-1

		length = j+1-i
		if length == 1:
			if filename is None:
				path = gen_filepath(self.sim, "png")
			else:
				path = f'figures/{filename}.png'

			self.generate_frame(i)
			plt.savefig(path)
			plt.close()

			print(f'Wrote to \"{path}\"')
		else:
			if filename is None:
				path = gen_filepath(self.sim, "")
			else:
				path = f'figures/{filename}'

			os.mkdir(path)
			for frame in range(i, j+1):
				self.generate_frame(frame)

				hashes = int(21*i/(j+1))
				print(f'Generating frames... |{"#"*hashes}{" "*(20-hashes)}|' + \
					f' {i+1}/{j+1} frames rendered.', flush=True, end='\r')

				plt.savefig(f'{path}/img{frame:03}.png')
				plt.close()

			print(flush=True)
			print(f'Wrote to folder \"{path}\"', flush=True)


	def render_video(self, time = 30, fps = None, filename = None):
		"""
		Renders plot(s) into image.
		:param scale: [float] how much of the domain to draw.
		:param area: [bool] set to false to not label area.
		:param filename: [str] name for static image.
		:param fps: [float] fps for image.
		:param only: [bool] set to True to only render diagram.
		"""
		if fps is None:
			if type(self.sim) == Flow:
				fps = min(len(self.sim)/time, 30)
			else:
				fps = 5

		step = len(self.sim)/(fps*time) if fps == 30 else 1
		# Iterate through desired frames.
		try:
			os.mkdir("figures/temp")
		except FileExistsError:
			pass

		frames = min(len(self.sim), int(fps * time))
		for j in range(frames):
			self.generate_frame(int(j*step))
			hashes = int(21*j/frames)
			print(f'Generating frames... |{"#"*hashes}{" "*(20-hashes)}|' + \
					f' {j+1}/{frames} frames rendered.', flush=True, end='\r')

			plt.savefig(f'figures/temp/img{j:03}.png')
			plt.close()

		print(flush=True)


		if filename is None:
			path = gen_filepath(self.sim, "mp4")
		else:
			path = f'figures/{filename}.mp4'

		# Convert to gif.
		print("Assembling MP4...", flush=True)
		os.system(f'ffmpeg -hide_banner -loglevel error -r {fps} -i figures/temp/img%03d.png' + \
				  f' -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -vf' + \
				  f' "scale=trunc(iw/2)*2:trunc(ih/2)*2" -f mp4 "{path}"')

		# Remove files.
		for j in range(frames):
			os.remove(f'figures/temp/img{j:03}.png')

		os.rmdir("figures/temp")
		print(f'Wrote to \"{path}\".', flush=True)
