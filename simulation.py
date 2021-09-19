from __future__ import annotations
from typing import Tuple, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import os, math, random, time, pickle, scipy, numpy as np

from packsim_core import VoronoiContainer, AreaEnergy, RadialALEnergy, RadialTEnergy
from timeit import default_timer as timer


INT = np.int64
FLOAT = np.float64

SYMM = np.array([[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])

def gen_filepath(sim: Simulation, ext: str, parent_dir='figures') -> str:
	"""
	Generates a filename based on the simulation.
	:param sim: [Simulation] simulation to generate file for.
	:param ext: [str] file extension.
	:return: [str] string for filename.
	"""
	energy = {AreaEnergy: "Area", RadialALEnergy: "Radial[AL]", 
				RadialTEnergy: "Radial[T]"}[sim.energy]
	mode = {Flow: "F", Search: "T", Shrink: "S"}[type(sim)]

	base_filename = f'{energy}{mode} - N{sim[0].n}R{round(sim[0].r, 1) :.1f} - {round(sim[0].w, 2):.2f}x{round(sim[0].h, 2):.2f}'
	base_path = f'{parent_dir}/{base_filename}'
	
	i = 1
	if ext == "":
		path = base_path
		while os.path.isdir(path):
			path = base_path + f'({i})'
			i += 1
	else:
		path = base_path + "." + ext
		while os.path.isfile(path):
			path = base_path + f'({i}).{ext}'
			i += 1
	return path


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
		diff = max_value-min_value
		ax.set_yticks(np.arange(int(min_value-diff/5), int(max_value+diff/5), diff/25))
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


class Simulation:
	"""
	Class for running simulations.
	:param n: [int] how many sites to generate.
	:param w: [float] width of the bounding domain.
	:param h: [float] height of the bounding domain.
	:param r: [float] radius of zero energy circle.
	:param energy: energy to use to calculate. Can 
	pass in class directly or use string.
	"""

	__slots__ = ['n', 'w', 'h', 'r', 'energy', 'frames']

	def __init__(self, n: int, w: float, h: float, r: float, energy: str):
		self.n, self.w, self.h, self.r = int(n), w, h, r
		self.frames = []
		if self.n < 2:
			raise ValueError("Number of objects should be larger than 2!")

		if self.w <= 0:
			raise ValueError("Width needs to be nonzero and positive!")

		if self.h <= 0:
			raise ValueError("Height needs to be nonzero and positive!")

		if isinstance(energy, str):
			try:
				self.energy = {"area": AreaEnergy, "radial-al": RadialALEnergy,
								"radial-t" : RadialTEnergy}[energy.lower()]
			except KeyError:
				raise ValueError("Invalid Energy!")
		else:
			if energy not in [AreaEnergy, RadialALEnergy, RadialTEnergy]:
				raise ValueError("Invalid Energy!")
			self.energy = energy

	def __getitem__(self, key: int) -> VoronoiContainer:
		return self.frames[key]

	def __len__(self):
		return len(self.frames)


	def initialize(self, points = None, torus = None, jitter = 0):
		"""
		Initializes the simulation
		:param points: Can be multiple types. Takes list-like data types.
		:param torus: Used or generating torus points. L value.
		:param jitter: [int] Add random*jitter movement to initial data.
		"""
		self.add_frame(points, torus, jitter)


	def add_frame(self, points = None, torus = None, jitter = 0.0):
		"""
		Adds a new frame to this simulation.
		:param points: Can be multiple types. Takes list-like data types.
		:param torus: Used or generating torus points. L value.
		:param jitter: [int] Add random*jitter movement to initial data.
		"""
		dim = np.array([self.w, self.h])
		if not (points is None):
			points = np.asarray(points)
			if points.shape[1] != 2:
				raise ValueError("Improper shape, points are 2 dimensional.")
		elif not torus is None:
			points = Simulation.torus_sites(self.n, self.w, self.h, torus)
		else:
			points = dim * np.random.random_sample((self.n, 2))

		points += (jitter*np.random.random_sample((self.n, 2)).astype(FLOAT)) % dim

		self.frames.append(self.energy(self.n, self.w, self.h, self.r, points))


	def generate_bar_info(self, stat: str, i: int, cumulative: bool, bins: int = 10,
							 bounds: Tuple[float] = None, avg: bool = False, reg = None) -> Tuple:
		"""
		Gets the bar info for matplotlib from the ith to jth frame.
		:param stat: [str] name of statistic to obtain.
		:param i: [int] frame to obtain
		:param cumulative: [bool] Will obtain all stats up to the ith frame if True.
		:param bins: [int] number of bins for the bar graph.
		:param bound: [Tuple[float]] lower and upper bounds for the bins. If not set,
								automatically take the min and max value.
		:param avg: [bool] Averages the counts over the number of frames if True.
		:param mark: If not None, set a specific marker.
		:return: [Tuple] returns a tuple of labels, values, and colors.
		"""
		if cumulative:
			values = np.concatenate([f.stats[stat] for f in self.frames[:(i+1)]])
		else:
			values = self.frames[i].stats[stat]

		#bins = 9
		if np.var(values) <= 1e-8:
			hist = np.zeros((bins,))
			val = np.average(values)
			hist[(bins+1) // 2 - 1] = len(values)
			bin_list = np.linspace(0, val, bins//2+1, endpoint=True)
			bin_list = np.concatenate((bin_list, (bin_list+val)[1:]))
			return hist, bin_list[not (bins%2):]

		hist, bin_edges = np.histogram(values, bins=bins, range=bounds)
		bin_list = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)] 

		if avg and cumulative:
			return hist / (i+1), bin_list

		return hist, bin_list

		# colors = ["C0"]*bins
		# if reg >= lb and reg <= ub:
		# 	colors[int((reg-lb)*bins/diff)] = "C3"
		
		# return (labels, count, colors)


	def get_distinct(self):
		distinct_hists = []
		distinct_avg_radii = []
		new_frames = []

		for frame in self.frames:
			if np.var(frame.stats["avg_radius"]) <= 1e-8:
				avg_radii = np.average(frame.stats["avg_radius"])
				if not np.any(np.isclose(avg_radii, distinct_avg_radii, atol=1e-5)):
					distinct_avg_radii.append(avg_radii)
					new_frames.append(frame)
			else:
				new_hist = np.histogram(frame.stats["avg_radius"], bins=8)[0]

				is_in = False
				for hist in distinct_hists:
					if np.all(hist == new_hist):
						is_in = True
						break

				if not is_in:
					distinct_hists.append(new_hist)
					new_frames.append(frame)

		self.frames = new_frames


	def save(self, filename: str = None):
		"""
		Saves the points at every point into a file.
		:filename: [str] name of the file
		"""
		if filename is None:
			path = gen_filepath(self, "sim", "simulations")
		else:
			path = f'new_simulations/{filename}.sim'

		all_info = []
		for frame in self.frames:
			frame_info = dict()
			frame_info["arr"] = frame.site_arr
			frame_info["energy"] = {AreaEnergy: "area", RadialALEnergy: "radial-al",
									RadialTEnergy: "radial-t"}[self.energy]
			frame_info["params"] = (frame.n, frame.w, frame.h, frame.r)
			all_info.append(frame_info)

		class_name = {Flow: "flow", Search: "search", Shrink: "shrink"}[self.__class__]

		with open(path, 'wb') as output:
			pickle.dump((all_info, class_name), output, pickle.HIGHEST_PROTOCOL)
		print("Wrote to " + path, flush=True)


	@staticmethod
	def load(filename: str) -> Simulation:
		"""
		Loads the points at every point into a file.
		:param filename: [str] name of the file
		"""
		frames = []
		with open(filename, 'rb') as data:
			all_info, sim_class = pickle.load(data)
			sim_class = {"flow": Flow, "search": Search, "shrink": Shrink}[sim_class]
			sim = sim_class(*all_info[0]["params"], all_info[0]["energy"], 0,0,0,0)
			for frame_info in all_info:
				frames.append(sim.energy(*frame_info["params"], frame_info["arr"]))
				#frames[-1].stats = frame_info["stats"]

			sim.frames = frames
		return sim	


	@staticmethod
	def torus_sites(n: int, w: float, h: float, L: Tuple[int]):
		"""
		Returns the points when you wrap a line
		around a torus, like in the periodic domain.
		:param n: [int] amount of points.
		:param w: [float] width of the domain.
		:param h: [float] height of the domain.
		:param L: [Tuple[int]] L = (u,v)
		"""
		dim = np.array([[w, h]])
		L = np.array(L)
		return (np.array([1,1])/2 + np.concatenate([(i*dim*L/n) for i in range(n)])) % dim 


class Flow(Simulation):
	"""
	Class for finding an equilibrium from initial points.
	:param n: [int] how many sites to generate.
	:param w: [float] width of the bounding domain.
	:param h: [float] height of the bounding domain.
	:param r: [float] radius of zero energy circle.
	:param energy: [str] energy to use to calculate.
	:param thres: [float] threshold for close enough to equilibrium.
	:param step_size: [float] size to step by for iteration.
	""" 

	__slots__ = ['thres', 'step_size']

	def __init__(self, n: int, w: float, h: float, r: float, energy: str, thres: float,
					step_size: float):
		super().__init__(n, w, h, r, energy)
		self.thres, self.step_size = thres, step_size 


	def run(self, log, log_steps):
		"""
		Runs the simulation.
		:param log: [bool] will log if True.
		"""
		if log:
			print(f'Find - N = {self.n}, R = {self.r}, {self.w} X {self.h}', flush=True)
		i, grad_norm = 0, float('inf')

		trial = 2
		while grad_norm > self.thres:	# Get to threshold.
			# Iterate and generate next frame using RK-3
			start = timer()
			change, grad = self.frames[i].iterate(self.step_size)
			new_frame = self.energy(self.n, self.w, self.h, self.r,
							self.frames[i].add_sites(change))
			grad_norm = np.sum(np.absolute(grad))/self.n
			end = timer()

			if new_frame.energy < self.frames[i].energy:	# If energy decreases.
				if trial < 20:	# Try increasing step size for 10 times.
					factor = 1 + .1**trial

					test_frame = self.energy(self.n, self.w, self.h, self.r,
											self.frames[i].add_sites(change*factor))
					# If increased step has less energy than original step.
					if test_frame.energy < new_frame.energy:
						self.step_size *= factor
						trial = max(2, trial-1)
						new_frame = test_frame
					else:	# Otherwise, increases trials, and use original.
						trial += 1
			else:	# Step size too large, decrease and reset trial counter.
				trial = 2
				shrink_factor = 1.5
				new_frame = self.energy(self.n, self.w, self.h, self.r,
								self.frames[i].add_sites(change/shrink_factor))
				self.step_size /= shrink_factor

			#self.step_size *= abs(.01/np.linalg.norm(error))**(1/3)
			#self.step_size = max(10e-4, self.step_size)
			self.frames.append(new_frame)

			i += 1
			if(log and i % log_steps == 0): 
				print(f'Iteration: {i:05} | Energy: {self.frames[i].energy: .5f}' + \
				 	  f' | Gradient: {grad_norm:.8f} | Step: {self.step_size: .5f} | ' + \
				 	  f'Time: {end-start: .3f}', flush=True)


class Search(Simulation):
	"""
	Class for traversing to other equilibriums from an equilbrium.
	:param n: [int] how many sites to generate.
	:param w: [float] width of the bounding domain.
	:param h: [float] height of the bounding domain.
	:param r: [float] radius of zero energy circle.
	:param energy: [str] energy to use to calculate.
	:param thres: [float] threshold for when to stop.
	:param kernel_step: [float] size to step when jumping off kernel.
	:param iter_step: [float] size to step by for iteration.
	:param iter: [int] number of iterations
	""" 

	__slots__ = ['thres', 'iter_step', 'kernel_step', 'iter']

	def __init__(self, n: int, w: float, h: float, r: float, energy: str, thres,
					iter_step: float, kernel_step: float, iter: int):
		super().__init__(n, w, h, r, energy)
		self.thres, self.iter = thres, iter
		self.kernel_step, self.iter_step = kernel_step, iter_step


	def run(self, log, log_steps):
		"""
		Runs the simulation.
		:param log: [bool] will log if True.
		"""

		if log:
			print(f'Travel - N = {self.n}, R = {self.r}, {self.w} X {self.h}', flush=True)

		dim = np.array([self.w, self.h])
		fixed = random.randint(0, self.n-1)
		center = dim / 2
		new_sites = self.frames[0].site_arr
		# Move fixed point to center.
		for i in range(self.iter):
			# Get to equilibrium.
			sim = Flow(self.n, self.w, self.h, self.r, self.energy, self.thres, 
							self.iter_step)
			sim.initialize(new_sites)
			sim.run(log, log_steps)

			self.frames[i] = sim[-1] # Replace frame with equilibrium frame.
			if log:
				print(f'Equilibrium: {i:04}\n', flush=True)
			# Calculate kernel, and travel in some direction.

			if self.kernel_step > 0:
				hess = self.frames[i].hessian(10e-5)
				ns = scipy.linalg.null_space(hess, 10e-4).T

				#self.frames[i].get_statistics()
				eigs = np.sort(np.linalg.eig(hess)[0])
				self.frames[i].stats["eigs"] = eigs

				zero_eigs = np.count_nonzero(np.isclose(eigs, np.zeros((len(eigs),)), atol=1e-4))
				if zero_eigs != 2:
					print("WARNING, 0 EIGS NOT 2", zero_eigs)

				if i == self.iter-1:
					break

				if len(ns) <= 2:
					new_sites = dim * np.random.random_sample((self.n, 2))
				else:
					vec = ns[random.randint(0, len(ns)-1)]	# Choose random vector
					new_sites = self.frames[i].add_sites(self.kernel_step*vec.reshape((self.n, 2)))
			
				new_sites += (center - new_sites[fixed]) % 	dim # Offset
			self.frames.append(None)


class Shrink(Simulation):
	"""
	Class for traversing to other equilibriums from an equilbrium.
	:param n: [int] how many sites to generate.
	:param w: [float] width of the bounding domain.
	:param h: [float] height of the bounding domain.
	:param r: [float] radius of zero energy circle.
	:param energy: [str] energy to use to calculate.
	:param thres: [float] threshold for when to stop.
	:param w_change: [float] percent to change w each iteration.
	:param stop_w: [int] percentage at which to stop iterating.
	:param step_size: [float] size to step by for iteration.
	""" 

	__slots__ = ['thres', 'w_change', 'stop_w', 'step_size']

	def __init__(self, n: int, w: float, h: float, r: float, energy: str, thres: float, 
				 step_size: float, w_change: float, stop_w: float):
		super().__init__(n, w, h, r, energy)
		self.thres, self.step_size = thres, step_size
		self.w_change, self.stop_w = w*w_change, w*stop_w


	def run(self, log, log_steps):
		"""
		Runs the simulation.
		:param log: [bool] will log if True.
		"""
		if log:
			print(f'Shrink - N = {self.n}, R = {self.r}, {self.w} X {self.h}', flush=True)

		while self.w >= self.stop_w:
			# Get to equilibrium.
			sim = Flow(self.n, self.w, self.h, self.r, self.energy, self.thres, 
							self.step_size)
			sim.initialize(self.frames[-1].site_arr)
			sim.run(log, log_steps)

			self.frames.append(sim[-1]) # Replace frame with equilibrium frame.
			self.frames[-1].get_statistics()
			if log:
				print(f'Width: {self.w:.4f}\n')
			
			self.w -= self.w_change

		del self.frames[0]
