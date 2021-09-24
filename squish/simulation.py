from __future__ import annotations
from typing import Optional

import pickle, numpy as np
from scipy.linalg import null_space
from timeit import default_timer as timer

from .common import DomainParams, Energy, generate_filepath, OUTPUT_DIR


class Simulation:
	"""Generic container for simulations.

	Attributes:
		domain (DomainParams): Domain Parameters for this simulation.
		energy (Energy): energy being used for caluclations.
		path (Path): Path to location of where to store simulation files.
		frames (List[VoronoiContainer]): Stores frames of the simulation.

	"""

	__slots__ = ['domain', 'energy', 'path', 'frames']

	def __init__(self, domain: DomainParams, energy: Energy, name: Optional[str] = None) -> None:
		self.domain, self.energy = domain, energy
		self.frames = []

		if name is None:
			self.path = generate_filepath(self, OUTPUT_DIR)
		else:
			self.path = OUTPUT_DIR / name


	def __iter__(self) -> Iterator:
		return iter(self.frames)

	def __getitem__(self, key: int) -> Energy:
		return self.frames[key]


	def __len__(self) -> int:
		return len(self.frames)


	def add_frame(self, points: Optional[numpy.ndarray]) -> None:
		if points is None:
			points = np.random.random_sample((self.domain.n, 2)) * self.domain.dim
		else:
			if points.shape[1] != 2 or len(points.shape) > 2:
				raise ValueError("Sites should be 2 dimensional!")

			if points.shape[0] != self.domain.n:
				raise ValueError("Number of sites provided do not match the array!")

		self.frames.append(self.energy.mode(*self.domain, points % self.domain.dim))


	def get_distinct(self) -> List[int]:
		"""Gets the distinct configurations based on the average radii of the sites.
		and returns the number of configurations for each distinct configuration.
		"""

		distinct_avg_radii, distinct_count, new_frames = [], [], []

		for frame in self.frames:
			avg_radii = np.sort(frame.stats["avg_radius"])
			is_in = False
			for i, dist_radii in enumerate(distinct_avg_radii):
				if np.allclose(avg_radii, dist_radii, atol=1e-5):
					is_in = True
					distinct_count[i] += 1
					break

			if not is_in:
				distinct_avg_radii.append(avg_radii)
				new_frames.append(frame)

		self.frames = new_frames
		return distinct_count


	def save(self, info: Dict) -> None:
		self.path.mkdir(exist_ok=True)
		path = self.path / 'data.squish'

		with open(path, 'ab') as out:
			pickle.dump(info, out, pickle.HIGHEST_PROTOCOL)


	def frame_data(self, index: int) -> None:
		f = self[index]
		info = {
			"arr": f.site_arr,
			"domain": (f.n, f.h, f.w, f.r),
			"energy": f.energy,
			"stats": f.stats
		}
		return info

		# all_info = []
		# for frame in self.frames:
		# 	frame_info = dict()
		# 	frame_info["arr"] = frame.site_arr
		# 	frame_info["energy"] = {AreaEnergy: "area", RadialALEnergy: "radial-al",
		# 							RadialTEnergy: "radial-t"}[self.energy]
		# 	frame_info["params"] = (frame.n, frame.w, frame.h, frame.r)
		# 	all_info.append(frame_info)

		# class_name = {Flow: "flow", Search: "search", Shrink: "shrink"}[self.__class__]

		# with open(path, 'wb') as output:
		# 	pickle.dump((all_info, class_name), output, pickle.HIGHEST_PROTOCOL)
		# print("Wrote to " + path, flush=True)


	@staticmethod
	def load(path: str) -> Tuple[Simulation, Generator]:
		def frames() -> Dict:
			with open(path, 'rb') as infile:
				while True:
					try:
						yield pickle.load(infile)
					except EOFError:
						break

		with open(path, 'rb') as infile:
			sim_info = pickle.load(infile)

			domain = DomainParams(*sim_info["domain"])
			energy = Energy(sim_info["energy"])
			sim = STR_TO_SIM[sim_info["mode"]](domain, energy, *list(sim_info.values())[3:])

			return sim, frames()


	@staticmethod
	def load_old(filename: str) -> Simulation:
		"""
		Loads the points at every point into a file.
		:param filename: [str] name of the file
		"""
		frames = []
		with open(filename, 'rb') as data:
			all_info, sim_class = pickle.load(data)
			if type(sim_class) == str:
				sim_class = {"flow": Flow, "search": Search, "shrink": Shrink}[sim_class]


			sim = sim_class(*all_info[0]["params"], "radial-t", 0,0)
			for frame_info in all_info:
				frames.append(sim.energy(*frame_info["params"], frame_info["arr"]))
				#frames[-1].stats = frame_info["stats"]

			sim.frames = frames
		return sim


class Flow(Simulation):
	"""Finds an equilibrium from initial sites.

	Attributes:
		domain (DomainParams): domain parameters for this simulation.
		energy (Energy): energy being used for caluclations.
		path (Path): path to the location of where to store simulation files.
		frames (List[VoronoiContainer]): stores frames of the simulation.
		step_size (float): size fo step by for each iteration.
		thres (float): threshold for the stopping condition.
		accel (bool): set to True if accelerated stepping is desired.

	"""

	__slots__ = ['step_size', 'thres', 'accel']
	attr_str = "flow"
	title_str = "Flow"

	def __init__(self, domain: DomainParams, energy: Energy, step_size: float, thres: float,
					accel: bool, name: Optional[str] = None) -> None:
		super().__init__(domain, energy, name=name)
		self.step_size, self.thres, self.accel = step_size, thres, accel


	@property
	def initial_data(self) -> Dict:
		info = {
			"mode": self.attr_str,
			"domain": (self.domain.n, self.domain.w, self.domain.h, self.domain.r),
			"energy": self.energy.attr_str,
			"step_size": self.step_size,
			"thres": self.thres,
			"accel": self.accel
		}
		return info


	def run(self, save: bool, log: bool, log_steps: int) -> None:
		if log: print(f"Find - {self.domain}", flush=True)
		if save: self.save(self.initial_data)
		if len(self) == 0: self.add_frame()

		i, grad_norm = 0, float('inf')

		trial = 2
		while grad_norm > self.thres:	# Get to threshold.
			if save: self.save(self.frame_data(i))

			# Iterate and generate next frame using RK-2
			start = timer()
			change, grad = self[i].iterate(self.step_size)
			new_frame = self.energy.mode(*self.domain, self[i].add_sites(change))
			grad_norm = np.linalg.norm(grad)
			end = timer()

			if self.accel:
				if new_frame.energy < self[i].energy:	# If energy decreases.
					if trial < 10:	# Try increasing step size for 10 times.
						factor = 1 + .1**trial

						test_frame = self.energy.mode(*self.domain,
												self[i].add_sites(change*factor))
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
					new_frame = self.energy.mode(*self.domain,
									self[i].add_sites(change/shrink_factor))
					self.step_size /= shrink_factor

				self.step_size = max(10e-4, self.step_size)

			self.frames.append(new_frame)

			i += 1
			if(log and i % log_steps == 0):
				print(f'Iteration: {i:05} | Energy: {self[i].energy: .5f}' + \
				 	  f' | Gradient: {grad_norm:.8f} | Step: {self.step_size: .5f} | ' + \
				 	  f'Time: {end-start: .3f}', flush=True)



class Search(Simulation):
	"""Searches for a given number of equilibria.

	Attributes:
		domain (DomainParams): domain parameters for this simulation.
		energy (Energy): energy being used for caluclations.
		path (Path): path to the location of where to store simulation files.
		frames (List[VoronoiContainer]): stores frames of the simulation.
		step_size (float): size fo step by for each iteration.
		thres (float): threshold for the stopping condition.
		accel (bool): set to True if accelerated stepping is desired.
		kernel_step (float): size to step on manifold if nullity of hessian > 2.
		count (int): number of equilibria to find.

	"""

	__slots__ = ['step_size', 'thres', 'accel', 'kernel_step', 'count']
	attr_str = "search"
	title_str = "Search"

	def __init__(self, domain: DomainParams, energy: Energy, step_size: float, thres: float,
					accel: bool, kernel_step: float, count: int,
					name: Optional[str] = None) -> None:
		super().__init__(domain, energy, name=name)
		self.step_size, self.thres, self.accel = step_size, thres, accel
		self.kernel_step, self.count = kernel_step, count


	@property
	def initial_data(self) -> Dict:
		info = {
			"mode": self.attr_str,
			"domain": (self.domain.n, self.domain.w, self.domain.h, self.domain.r),
			"energy": self.energy.attr_str,
			"step_size": self.step_size,
			"thres": self.thres,
			"accel": self.accel,
			"kernel_step": self.kernel_step,
			"count": self.count
		}
		return info


	def run(self, save: bool, log: bool, log_steps: int) -> None:
		if log: print(f'Travel - {self.domain}', flush=True)
		if save: self.save(self.initial_data)

		if len(self) != 0:
			new_sites = self[0].site_arr
			self.frames = []
		else:
			new_sites = None

		for i in range(self.count):
			# Get to equilibrium.
			sim = Flow(self.domain, self.energy, self.step_size, self.thres, self.accel)
			sim.add_frame(new_sites)
			sim.run(False, log, log_steps)

			self.frames.append(sim[-1])
			if save: self.save(self.frame_data(i))
			if log: print(f'Equilibrium: {i:04}\n', flush=True)

			# Get Hessian,and check nullity. If > 2, perturb.
			hess = self.frames[i].hessian(10e-5)
			eigs = np.sort(np.linalg.eig(hess)[0])
			self.frames[i].stats["eigs"] = eigs

			zero_eigs = np.count_nonzero(np.isclose(eigs, np.zeros((len(eigs),)), atol=1e-4))

			if zero_eigs == 2:
				new_sites = None
			else:
				print("Warning: Nullity > 2. Expected if AreaEnergy.", flush=True)
				ns = null_space(hess, 10e-4).T
				vec = ns[random.randint(0, len(ns)-1)].reshape((self.domain.n, 2))	# Random vector.
				new_sites = self.frames[i].add_sites(self.kernel_step*vec)



class Shrink(Simulation):
	"""Shrinks width and finds nearest equilibrium.

	Attributes:
		domain (DomainParams): domain parameters for this simulation.
		energy (Energy): energy being used for caluclations.
		path (Path): path to the location of where to store simulation files.
		frames (List[VoronoiContainer]): stores frames of the simulation.
		step_size (float): size fo step by for each iteration.
		thres (float): threshold for the stopping condition.
		accel (bool): set to True if accelerated stepping is desired.
		delta (float): percent to change w each iteration.
		stop_width (float): percent at which to stop iterating.

	"""

	__slots__ = ['step_size', 'thres', 'accel', 'delta', 'stop_width']
	attr_str = "shrink"
	title_str = "Shrink"


	def __init__(self, domain: DomainParams, energy: Energy, step_size: float, thres: float,
					accel: bool, delta: float, stop_width: float,
					name: Optional[str] = None) -> None:
		super().__init__(domain, energy, name=name)
		self.step_size, self.thres, self.accel = step_size, thres, accel
		self.delta, self.stop_width = self.domain.w*delta/100, self.domain.w*stop_width


	@property
	def initial_data(self) -> Dict:
		info = {
			"mode": self.attr_str,
			"domain": (self.domain.n, self.domain.w, self.domain.h, self.domain.r),
			"energy": self.energy.attr_str,
			"step_size": self.step_size,
			"thres": self.thres,
			"accel": self.accel,
			"delta": self.delta,
			"stop_width": self.stop_width
		}
		return info


	def run(self, save: bool, log: bool, log_steps: int) -> None:
		if log: print(f'Shrink - {self.domain}', flush=True)
		if save: self.save(self.initial_data)

		if len(self) != 0:
			new_sites = self[0].site_arr
			self.frames = []
		else:
			new_sites = None

		width = self.domain.w
		i = 0
		while width >= self.stop_width:
			# Get to equilibrium.
			new_domain = DomainParams(self.domain.n, width, self.domain.h, self.domain.r)
			sim = Flow(new_domain, self.energy, self.step_size, self.thres, self.accel)
			sim.add_frame(new_sites)
			sim.run(False, log, log_steps)
			new_sites = sim[-1].site_arr

			self.frames.append(sim[-1])
			if save: self.save(self.frame_data(i))

			if log: print(f'Width: {width:.4f}\n')

			width -= self.delta
			i += 1

STR_TO_SIM = {
	"flow": Flow,
	"search": Search,
	"shrink": Shrink
}