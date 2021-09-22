from __future__ import annotations
from typing import Optional

import pickle, numpy as np
from scipy.linalg import null_space
from timeit import default_timer as timer

from .common import DomainParams, Energy, Simulation


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


	def save_initial(self) -> None:
		info = {
			"mode": self.attr_str,
			"step_size": self.step_size,
			"thres": self.thres,
			"accel": self.accel
		}

		with open(self.path, 'wb') as out:
			pickle.dump(info, out, pickle.HIGHEST_PROTOCOL)
		print("Created simulation file at:", self.path, flush=True)


	def run(self, save: bool, log: bool, log_steps: int) -> None:
		if log: print(f"Find - {self.domain}", flush=True)
		if save: self.save_initial()
		if len(self) == 0: self.add_frame()

		i, grad_norm = 0, float('inf')

		trial = 2
		while grad_norm > self.thres:	# Get to threshold.
			if save: self.save_frame(i)

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


	def save_initial(self) -> None:
		info = {
			"mode": self.attr_str,
			"step_size": self.step_size,
			"thres": self.thres,
			"accel": self.accel,
			"kernel_step": self.kernel_step,
			"count": self.count
		}

		with open(self.path, 'wb') as out:
			pickle.dump(info, out, pickle.HIGHEST_PROTOCOL)
		print("Created simulation file at:", self.path, flush=True)


	def run(self, save: bool, log: bool, log_steps: int) -> None:
		if log: print(f'Travel - {self.domain}', flush=True)
		if save: self.save_initial()

		if len(self) != 0:
			new_sites = self[0].site_arr
			self.frames = []
		else:
			new_sites = None

		for i in range(self.count):
			# Get to equilibrium.
			sim = Flow(self.domain, self.energy, self.thres, self.step_size, self.accel)
			sim.add_frame(new_sites)
			sim.run(False, log, log_steps)

			self.frames.append(sim[-1])
			if save: self.save_frame(i)
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
		self.delta, self.stop_width = self.domain.w*delta, self.domain.w*stop_width


	def save_initial(self) -> None:
		info = {
			"mode": self.attr_str,
			"step_size": self.step_size,
			"thres": self.thres,
			"accel": self.accel,
			"kernel_step": self.kernel_step,
			"count": self.count
		}

		with open(self.path, 'wb') as out:
			pickle.dump(info, out, pickle.HIGHEST_PROTOCOL)
		print("Created simulation file at:", self.path, flush=True)


	def run(self, save: bool, log: bool, log_steps: int) -> None:
		if log: print(f'Shrink - {self.domain}', flush=True)
		if save: self.save_initial()

		if len(self) != 0:
			new_sites = self[0].site_arr
			self.frames = []
		else:
			new_sites = None

		width = self.domain.w
		while width >= self.stop_width:
			# Get to equilibrium.
			sim = Flow(self.domain, self.energy, self.thres, self.step_size, self.accel)
			sim.add_frame(new_sites)
			sim.run(False, log, log_steps)
			new_sites = sim[-1].site_arr

			self.frames.append(sim[-1])
			if save: self.save_frame(i)

			if log: print(f'Width: {self.w:.4f}\n')

			width -= self.delta
