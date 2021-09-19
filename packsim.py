#!/usr/bin/env python3

from __future__ import annotations
import argparse, json
from simulation import Diagram, Flow, Search, Shrink


def get_diagram(sim, t):
	if t == "flow":
		diagram = Diagram(sim, np.array([["voronoi", "energy"]]))
	elif t == "stats":
		diagram = Diagram(sim, np.array([
							["voronoi", "eigs", "site_edge_count"],
							["site_isos", "site_energies", "edge_lengths"]						
				]), cumulative=False)
	elif t == "eigs":
		diagram = Diagram(sim, np.array([["voronoi", "eigs"]]))
	elif t == "shrink":
		diagram = Diagram(sim, np.array([["voronoi", "avg_radius", "isoparam_avg"]]), cumulative=False)

	return diagram


def main():
	# Loading configuration and settings.
	parser = argparse.ArgumentParser("Processes packing simulations.")
	parser.add_argument('sim_conf', metavar='path/to/config',
						help="configuration file for a simulation")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")
	parser.add_argument('-l', '--log', dest='log_steps', action='store_true', default=50,
						help="number of iterations before logging")
	parser.add_argument('-i', '--input', dest='input_file')
	parser.add_argument('-o', '--output', dest='output_file')

	args = parser.parse_args()

	if args.input_file is None:
		config_sim(args)
	else:
		loaded_sim(args)


def config_sim(args):
	with open(args.sim_conf) as f:
		params = json.load(f)

	calc_params, sim_params = params["calc"], params["sim"]
	n, w, h, r, energy = calc_params["n_objects"], calc_params["width"], calc_params["height"], \
				calc_params["natural_radius"], calc_params["energy"]

	mode, thres, step = sim_params["mode"], sim_params["threshold"], sim_params["step_size"]

	# Running simulation
	if mode == "flow":
		sim = Flow(n, w, h, r, energy, thres, step)
	elif mode == "search":
		sim = Search(n, w, h, r, energy, thres, step, sim_params["manifold_step"], 
						sim_params["count"])
	elif mode == "shrink":
		sim = Shrink(n, w, h, r, energy, thres, step, sim_params["delta_width"], 
						sim_params["stop_width"])

	sim.initialize()
	sim.run(not args.quiet, args.log_steps)


def loaded_sim(args):
	pass


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")