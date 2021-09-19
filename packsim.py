from __future__ import annotations
from typing import Dict
import argparse, json, numpy as np, os
from shutil import which
from simulation import Diagram, Flow, Search, Shrink

from packsim_core import RadialTEnergy

dia_presets = {
	"animate": [["voronoi"]],
	"energy": [["voronoi", "energy"]],
	"stats": [
		["voronoi", "eigs", "site_edge_count"],
		["site_isos", "site_energies", "edge_lengths"]
	],
	"eigs": [["voronoi", "eigs"]]
}

def check_params(container: Dict, needed: List[str], valid: Dict):
	"""
	Checks container for the necessary items, and raises
	an error if the parameter is not found.
	:param container: [Dict] contains the submitted parameters.
	:param needed: [List[str]] contains the needed parameters.
	:param valid: [Dict] if there are specific valid parameters,
	will also check for those.
	"""
	for need in needed:
		if need not in container:
			raise ValueError(f"Parameter \'{need}\' is required.")

		if need in valid:
			if type(valid[need]) is list:
				if container[need] not in valid[need]:
					raise ValueError(f"Parameter \'{need}\' must be one of these values: " + \
										 f"{str(valid[need])[1:-1]}.")
			elif valid[need] == "positive":
				if container[need] < 0:
					raise ValueError(f"Parameter \'{need}\' must be positive.")


def main():
	# Loading configuration and settings.
	parser = argparse.ArgumentParser("PackSim")
	parser.add_argument('sim_conf', metavar='/path/to/config.json',
						help="configuration file for a simulation")
	parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
						help="suppress all normal output")
	parser.add_argument('-l', '--log', dest='log_steps', default=50, type=int,
						help="number of iterations before logging")

	parser.add_argument('--n_objects', dest='n', type=int, help="objects in domain")
	parser.add_argument('--width', dest='w', type=float, help="width of domain")
	parser.add_argument('--height', dest='h', type=float, help="height of domain")
	parser.add_argument('--natural_radius', dest='r', type=float, help="natural radius of object")
	parser.add_argument('--energy', dest='energy', help="energy type of system")

	args = parser.parse_args()
	config_sim(args)
	# if args.input_file is None:
	# 	config_sim(args)
	# else:
	# 	loaded_sim(args)


def config_sim(args):
	with open(args.sim_conf) as f:
		params = json.load(f)

	check_params(params, ["domain", "simulation"], {})
	dmn_params, sim_params = params["domain"], params["simulation"]

	overrides = {args.n: "n_objects", args.w: "width", args.h: "height", args.r: "natural_radius",
				 args.energy: "energy"}
	for arg, arg_name in overrides.items():
		if arg is not None:
			dmn_params[arg_name] = arg

	check_params(dmn_params, ["n_objects", "width", "height", "natural_radius", "energy"], {
		"n_objects": "positive", "width": "positive", "height": "positive",
		"natural_radius": "positive", "energy": ["area", "radial-al", "radial-t"]
	})
	n, w, h, r, energy = dmn_params["n_objects"], dmn_params["width"], dmn_params["height"], \
				dmn_params["natural_radius"], dmn_params["energy"]



	points = None
	if "points" in dmn_params:
		points = np.asarray(dmn_params["points"])

	check_params(sim_params, ["mode", "step_size", "threshold", "save_sim"], {
		"mode": ["flow", "search", "shrink"], "step_size": "positive", "threshold": "positive"
	})
	mode, step, thres, save_sim = sim_params["mode"], sim_params["step_size"], \
									sim_params["threshold"], sim_params["save_sim"]

	name = sim_params.get("name")

	if mode == "flow":
		sim = Flow(n, w, h, r, energy, thres, step)
	elif mode == "search":
		check_params(sim_params, ["manifold_step_size"], {"manifold_step_size": "positive"})
		sim = Search(n, w, h, r, energy, thres, step, sim_params["manifold_step"], 
						sim_params["count"])
	elif mode == "shrink":
		check_params(sim_params, ["width_change", "width_stop"], {
			"width_change": "positive", "width_stop": "positive"
		})
		sim = Shrink(n, w, h, r, energy, thres, step, sim_params["width_change"],
						sim_params["width_stop"])

	save_diagram = False
	if "diagram" in params:
		save_diagram = True
		dia_params = params["diagram"]
		check_params(dia_params, ["filetype", "figures"], {
			"filetype": ["img", "mp4"]
		})
		if dia_params["filetype"] == "mp4":
			if which("ffmpeg") is None:
				raise ValueError("The program 'ffmpeg' needs to be installed on your system.")

		if type(dia_params["figures"]) is str:
			dia_params["figures"] = np.asarray(dia_presets[dia_params["figures"]])
		else:
			dia_params["figures"] = np.asarray(dia_params["figures"])

	sim.initialize(points)
	sim.run(not args.quiet, args.log_steps)
	if save_sim: sim.save(filename=name)

	if save_diagram:
		diagram = Diagram(sim, dia_params["figures"])
		if dia_params["filetype"] == "img":
			diagram.render_static(0, filename=name)
		elif dia_params["filetype"] == "mp4":
			diagram.render_video(filename=name)


def loaded_sim(args):
	pass


if __name__ == '__main__':
	os.environ["QT_LOGGING_RULES"] = "*=false"
	try:
		main()
	except KeyboardInterrupt:
		print("Program terminated by user.")