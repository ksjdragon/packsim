from __future__ import annotations
from typing import List
from simulation import Diagram, Simulation
import argparse, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def main():
	nums = [13, 23, 57, 59, 83, 131, 179]

	for n in nums:
		Path(f"new_simulations/Radial[T]T - N{n}R4.0").mkdir(exist_ok=True)
		for file in Path(f"simulations/Radial[T]T - N{n}R4.0").iterdir():
			sim = Simulation.load(file)
			sim.get_distinct()
			sim.save(file.name[:-4].replace("x10.0", "x10.00"))
	#sim = Simulation.load("simulations/AreaT - N30R4 - 10x10.sim")


if __name__ == '__main__':
	main()