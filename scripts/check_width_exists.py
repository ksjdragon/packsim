from pathlib import Path
import numpy as np, pickle, sys

from squish import Simulation

def main():
	n = int(sys.argv[1])
	all_widths = set(np.round(np.arange(3, 10.05, 0.05), 2))
	for file in Path(f"squish_output/Radial[T]Search - N{n} - 500").iterdir():
		sim, frames = Simulation.load(file / 'data.squish')

		if sim.domain.n == n:
			try:
				all_widths.remove(next(frames)["domain"][1])
			except StopIteration:
				pass

	remain_widths = sorted(list(all_widths))[::-1]
	print("Remaining:", remain_widths)


if __name__ == "__main__":
	main()