#!/usr/bin/env python3

from pathlib import Path
import sys, numpy as np

def main():
	n = int(sys.argv[1])
	all_widths = set(np.round(np.arange(3, 10.05, 0.05), 2))
	for file in Path(f"simulations/Radial[T]T - N{n}R4.0").iterdir():
		i = file.name.index("x")
		all_widths.remove(float(file.name[i-4:i]))

	remain_widths = sorted(list(all_widths))[::-1]
	print(remain_widths)
	print([int(round((10-w)/.05)) for w in remain_widths])


if __name__ == "__main__":
	main()