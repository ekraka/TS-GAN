import os
import atom_data
import sys
import numpy as np 
import math


def distance(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def job(file):
	f = open(file, 'r')
	lines = f.readlines()
	f.close()

	atoms, cords = [], []
	a_d = atom_data.symbol_dict(sys.argv[0])
	for line in lines[2:]:
		k = line.strip().split()
		if len(k) < 2:
			continue
		a,x,y,z = k
		x,y,z = list(map(float, [x,y,z]))
		atoms.append(a_d[a])
		cords.append([x,y,z])

	pad = 50
	clmb_d = [] # columb matrix
	for i in range (pad):
		if len(atoms) <= i:
			clmb_d.append([0]*pad)
			continue

		clmb_d.append([])

		for j in range (pad):

			if len(atoms) <= j:
				clmb_d[-1].append(0)
				continue

			q1q2 = atoms[i]*atoms[j]
			if i == j:
				clmb_d[-1].append(q1q2**0.5)
				continue

			r = distance(cords[i], cords[j])
			

			clmb_d[-1].append(q1q2/r)

	return np.array(clmb_d)

if __name__ == '__main__':
	print (job(sys.argv[1]).shape)




