from scipy.optimize import minimize
import numpy as np
import os
import sys

def get_dist(cords):

        def distance(a, b):
                return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)**0.5

        D = []
        for i in cords:
                D.append([])
                for j in cords:
                        D[-1].append(distance(i, j))

        return np.array(D)

def job(arr, coords, atoms = None):

	if atoms:

		g = open('temp_mov.xyz', 'w')

	def loss(x0):
		crd = x0.reshape((-1, 3))
		if atoms:
			g.write(str(len(crd)) + '\nComment\n')
			for j in range (len(atoms)):
				g.write(atoms[j] + ' ' + ' '.join(list(map(str, crd[j]))) + '\n')
		D = get_dist(crd)

		res = sum((D.flatten() - arr.flatten())**2)

		return res



	x0 = np.copy(coords).flatten()

	res = minimize(loss, x0).x

	g.close()

	res = res.reshape((-1, 3))

	#print (res)

	return res


