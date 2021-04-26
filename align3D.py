import sys
import numpy as np
import rmsd

def alin(A,B):
	A -= rmsd.centroid(A)
	B -= rmsd.centroid(B)
	#print (B)
	U = rmsd.kabsch(A, B)
	A = np.dot(A, U)

	return rmsd.rmsd(A,B)


# chege xyz to npy 
def xyz_np(f):
	g = open(f)
	lines= g.readlines()
	g.close()

	arr = []
	for line in lines[2:]:
		k = line.strip().split()
		if len(k) == 4:
			arr.append(list(map(float, k[1:])))
		else:
			break
	return np.array(arr)


if __name__=='__main__':
	b = xyz_np(sys.argv[1]) #first file  #np.array([[0,0,0], [1,1,1], [2,2,2]], dtype = 'float') # original TS/prod/reac
	a = xyz_np(sys.argv[2]) # second file  #np.array([[0,0,0], [-1,-1,-1], [-2,-2,-2]], dtype = 'float') # fake TS
	print (alin(a, b))





