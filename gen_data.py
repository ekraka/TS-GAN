import numpy as np
import os
import glob
import xyz_to_clmb as clb

mol_name = [ ]
rev_list = [ ]
for_list = [ ]
ts_list = [ ]
ma = 100


pr, re, ts, nam = [], [], [], []

for f in glob.glob("*.xyz"):
	if "rev" in f:
		current_split_list = f.split("_")
		current_num = str(current_split_list[1])
		mol_name = np.array(current_num)

		g,h = f.replace('rev', 'for'), f.replace('rev', '')
		k = os.listdir('.')
		if g in k and h in k:
			print ('Error in ', mol_name, 'Ignored')
			continue
		
		#print(mol_name)
		rev = clb.job(f)
		print (rev.shape)
		rev_list = np.array(rev)
		#print(rev_list)
		re.append(rev_list)
		#print(data)

		f = f.replace('rev', 'for')
		current_split_list = f.split("_")
		current_num = str(current_split_list[1])
		mol_name = np.array(current_num)
		#print(mol_name)
		forl = clb.job(f)
		for_list = np.array(forl)
		pr.append(for_list)

		f = f.replace('_for', '')

		current_split_list = f.split("_")
		current_num = str(current_split_list[1])
		mol_name = np.array(current_num)
		#print(mol_name)
		tsl = clb.job(f)
		ts_list = np.array(tsl)
		#print(ts_list)
		ts.append(ts_list)

		nam.append(mol_name)
			
a,b,c,d = [], [], [], []

for i in range (len(pr)):
	if pr[i].shape == (ma, ma) and re[i].shape == (ma, ma) and ts[i].shape == (ma, ma):
		a.append(pr[i])
		b.append(re[i])
		c.append(ts[i])
		d.append(nam[i])

data = {'products': np.array(a), 'reactants': np.array(b), 'ts': np.array(c), 'names': np.array(d)}

np.save('data.npy', data)



