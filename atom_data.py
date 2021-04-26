import sys

def data(path):
	path='/'.join(path.split('/')[:-1])+'/data.json'
	if path=='/data.json':
		path='data.json'
	f=open(path,'r')
	lines=f.readlines()
	f.close()

	d=eval(''.join(lines))
	fd={}
	for i in range (len(d)):
		fd[i+1]=d[i]
	return fd

def symbol_dict(path):
	fd=data(path)
	symbol={}
	for i in range (len(fd)):
		num=i+1
		symbol[fd[num]['symbol']]=num
	return symbol
