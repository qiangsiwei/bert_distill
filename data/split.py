import os, random

for name in ('clothing','fruit','hotel','pda','shampoo'):
	# os.mkdir(name)
	lines = open(os.path.join(name,name+'.txt')).read().strip().split('\n')
	random.shuffle(lines)
	with open(os.path.join(name,'train.txt'),'w') as out:
		out.write('\n'.join(lines[:int(0.1*len(lines))]))
	with open(os.path.join(name,'dev.txt'),'w') as out:
		out.write('\n'.join(lines[int(0.1*len(lines)):int(0.9*len(lines))]))
	with open(os.path.join(name,'test.txt'),'w') as out:
		out.write('\n'.join(lines[int(0.9*len(lines)):]))
