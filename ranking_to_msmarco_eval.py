import sys


f_out = open( sys.argv[1] + '_msmarco_eval' , 'w')
with open(sys.argv[1]) as f:
	count = 0
	prev = None
	for l in f:
		spl = l.split()
		if prev == None:
			prev = spl[0]
		if spl[0] != prev:
			prev = spl[0]
			count = 0
		
		
		count += 1
		f_out.write(f'{spl[0]}\t{spl[2]}\t{count}\n')
