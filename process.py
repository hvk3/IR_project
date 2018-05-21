import sys

dic = {}
for i in range(3):
	dic[str(i)] = []

with open(sys.argv[1], 'r') as f:
	for line in f:
		id, tag = line.rstrip().split(',')
		dic[tag].append(id)

for i in range(3):
	dic[str(i)] = dic[str(i)][:80]

with open("selected.txt", 'w') as f:
	for i in range(3):
		for key in dic[str(i)]:
			f.write(key + "," + str(i) + "\n")
