import json

with open('speed-stb.json') as f:
	j = json.loads(f.read())

csv = open('speed-stb.csv', 'w')

csv.write("time,speed\n")
for i in range(len(j)):
	csv.write("{},{}\n".format(j[i][0],j[i][1]))

csv.close