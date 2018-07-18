import json
import sys

with open(sys.argv[1]) as f:
	j = json.load(f)

def make_wkt(pts):
	return "LINESTRING (" + ", ".join(["{} {}".format(x['lat'], x['lon']) for x in pts]) + ")"

def make_geojson(pts):
	return {
		"type": "Feature",
		"properties": {},
		"geometry": {
			"type": "LineString",
			"coordinates": [[x['lon'], x['lat']] for x in pts],
		},
	}

if False:
	print("time,geometry")
	first = None
	pts = []
	for i in range(len(j)):
		if first is None:
			first = j[0]['time']
		if i != 0 and (i % 20 == 0 or i == len(j) - 1):
			print("{},\"{}\"".format(first, make_wkt(pts)))
			pts = []
			first = None
		pts.append(j[i])

if False:
	print("time,speed,lat,lon")
	for p in j:
		print("{},{},{},{}".format(p['time'], p['speed'], p['lat'], p['lon']))

if True:
	top = {
		"type": "FeatureCollection",
		"features": [],
	}
	first = None
	pts = []
	for i in range(len(j)):
		if first is None:
			first = j[0]['time']
		if i != 0 and (i % 20 == 0 or i == len(j) - 1):
			top['features'].append(make_geojson(pts))
			#print("{},\"{}\"".format(first, make_wkt(pts)))
			pts = pts[len(pts)-1:]
			first = None
		pts.append(j[i])
	print(json.dumps(top, indent=4))
