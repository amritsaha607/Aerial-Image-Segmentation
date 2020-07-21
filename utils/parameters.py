# 0 : Background
# 1 : House
# 2 : Street

index2name = {
	0 : 'Background',
	1 : 'Building',
	2 : 'Street',
}
index2name_street = {
	0: 'Background',
	1: 'Street',
}
index2name_building = {
	0: 'Background',
	1: 'Building',
}

color2index = {
	(255, 255, 255) : 0,
	(0,     0, 255) : 1,
	(255,   0,   0) : 2,
	# (0,   255,   0) : 3,
	# (255, 255,   0) : 4,
	# (0,   255, 255) : 5,
}
color2index_street = {
	(255, 255, 255) : 0,
	(0,     0, 255) : 0,
	(255,   0,   0) : 1,
}
color2index_building = {
	(255, 255, 255) : 0,
	(0,     0, 255) : 1,
	(255,   0,   0) : 0,
}

colors = ['white', 'green', 'red']