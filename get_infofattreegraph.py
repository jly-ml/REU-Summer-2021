import math
import networkx as nx
import numpy as np

def info_fattree_graph(k , maxServer) :
	print('CHECK: Max amount of servers given k = ', k, ' is ', maxServer, 'servers')

	# initializing
	coreCT = int(math.pow((k / 2), 2))
	interswitch = int(k / 2)
	servers = int(math.pow((k / 2), 2))
	edgeS = maxServer
	aggS = maxServer + (2 * servers) - 1
	coreS = aggS + (2 * servers) - int(interswitch / 2) - 1
	if (coreS % 2 != 0):
		coreS = coreS + 1
	icoreS = coreS
	#print(coreS)
	print('INFO CHECK')
	print('Number of Core Switches: ', coreCT, ' and number of inter switches(aggregate and edge switches) : ', interswitch)

	# initializing lists to add edges to
	ft = list()
	results = []

	serverID = 0
	switchCT = 0
	aggCT = 0
	Cct = 1
	v = 2 * (int(math.log(k,2))-1)

	# graph to use later for the search
	graph = list()
	switchlist = list()
	l1 = list()  # edge between PM_ID => EDGE SWITCH
	l2 = list()  # edge between EDGE SWITCH => AGG SWITCH
	l3 = list()  # edge between AGG SWITCH => CORE SWITCH

	print(aggS, coreS, icoreS, v)
	for pod in range(0, (k)):
		for a in range(0, (servers)):
			ft.append(serverID)
			l1.append(serverID)
			serverID = serverID + 1
			switchCT = switchCT + 1
			aggCT = aggCT + 1

			if (switchCT <= interswitch):
				ft.append(edgeS)
				l1.append(edgeS)
				aggS = aggS + 1
				ft.append(aggS)
				l2 = [edgeS, aggS]
				coreS = coreS + v
			else:
				edgeS = edgeS + 1
				ft.append(edgeS)
				l1.append(edgeS)
				switchCT = 1
				Cct = Cct + 1
				coreS = icoreS + v + Cct - 1

				if (aggCT >= servers):
					aggS = aggS + 1
					ft.append(aggS)
					aggCT = 1
				else:
					aggS = aggS - (interswitch - 1)
					ft.append(aggS)

			l2 = [edgeS, aggS]
			ft.append(coreS)
			l3 = [aggS, coreS]
			graph.append(tuple(l1))
			graph.append(tuple(l2))
			graph.append(tuple(l3))
			results.append(ft)
			switchlist.append(l2)
			switchlist.append(l3)
			print(ft)
			ft, l1, l2, l3 = ([] for i in range(4))
		Cct = 0

	# calculate upper boundary
	maxV = np.max(results) + 1
	# print(maxV)


	# convert list graph into an actual graph
	G = nx.Graph()
	G.add_edges_from(graph)
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos)
	nx.draw_networkx_edges(G, pos)
	nx.draw_networkx_labels(G, pos)
	#plt.show()

	switchlist = [y for x in switchlist for y in x]
	switchlist = list(set(switchlist))
	#print(switchlist)
	return G, maxV, switchlist