import pandas as pd
from get_infofattreegraph import *
from pathfinder import *
from cost import *
#converts fattree array into a graph representation with its corresponding weights

RESOURCE_CAP = 2


"""creates a subgraph of individual VM to calculate its total cost of communicating with each PM, recreate FIG 6. Manually run the MCF algorithm on the same environment to compare with the agent's performance at the end of training """

confirmation = 'Y'
key = ['source', 'dest']

def node_comm_cost(NUM_VMP, PM_S, PM_D, vm_count, mb_count, PM2PM ,TR, PM2MB,miu,VM_PM):
    nodecost = 0
    vm = 0
    for x in range(0,NUM_VMP):
      nodecost = miu*(PM2PM[PM_S, PM_D]) + TR[int(vm_count/2)]*(PM2MB[mb_count,PM_D])
      nodecost2 = miu*(PM2PM[PM_S, PM_D]) + TR[int(vm_count/2)]*(PM2MB[mb_count+1,PM_D])

      #print(vm_count-1, vm_count, PM_D, nodecost)
      VM_PM[vm_count- 1][PM_D] = nodecost
      VM_PM[vm_count][PM_D] = nodecost2
      nodecost = nodecost + nodecost2
      vm = vm_count + 2
    return  nodecost

while (confirmation == 'Y'):
    masterlist = []
    ORIGINAL_COMM_COST = 0
    k = int(input("Enter a k: "))
    maxServer = int(math.pow(k, 3) / 4)


    vm_pair = int(input('Enter in how many VM pairs you would like: '))
    while 2*vm_pair > (RESOURCE_CAP*maxServer):
        vm_pair = int(input('Number of pairs is more than what each Physical Machine can handle, pick a smaller amount of VM pairs... '))

    miu = int(input("Enter migration coefficient: "))
    n = int(input('Enter in amount of MBs: '))

    VM_PM = 0 * np.ndarray(shape=((2*vm_pair), maxServer))
    traffic_rate =  np.random.randint(1,vm_pair, size=vm_pair)
    pm2pm_table = 0*np.ndarray(shape = (maxServer, maxServer))
    pm2mb_table = 0*np.ndarray(shape = (n, maxServer))
    vm2pm_table = 0*np.ndarray(shape = (RESOURCE_CAP, maxServer))


    pm_id = [*range(0,maxServer, 1)]
    source = []
    dest = []


    #randomly assign vm pairs to their housing PMs,verify that selected PM is within its resource capacity
    vm2pm_table = 0 * np.ndarray(shape=(int(RESOURCE_CAP / 2), maxServer), dtype=int)
    PM = np.arange(0, maxServer, 1)

    if vm_pair == maxServer:
        vm2pm_table.fill(1)
    else:
        #first create a flattened array of our vm2pm table then assign '1' vm_pair amount and shuffle the array to 'randomly assign' PMs their VM pair so long as it doesn't violate their RC
        vm = np.ones(vm_pair, dtype=int)
        VM2PM_flat = vm2pm_table.flatten()
        c = np.zeros((len(VM2PM_flat) - len(vm)), dtype=int)
        buffer = np.append(vm, c)
        VM2PM_flat = VM2PM_flat + buffer
        np.random.shuffle(VM2PM_flat)
        vm2pm_table = (np.reshape(VM2PM_flat, (int(RESOURCE_CAP / 2), len(PM))))
    # 1d array that shows how many VM pairs are stored in each PM, PMid = index value
    num_vmp = np.count_nonzero(vm2pm_table == 1, axis=0)

    for a in range (0, vm_pair):
        pair = random.sample(pm_id, 2)
        source.append(pair[0])
        dest.append(pair[1])


    graph, maxV, masterswitch_list = info_fattree_graph(k,maxServer)

    pathchart = list ()
    sourcelist = list()

    for pm_source in range(0, maxServer):
        for pm_dest in range(0, maxServer):
            if pm_dest == pm_source:
                pm2pm_table[pm_source, pm_dest] = 0
            else:
                pm2pm_table[pm_source, pm_dest] = pathfinder(pm_source, pm_dest, graph, maxServer, maxV, k)

    mb_1 = maxServer
    n_counter = 0

    for mb in range(mb_1, mb_1 + n):
        for pm_source in range(0, maxServer):
            pm2mb_table[n_counter, pm_source] = pathfinder(pm_source, mb, graph, maxServer, maxV,k)
        n_counter = n_counter + 1

    total_comm_cost = 0
    vmp_count = 1
    mb_count = 0
    for x in range(0,len(num_vmp)):
        if num_vmp [x] != 0:
            total_comm_cost = node_comm_cost(num_vmp[x],x,x,vmp_count,mb_count,pm2pm_table, traffic_rate, pm2mb_table,miu, VM_PM)
            total_comm_cost =  total_comm_cost +  total_comm_cost
            vmp_count = vmp_count+ 2*num_vmp [x]
    """
    switch_ei = [[] for i in range(n)]
    switch_ct = 0
    for y in range(0,maxServer):
       a = np.asarray(np.nonzero(VM_PM[:y]))
       print(VM_PM[:y], 'AND', a)
       for x in range(0, len(a[0])):
         if switch_ct != n:
             print(switch_ct, y, a[0][x-1])
             print('VM',VM_PM[y,a[0][x]])
             switch_ei[switch_ct].append(VM_PM[y,a[0][x]])
             switch_ct = switch_ct + 1
         else:
             switch_ct = 0
    interswitchcost = []
    for l in switch_ei:
        print('l',l)
        interswitchcost.append(min(l))
    VM_PM_IEGRESS =  sum(interswitchcost)
    """
    print('ORIGNAL COMMUNICATION COST IS: ', (total_comm_cost))

    """calculates the entire path table for all vms to all pms"""

    COMPLETE_VMPM =np.copy(VM_PM)
    for x in range(0, 2*vm_pair):
        for y in range(0,maxServer):
            if COMPLETE_VMPM[x][y] == 0:
                COMPLETE_VMPM[x][y] = miu*(pm2pm_table[np.asarray(np.nonzero(VM_PM[x:,]))[0][0],y]) + traffic_rate[int(x/2)]

    """ calculates the MCF of the given network, used for results comparision with the original communication cost and with the result of our q learning agent """
    key = ['source', 'dest']
    G = nx.DiGraph()
    G.add_node(key[0], demand=-(2*vm_pair))  # source
    G.add_node(key[1], demand=(2*vm_pair))  # sink/destination

    for x in range(0,(2*vm_pair)):
        G.add_edge(key[0], str(x), weight=0, capacity=1)

    for x in range(0, 2*vm_pair):
        for y in range(0,maxServer ):
            G.add_edge( str(x),       'PM_'+str(y), weight=int(COMPLETE_VMPM[x][y]), capacity=1)

    for y in range(0,maxServer):
        G.add_edge('PM_' + str(y), key[1], weight=0, capacity=int(vm_pair))

   # print('TOTAL COMMUNICATION COST:  mincost flow cost: ', nx.min_cost_flow_cost(G) + iegress_cost(G))
    print('TOTAL COMMUNICATION COST:  mincost flow cost: ', nx.min_cost_flow_cost(G) )

    #edges
    # nx.max_flow_min_cost(G, 'source', 'dest')

    confirmation = str(input('Start over? Y/N    ')).upper()


