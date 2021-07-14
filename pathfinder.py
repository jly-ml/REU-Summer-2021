
import numpy as np
import random
import math

def pathfinder(source, dest,G,maxServer,maxV,k):

    # rewards matrix
    R = np.matrix(np.zeros(shape=(maxV, maxV)))
    for x in G[dest]:
        R[x, dest] = 100

    # Q matrix
    Q = np.matrix(np.zeros(shape=(maxV, maxV)))
    Q -= 100
    for node in G.nodes:
        for x in G[node]:
            Q[node, x] = 0
            Q[x, node] = 0


    def next_number(start, er):
        random_value = random.uniform(0, 1)
        if random_value < er:
            if ((start > maxServer)):
                sample = G[start]
            else:
                sample = np.where(Q[start,] == np.max(Q[start,]))[1]
        else:
            sample = np.where(Q[start,] == np.max(Q[start,]))[1]
        next_node = int(np.random.choice(sample, 1))
        return next_node


    def updateQ(n1, n2, lr, discount):
        max_index = np.where(Q[n2,] == np.max(Q[n2,]))[1]
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)
        max_value = Q[n2, max_index]

        Q[n1, n2] = int((1 - lr) * Q[n1, n2] + lr * (R[n1, n2] + discount * max_value))


    walk = 100 * (pow(6,int(math.log(k))*2) )  # as k increases, the walks needs to increase as well.. especially k = 32
    #print(walk)

    def learn(er, lr, discount):
        for i in range(int(walk)):
            start = np.random.randint(0, maxV)
            next_node = next_number(start, er)
            updateQ(start, next_node, lr, discount)


    def sp(source, dest):
        path = [source]
        next_node = np.argmax(Q[source,])
        path.append(next_node)
        while next_node != dest:
            next_node = np.argmax(Q[next_node,])
            path.append(next_node)
        return path

    # begin the walk
    learn(0.4, 0.8, 0.7)

    final_path = sp(source, dest)
    hops = len(final_path)
    local_cost = hops
    print('From source vm in PM: [', source, '] to [', dest, '] destination PM takes', hops , 'hops!')
    print('Final path: ', final_path)
   # print('local cost is:', local_cost)

    return local_cost