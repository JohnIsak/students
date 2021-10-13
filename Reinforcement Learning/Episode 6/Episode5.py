import copy

import numpy as np
import matplotlib.pyplot as plt

def main():
    the_neighborhood = np.loadtxt("the_neighborhood.txt")
    shape = the_neighborhood.shape
    shape = shape + (10, 4)
    #Q_s_a = np.zeros(shape)
    Q_s_a = np.load("Q_s_a_new_place.npy")
    #Q_s_a = np.load("Q_s_a_2.npy")
    #Q_s_a = np.load("Q_s_a_eps_0.1.npy")
    #Q_s_a = n_step_sarsa(Q_s_a, the_neighborhood)
    #np.save("Q_s_a_new_place", Q_s_a)
    #plt.imshow(the_neighborhood)
    #plt.show()
    map = copy.deepcopy(the_neighborhood)
    plot_route(Q_s_a, map)


def plot_route(Q_s_a, map):
    x = 2
    y = 2
    charge = 9
    charge_step = 5
    asd = True
    reward_tot = -200
    steps = 0
    while (asd):
        a = eps_greedy(Q_s_a, map, x, y, charge, 0)
        x_n, y_n, charge_n, charge_step, reward = do_action(x, y, charge, charge_step, a)
        #print("step")
        steps += 1
        if (reward != 0):
            reward_tot += reward
            print("new reward found at", steps)
        if x_n == 30 and y_n == 27:
            #Q_s_a[x, y, charge, a] = Q_s_a[x, y, charge, a] + 0.1 * (reward - Q_s_a[x, y, charge, a])
            asd = False
        #max_a = find_max_action(Q_s_a, charge_n, map, x_n, y_n)
        # Q_s_a[x, y, charge, a] += 0.1*(reward+Q_s_a[x_n, y_n, charge_n, max_a]-Q_s_a[x,y,charge, a])
        #Q_s_a[x, y, charge, a] += 0.1 * reward
        #Q_s_a[x, y, charge, a] += 0.1 * Q_s_a[x_n, y_n, charge_n, max_a]
        #Q_s_a[x, y, charge, a] -= 0.1 * Q_s_a[x, y, charge, a]
        x = x_n
        y = y_n
        map[x_n,y_n] = 2
        # print(x, y, Q_s_a[x, y, charge, a])
        charge = charge_n
    print("expected reward", reward_tot)
    print("steps", steps)
    plt.imshow(map)
    plt.show()


def eps_greedy(Q_s_a, the_neighborhood, x, y, c, eps):
    shape = the_neighborhood.shape
    if np.random.random() < eps:
        while(True):
            a = np.random.randint(0,4)
            x_n, y_n = a_to_mov(a, x, y)
            if (-1 < x_n < shape[0] and -1 < y_n < shape[1]) and the_neighborhood[x_n,y_n] >= 1:
                return a
    a = find_max_action(Q_s_a, c, the_neighborhood, x, y)
    return a


def find_max_action(Q_s_a, c, the_neighborhood, x, y):
    a = None
    max = np.NINF
    options = Q_s_a[x, y, c, :]
    for i in range(len(options)):
        x_n, y_n = a_to_mov(i, x, y)
        #print("Checking option", i, x_n, y_n)
        if -1 < x_n < the_neighborhood.shape[0] and -1 < y_n < the_neighborhood.shape[1]:
            if the_neighborhood[x_n,y_n] >= 1 and options[i] > max:
                a = i
                max = Q_s_a[x, y, c, a]
               # print("current max", max)
    return a

def n_step_sarsa(Q_s_a, map):
    #step size
    #alpha = (100/(100+j))
    n = 3
    state_memory = np.empty(n+1, dtype=tuple)
    action_memory = np.empty(n+1, dtype=int)
    reward_memory = np.empty(n+1)
    iterations = 10000
    sf = iterations/100
    for j in range(iterations):
        print("iteration:", j + 1)
        eps = 0.5 * (sf/(sf+j))
        x = 2
        y = 2
        charge = 9
        charge_step = 5
        state_memory[0] = (x,y,charge)
        a = eps_greedy(Q_s_a, map, x, y, charge, eps)
        action_memory[0] = a
        T = np.inf
        t = 0
        while True:
            if (t % 100000 == 0 and t != 0):
                print("still going")
            if t < T:
                x_n, y_n, charge_n, charge_step, reward = do_action(x, y, charge, charge_step, a)
                if (x == x_n and y == y_n):
                    print("They are the same")
                state_memory[(t+1)%(n+1)] = (x_n, y_n, charge_n)
                reward_memory[(t+1)%(n+1)] = reward
                if x_n == 30 and y_n == 27:
                    T = t+1
                else:
                    a_n = eps_greedy(Q_s_a, map, x_n, y_n, charge_n, eps)
                    action_memory[(t+1)%(n+1)] = a_n
            tau = t - n+1
            #print(tau)
            if tau >= 0:
                if T == np.inf:
                    G = np.sum(reward_memory)
                else:
                    find_G(tau, n, T, reward_memory)
                if tau+n < T:
                    x, y, charge = state_memory[(tau+n)%(n+1)]
                    a = action_memory[(tau+n)%(n+1)]
                    #print(x,y,charge,a)
                    G += Q_s_a[x,y,charge,a]
                x, y, charge = state_memory[tau%(n+1)]
                a = action_memory[tau%(n+1)]
                Q_s_a[x,y,charge,a] += (sf/(sf+j))*(G-Q_s_a[x,y,charge,a])
            if tau == T-1:
                break
            x = x_n
            y = y_n
            charge = charge_n
            a = a_n
            t = t+1

    return Q_s_a

def find_G(tau, n, T, reward_memory):
    G = 0
    i = (tau+1)%(n+1)
    while(True):
        if i == T%(n+1):
            break
        G = G + reward_memory[i]
        i = (i+1)%(n+1)
    return G


def do_action(x,y, charge, charge_step, a):
    x_n, y_n = a_to_mov(a, x, y)
    charge_step -= 1
    if (charge_step == 0):
        charge -= 1
        charge_step = 5
    if charge == -1:
        return x_n, y_n, 9, 5, -100
    if x_n == 30 and y_n == 27:
        return x_n, y_n, 0, 0, 200
    if x_n == 0 and y_n == 21:
        return x_n, y_n, 8, 5, -100
    if x_n == 14 and y_n == 17:
        return x_n, y_n, 8, 5, -22+2*(charge+1)
    if x_n == 23 and y_n == 11:
        return x_n, y_n, 8, 5, -18
    if x_n == 23 and y_n == 28:
        reward = -1 if charge+1 >= 7 else -10
        return x_n, y_n, 8, 5, reward
    if x_n == 28 and y_n == 1:
        return x_n, y_n, 8, 5, -15+(charge+1)
    if x_n == 30 and y_n == 1:
        return x_n, y_n, 8, 5, -7
    return x_n, y_n, charge, charge_step, 0

def a_to_mov(a,x,y):
    if a == 0:
        return x+1, y
    if a == 1:
        return x, y+1
    if a == 2:
        return x-1, y
    if a == 3:
        return x, y-1

main()