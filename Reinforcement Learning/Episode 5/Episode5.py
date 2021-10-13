import copy

import numpy as np
import matplotlib.pyplot as plt

def main():
    the_neighborhood = np.loadtxt("the_neighborhood.txt")
    shape = the_neighborhood.shape
    shape = shape + (10, 4)
    Q_s_a = np.zeros(shape)
    Q_s_a = np.load("Q_s_a_new_place.npy")
    #Q_s_a = np.load("Q_s_a_2.npy")
    #Q_s_a = np.load("Q_s_a_eps_0.1.npy")
    Q_s_a = q_learning(Q_s_a, the_neighborhood)
    np.save("Q_s_a_new_place", Q_s_a)
    plt.imshow(the_neighborhood)
    plt.show()
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
        x_n, y_n, charge_n, charge_step, reward, a = do_action(x, y, charge, charge_step, eps_greedy, Q_s_a, map)
        #print("step")
        steps += 1
        if (reward != -1):
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


def q_learning(Q_s_a, map):
    for j in range(100000):
        print("iteration:", j+1)
        x = 2
        y = 2
        charge = 8
        charge_step = 5
        asd = True
        while(asd):
            x_n, y_n, charge_n, charge_step, reward, a = do_action(x, y, charge, charge_step, eps_greedy, Q_s_a, map)
            if x_n == 30 and y_n == 27:
                Q_s_a[x, y, charge, a] += (100/(100+j))* (reward - Q_s_a[x,y,charge, a])
                asd = False
            max_a = find_max_action(Q_s_a, charge_n, map, x_n, y_n)
            Q_s_a[x, y, charge, a] += (100/(100+j))*(reward+Q_s_a[x_n, y_n, charge_n, max_a]-Q_s_a[x,y,charge, a])
           # Q_s_a[x, y, charge, a] = reward+Q_s_a[x_n, y_n, charge_n, max_a]
            #Q_s_a[x, y, charge, a] += 0.1 *reward
            #Q_s_a[x, y, charge, a] += 0.1 *Q_s_a[x_n, y_n, charge_n, max_a]
            #Q_s_a[x, y, charge, a] -= 0.1 * Q_s_a[x, y, charge, a]
            x = x_n
            y = y_n
            #print(x, y, Q_s_a[x, y, charge, a])
            charge = charge_n
    return Q_s_a




def do_action(x,y, charge, charge_step, policy, Q_s_a, map):
    a = policy(Q_s_a, map, x, y, charge, 0)
    x_n, y_n = a_to_mov(a, x, y)
    charge_step -= 1
    if (charge_step == 0):
        charge -= 1
        charge_step = 5
    if charge == -1:
        return 0, 21, 8, 5, -100, a
    if x_n == 30 and y_n == 27:
        return x_n, y_n, 0, 0, 200, a
    if x_n == 0 and y_n == 21:
        return x_n, y_n, 8, 5, -100, a
    if x_n == 14 and y_n == 17:
        return x_n, y_n, 8, 5, -22+2*(charge+1), a
    if x_n == 23 and y_n == 11:
        return x_n, y_n, 8, 5, -18, a
    if x_n == 23 and y_n == 28:
        reward = -1 if charge+1 >= 7 else -10
        return x_n, y_n, 8, 5, reward, a
    if x_n == 28 and y_n == 1:
        return x_n, y_n, 8, 5, -15+(charge+1), a
    if x_n == 30 and y_n == 1:
        return x_n, y_n, 8, 5, -7, a
    return x_n, y_n, charge, charge_step, -1, a

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