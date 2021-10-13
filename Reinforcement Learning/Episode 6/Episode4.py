import numpy as np
import matplotlib.pyplot as plt


def show_lake(the_icy_lake):
    print(the_icy_lake.shape)
    plt.imshow(the_icy_lake)
    plt.show()


def move(x,y, v_x, v_y, d_v_x, d_v_y, the_lake):
    misfire = np.random.random() > 0.95
    if (misfire):
        return x,y,0,0
    if(np.square(v_x+d_v_x)+np.square(v_y+d_v_y) <= 16):
        v_x = v_x+d_v_x
        v_y = v_y+d_v_y
    ##Finish condition
    if (-1 < x+v_x < 75 and  y+v_y >= 150):
        return x+v_x, y+v_y, v_x, v_y
    if ((-1 < x+v_x < 75) and (-1 < y+v_y < 150)) and the_lake[x+v_x,y+v_y] == 1:
        return x+v_x, y+v_y, v_x, v_y
    return x, y, 0, 0

def do_run(the_lake, policy, epsilon):
    max_trial_length = 1000000
    while(True):
        visited_states = np.empty(max_trial_length, dtype=tuple)
        x = np.random.randint(0, 75)
        y = 0
        v_x = 0
        v_y = 0
        for i in range(max_trial_length):
            # d_v_x, d_v_y = policy(i)
            # a = velocities_a(d_v_x, d_v_y)
            a = e_soft(policy[x,y,v_x,v_y], epsilon)
            d_v_x, d_v_y = a_velocities(a)
            visited_states[i] = (x, y, v_x, v_y, a)
            x, y, v_x, v_y = move(x, y, v_x, v_y, d_v_x, d_v_y, the_lake)
            if y >= 149:
                return visited_states, i

    return visited_states,  max_trial_length-1


def e_soft(policy, epislon):
    if np.random.random() <= epislon:
        a = np.random.randint(0, 4)
        if (a == 4):
            print("A is 4 on np.random.randint()")
        return a
    a = np.argmax(policy)
    if (a == 4):
        print("A is 4 on np.argmax policy")
        print(policy.shape)
    return np.argmax(policy)


def policy(i):
    backwards = False
    if i >= 1000 and (0 <= i % 200 <= 50):
        backwards = True
    p = np.random.random()
    if p > 0.5:
        if backwards:
            return 0, -1
        return 0, 1
    if p > 0.25:
        return 1, 0
    return -1, 0

def monte_carlo_control(shape, the_icy_lake):
    Q_s_a = np.zeros(shape)
    optimal_policy = np.empty((shape[0], shape[1], shape[2], shape[3]))
    seen = np.zeros(shape)
    print(Q_s_a)
    for j in range(1000):
        print("Trial", j+1)
        visited_states, trial_length = do_run(the_icy_lake)
        for i in range(trial_length-1, -1, -1):
            x, y, v_x, v_y, a = visited_states[i]
            seen[x,y,v_x,v_y,a] += 1
            Q_s_a[x,y,v_x,v_y,a] =  Q_s_a[x,y,v_x,v_y,a] + (-trial_length+i)/seen[x,y,v_x,v_y,a]
            optimal_policy[x,y,v_x,v_y] = np.amax(Q_s_a[x,y,v_x,v_y,:])
    return optimal_policy, Q_s_a,

def monte_carlo_control_2(shape, the_icy_lake):
    Q_s_a = np.zeros(shape, dtype=float)
    #optimal_policy = np.empty((shape[0], shape[1], shape[2], shape[3]))
    seen = np.zeros(shape)
    epsilon = 0.8
    for j in range(100000):
        #print("Trial", j + 1)
        #print(Q_s_a)

        if (j % 10000 == 0 and j < 30000):
            epsilon -= 0.2
            print("Epsilon:",epsilon)
        if (j % 5000 == 0 and 30000 <= j < 60000):
            epsilon -= 0.03
            print("Epsilon:", epsilon)
        if (j % 5000 == 0  and j > 60000):
            epsilon -= 0.002
            print("Epsilon:", epsilon)
        if (epsilon <= 0):
            print(Warning)
            return
        visited_states, trial_length = do_run(the_icy_lake, Q_s_a, epsilon)
        for i in range(trial_length-1, -1, -1):
            x, y, v_x, v_y, a = visited_states[i]
            seen[x,y,v_x,v_y,a] += 1
            Q_s_a[x,y,v_x,v_y,a] =  Q_s_a[x,y,v_x,v_y,a] + ((-trial_length+i)-Q_s_a[x,y,v_x,v_y,a])/seen[x,y,v_x,v_y,a]
            #optimal_policy[x,y,v_x,v_y] = np.amax(Q_s_a[x,y,v_x,v_y,:])
       # Q_s_a = Q_s_a / np.abs(np.amin(Q_s_a))

    return Q_s_a

def velocities_a(x,y):
    if x == -1 and y == 0:
        return 1
    if x == 0 and y == 1:
        return 0
    if x == 1 and y == 0:
        return 2
    return 3


def a_velocities(n):
    if n == 0:
        return 0, 1
    if n == 1:
        return -1, 0
    if n == 2:
        return 1, 0
    #if n == 3:
    return 0, -1


def n_step_sarsa(Q_s_a, map):
    #step size
    #alpha = (100/(100+j))
    n = 10
    state_memory = np.empty(n+1, dtype=tuple)
    action_memory = np.empty(n+1, dtype=int)
    reward_memory = np.empty(n+1)
    iterations = 10000
    sf = iterations/100
    gamma = 1
    for j in range(iterations):
        if(j % 1000 == 0):
            print("iteration:", j + 1)
        eps = 0.5
        #eps = 0.5*(10/(10+j))
        x = np.random.randint(0, 75)
        #alpha = (10/(10+j))
        alpha = 0.5
        y = 0
        v_x = 0
        v_y = 0
        state_memory[0] = (x,y,v_x, v_y)
        a = e_soft(Q_s_a[x,y,v_x, v_y], eps)
        action_memory[0] = a
        T = np.inf
        t = 0
        while True:
            if (t % 100000 == 0 and t != 0):
                print("still going")
            if t < T:
                d_v_x, d_v_y = a_velocities(a)
                x_n, y_n, v_x_n, v_y_n = move(x, y, v_x, v_y, d_v_x, d_v_y, map)
                reward = -1
                if y_n >= 150:
                    y_n = 149
                    reward = 200
                    T = t+1
                else:
                    a_n = e_soft(Q_s_a[x_n, y_n, v_x_n, v_y_n], eps)
                    action_memory[(t+1)%(n+1)] = a_n
                state_memory[(t+1)%(n+1)] = (x_n, y_n, v_x, v_y)
                reward_memory[(t+1)%(n+1)] = reward
            tau = t - n+1
            #print(tau)
            if tau >= 0:
                G = find_G(tau, n, T, gamma, reward_memory)
                if tau+n < T:
                    x, y, d_v_x, d_v_y = state_memory[(tau+n)%(n+1)]
                    a = action_memory[(tau+n)%(n+1)]
                    #print(x,y,charge,a)
                    G += (gamma**n)*Q_s_a[x,y,d_v_x, d_v_y,a]
                x, y, d_v_x, d_v_y = state_memory[tau%(n+1)]
                a = action_memory[tau%(n+1)]
                Q_s_a[x,y,d_v_x, d_v_y,a] += alpha*(G-Q_s_a[x,y,d_v_x, d_v_y ,a])
            if tau == T-1:
                break
            x = x_n
            y = y_n
            v_x = v_x_n
            v_y = v_y_n
            a = a_n
            t = t+1

    return Q_s_a

def find_G(tau, n, T, gamma, reward_memory):
    G = 0
    i = (tau+1)%(n+1)
    pow = 0
    while(True):
        if i == T%(n+1) or i == (tau+n)%(n+1):
            break
        G = G + (gamma**pow)*reward_memory[i]
        i = (i+1)%(n+1)
        pow += 1
    return G

def main():
    the_icy_lake = np.loadtxt("the_icy_lake.txt")
    shape = the_icy_lake.shape + (9, 9, 4)
    Q_s_a = np.zeros(shape)

    #Q_s_a = monte_carlo_control_2(shape, the_icy_lake)
    #np.save("Q_s_a_e_soft", Q_s_a)
    #Q_s_a = np.load("Q_s_a_e_soft.npy")
    Q_s_a = n_step_sarsa(Q_s_a, the_icy_lake)
    np.save("Q_s_a_sarsa_4",Q_s_a)
    #Q_s_a = np.load("Q_s_a_sarsa_4.npy")
    #img = np.mean(Q_s_a, (2,3,4))
    #print(img.shape)
    #print(img)
    #plt.imshow(img)
    #plt.show()

    visited_states, trial_length = do_run(the_icy_lake, Q_s_a, 0)
    # print(Q_s_a)
    Q_s_a[Q_s_a == 0.0] = np.NINF
    print("Run completed in", trial_length)
    for i in range(trial_length, -1, -1):
        the_icy_lake[visited_states[i][0], visited_states[i][1]] = 2
    plt.imshow(the_icy_lake)
    plt.show()



main()