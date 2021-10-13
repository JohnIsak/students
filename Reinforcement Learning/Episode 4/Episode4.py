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
    if (-1 < x+v_x < 75 and  y+v_y == 150):
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
            if y == 150:
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

def main():
    the_icy_lake = np.loadtxt("the_icy_lake.txt")
    shape = the_icy_lake.shape + (9, 9, 4)
    #print(shape)
    # show_lake(the_icy_lake)
    #visited_states, trial_length = do_run(the_icy_lake)
    #for i in range(trial_length, -1, -1):

    #    the_icy_lake[visited_states[i][0], visited_states[i][1]] = 2
    #print("Trial Length", trial_length)
    #optimal_policy, Q_s_a = monte_carlo_control(shape, the_icy_lake)
    #np.save("optimal_policy", optimal_policy)
    #np.save("Q_s_a", Q_s_a)
    #optimal_policy = np.load("optimal_policy.npy")
    #Q_s_a = np.load("Q_s_a.npy")
    #Q_s_a[Q_s_a == 0] = np.NINF
    #img = np.mean(optimal_policy, (2,3)) * -1
    #print(img.shape)
    #print(img)
    #plt.imshow(img)
    #plt.show()

    #Q_s_a = monte_carlo_control_2(shape, the_icy_lake)
    #np.save("Q_s_a_e_soft", Q_s_a)
    Q_s_a = np.load("Q_s_a_e_soft.npy")
    img = np.mean(Q_s_a, (2,3,4))
    print(img.shape)
    print(img)
    plt.imshow(img)
    plt.show()

    visited_states, trial_length = do_run(the_icy_lake, Q_s_a, 0)
    # print(Q_s_a)
    Q_s_a[Q_s_a == 0.0] = np.NINF
    print("Run completed in", trial_length)
    for i in range(trial_length, -1, -1):
        the_icy_lake[visited_states[i][0], visited_states[i][1]] = 2
    plt.imshow(the_icy_lake)
    plt.show()



main()