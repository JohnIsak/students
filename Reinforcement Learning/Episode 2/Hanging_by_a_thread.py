import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

class String:
    def __init__(self, k_mean, k_std, h_mean, h_std, price):
        self.k_mean = k_mean
        self.k_std = k_std
        self.h_mean = h_mean
        self.h_std = h_std
        self.price = price


def main():
    #level1(100000)
    #level2(100000)
    #level3(100000)
    #level4(1000, 0)
    make_plot_level_4()

def make_plot_level_4():
    np.random.seed(1337)
    rewards = np.zeros((1000, 1000))
    avg_rewards = np.zeros(1000)
    fig, ax = plt.subplots()
    epsilons = np.array([0, 0.01, 0.1, 1])
    labels = np.array(["$\epsilon = 0$","$\epsilon = 0.01$", "$\epsilon = 0.1$", "$\epsilon = 1$"])
    for j in range(4):
        eps = epsilons[j]
        for i in range(1000):
            rewards[i] = level4(1000, eps)

        for i in range(1000):
            avg_rewards[i] = np.mean(rewards[:,i])

        ax.plot(avg_rewards, label=labels[j])
        plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("Average reward for a greedy algorithm with different epsilons")

    plt.show()


def level4(it, eps):
    strings = generate_strings()
    action_values = np.repeat(40, 10)
    #action_values = np.zeros(10)
    action_n = np.zeros(10)
    rewards = np.zeros(it)
    for i in range(it):
        if (np.random.uniform(0,1) > eps):
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0,10)
        rewards[i] = hubert_does_his_best(strings[action])
        action_n[action] += 1
        #action_values[action] += (1/action_n[action])*(rewards[i]-action_values[action])
        action_values[action] += 0.1*(rewards[i] - action_values[action])
    print(action_n)
    return rewards

def hubert_does_his_best(string):
    m = np.power(np.random.uniform(1, 10), 1.5)
    reward = -string.price
    if snaps(string, m):
        reward += m * 0.7
    else:
        reward += 100 + 1.1 * np.sqrt(m)
    return reward


def level1(it):
    np.random.seed(42)
    M = np.zeros(it)
    for i in range(it):
        m = np.power(np.random.uniform(1, 10), 1.5)
        M[i] = m
    cost = M * 0.7
    print(np.mean(cost))
    print(st.norm.interval(alpha=0.95, loc=np.mean(cost), scale=st.sem(cost)))


def level2(it):
    np.random.seed(42)
    profit = np.zeros(it)
    for i in range(it):
        m = np.power(np.random.uniform(1, 10), 1.5)
        profit[i] = 100 + 1.1*np.sqrt(m)
    print(np.mean(profit))
    print(st.norm.interval(alpha=0.95, loc=np.mean(profit), scale=st.sem(profit)))


def level3(it):
    np.random.seed(42)
    strings = generate_strings()
    means = np.zeros(10)
    errors = np.zeros((2,10))
    names = np.empty(10, dtype=str)
    for j in range(10):
        profit = np.zeros(it)
        string = strings[j]

        for i in range(it):
            m = np.power(np.random.uniform(1, 10), 1.5)
            profit[i] -= strings[j].price
            if snaps(string, m):
                profit[i] += m*0.7
            else:
                profit[i] += 100 + 1.1*np.sqrt(m)

        print("String", j+1)
        print(np.mean(profit))
        print(st.norm.interval(alpha=0.95, loc=np.mean(profit), scale=st.sem(profit)))
        means[j] = np.mean(profit)
        names[j] = str(j + 1)
        errors[:, j] = st.norm.interval(alpha=0.95, loc=np.mean(profit), scale=st.sem(profit))
        errors[0, j] = np.abs(means[j] - errors[0, j])
        errors[1, j] = np.abs(means[j] - errors[1, j])

    plot_means_per_string(errors, means, names)


def plot_means_per_string(errors, means, names):
    x_pos = np.arange(10)
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=errors, align="center", ecolor="red", capsize=10)
    ax.set_ylabel("Profit (Kr)")
    ax.set_xlabel("String-type")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title("Expected profit for different string-types")
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()


def snaps(string, m):
    h = np.abs(np.random.normal(string.h_mean, string.h_std))
    k = np.abs(np.random.normal(string.k_mean, string.k_std))
    exponent = -np.power((m/(1+h)), 1+k)
    p_oh_no = 1 - np.power(np.e, exponent)
    return p_oh_no >= np.random.uniform(0, 1)


def generate_strings():
    strings = np.empty(10, String)
    strings[0] = String(9, 1, 9, 3, 10)
    strings[1] = String(7, 2, 11, 2, 15)
    strings[2] = String(15, 3, 4, 2, 5)
    strings[3] = String(12, 1, 4, 1, 15)
    strings[4] = String(15, 5, 15, 5, 15)
    strings[5] = String(14, 3, 8, 2, 10)
    strings[6] = String(9, 1, 9, 4, 10)
    strings[7] = String(6, 3, 7, 3, 5)
    strings[8] = String(13, 1, 12, 1, 15)
    strings[9] = String(10, 2, 6, 3, 5)
    return strings


main()
