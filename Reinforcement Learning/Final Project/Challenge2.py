import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.functional as F
import matplotlib.pyplot as plt

# One hot encoding av state:
# 0 nothing/not seen
# 1 empty tile
# 2 wall
# 3 jewel
# 4 Deadly chemical
# 5 Trash
# 6 Bin
# 7 Treasure
# 8 Safe

# + En egen variabel som sier om det er en robot der eller ikke.
# 0 1
# + En egen variabel som sier om den er visited eller ikke
# 0 1

# Actions:
# 0 Up
# 1 Left
# 2 Down
# 3 Right
# 4 Pick-Up
# 5 Drop

device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")


def generate_map():
    map = np.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 7, 2, 5, 5, 5, 7, 5, 7, 2],
                    [2, 7, 5, 2, 5, 5, 2, 5, 7, 2],
                    [2, 7, 4, 5, 3, 7, 7, 7, 2, 2],
                    [2, 5, 7, 2, 2, 5, 4, 5, 7, 2],
                    [2, 5, 7, 2, 7, 5, 4, 4, 2, 2],
                    [2, 5, 7, 1, 7, 2, 5, 7, 5, 2],
                    [2, 5, 1, 5, 7, 2, 7, 4, 2, 2],
                    [2, 6, 8, 2, 4, 2, 2, 5, 7, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    map[map == 0] = 1
    #print(map)
    return map


class Robots():
    def __init__(self, positions, directions):
        self.positions = positions
        self.directions = directions

    def move(self, map):
        for i in range(len(self.positions)):
            #print(self.positions[i])
            # Change directions
            if np.random.random() > 0.5:
                self.directions[i] = np.random.randint(0, 4)
            # Move forward
            else:
                new_position = np.copy(self.positions[i])
                direction = np.copy(self.directions[i])
                if direction == 0:
                    new_position[0] += 1
                elif direction == 1:
                    new_position[1] += 1
                elif direction == 2:
                    new_position[0] -= 1
                elif direction == 3:
                    new_position[1] -= 1
                if (len(map) > new_position[0] >= 0 and len(map[0]) > new_position[1] >= 0
                        and map[new_position[0], new_position[1]] != 2):
                    # print("here")
                    self.positions[i] = new_position
            # Gemerate trash:
            # print(self.positions[i])
            # print(tuple(self.positions[i]))
            if map[tuple(self.positions[i])] == 1:
                if np.random.random() < 0.1:
                    map[tuple(self.positions[i])] = 5

    def check_collision(self, position):
        for i in range(len(self.positions)):
            if self.positions[i][0] == position[0] and self.positions[i][1] == position[1]:
                return True
        return False

def plot_shit():
    counts = np.load("Counts.npy")
    rewards = np.load("Rewards.npy")
    #print(counts[900:950])
    #print(rewards[900:950])
    avg_rewards_per_step = rewards / counts
    avg = pd.DataFrame(avg_rewards_per_step)
    #print(avg)
    avg = avg.iloc[:, 0].rolling(window=100).mean()
    plt.plot(avg)
    plt.title("First 1000 iterations, Moving avg, window=100")
    plt.ylabel("average reward per step")
    plt.xlabel("Iteration")
    plt.show()
    #print(rewards)
    #print(counts)
    try_2 = np.zeros(10)
    j = 0
    for i in range(1000):
        try_2[j] += rewards[i]/counts[i]
        if (i+1) % 100 == 0:
            j =j+1
            print(j)
    #print(try_2)
    #plt.plot(try_2)
    #plt.show()

    try_3_r = np.zeros(10)
    try_3_c = np.zeros(10)
    j = 0
    for i in range(1000):
        try_3_c[j] += counts[i]
        try_3_r[j] += rewards[i]
        if (i+1) % 100 == 0:
            j += 1

    test = try_3_r/try_3_c
    nums = np.arange(len(test))
    nums += 1
    print(test)
    plt.plot(nums, test)
    plt.title("Average rewards per step per 100 iterations")
    plt.ylabel("Average reward per step")
    plt.xlabel("Iteraton")
    plt.show()
    quit()

def main():
    #plot_shit()
    model = nn.Sequential(
        nn.Linear(258, 64),
        nn.Linear(64, 64),
        nn.Linear(64, 32),
        nn.Linear(32, 1)
    )
    #model = torch.load("model_continuous_2")
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_f = nn.MSELoss()
    model.train()
    counts = np.zeros(1000)
    rewards = np.zeros(1000)
    for i in range(1000):
        map = generate_map()
        visited = np.zeros((10,10))
        treasure_carried = 0
        trash_carried = 0
        robots = Robots(np.array([[1, 1], [1, 2],[1, 8], [8, 8]]), np.array([2, 2, 2, 0]))
        position = [8, 1]
        s_A, actions = update_state(position, map, robots, visited, trash_carried, treasure_carried)
        #print(s_A)

        with torch.no_grad():
            new_q_s_A = model(s_A)
        max, max_a, index = eps_greedy(new_q_s_A, actions)
        max = model(s_A[index])

        j = 0

        while(True):
            visited[tuple(position)] = 1
            map, reward, position, trash_carried, treasure_carried = do_action(map, max_a, position, robots, trash_carried, treasure_carried, visited)
            rewards[i] += reward
            robots.move(map)
            new_s_A, actions = update_state(position, map, robots, visited, trash_carried, treasure_carried)
            # print(reward)
            if map[tuple(position)] == 3 and max_a == 4:
                # print("here")
                y = torch.tensor([reward], dtype=torch.float, device=device)
                loss += loss_f(max, y)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                break

            with torch.no_grad():
                new_q_s_A = model(new_s_A)

            new_max, new_max_a, index = eps_greedy(new_q_s_A, actions, eps=0.2)

            y = new_max.data + reward

            # print(max, y)
            if j == 0:
                loss = loss_f(max, y)

            model.zero_grad()
            if (j+1) %1024 == 0:
                loss.backward()
                optimizer.step()
                loss = loss_f(max, y)

            else:
                loss += loss_f(max, y)
            #with torch.autograd.set_detect_anomaly(True):
            #   loss.backward()
            new_max = model(new_s_A[index])
            #new_max = max[0]
            #new_max, hidden = model(new_s_A[index], hidden)

            max_a = new_max_a
            max = new_max
            if counts[i] > 10000:
                print(max, max_a, reward, position)
            counts[i] += 1
            j += 1

        #loss.backward()

        print(visited)
        print(map)
        print(counts[i], rewards[i], rewards[i]/counts[i], i)
    np.save("Counts", counts)
    np.save("Rewards", rewards)
    torch.save(model, "model_continuous_3")


def eps_greedy(new_q_s_A, actions, eps=0.1):
    new_max = torch.tensor([np.NINF])
    new_max_a = None
    if np.random.random() > eps:
        for i, q_s_a in enumerate(new_q_s_A):
            if q_s_a.data[0] > new_max.data[0]:
                new_max = q_s_a
                new_max_a = actions[i]
                index = i
    else:
        choice = np.random.randint(0, len(new_q_s_A))
        new_max = new_q_s_A[choice]
        new_max_a = actions[choice]
        index = choice
    return new_max, new_max_a, index


def find_possible_actions(position, map, trash_carried, treasure_carried):
    possible_actions = list()
    if position[0] + 1 < len(map) and map[position[0] + 1, position[1]] != 2:
        possible_actions.append(0)
    if position[1] + 1 < len(map[0]) and map[position[0], position[1] + 1] != 2:
        possible_actions.append(1)
    if position[0] > 0 and map[position[0] - 1, position[1]] != 2:
        possible_actions.append(2)
    if position[1] > 0 and map[position[0], position[1] - 1] != 2:
        possible_actions.append(3)
    if (map[tuple(position)] == 3 or (map[tuple(position)] == 5 and trash_carried < 10) or
       (map[tuple(position)] == 7 and treasure_carried < 10)):
        possible_actions.append(4)
    if (map[tuple(position)] == 6 and trash_carried > 0) or map[tuple(position)] == 8 and treasure_carried > 0:
        possible_actions.append(5)

    return possible_actions

def check_wall(x, y, position, map):
    wall = False
    if x == -2 and map[position[0]+y, position[1]+x+1] == 2:
        wall = True
    if y == -2 and map[position[0] + y+1, position[1] + x] == 2:
        wall = True
    if x == 2 and map[position[0]+y, position[1]+x-1] == 2:
        wall = True
    if y == 2 and map[position[0] + y-1, position[1] + x] == 2:
        wall = True
    return wall


def update_state(position, map, robots, visited, trash_carried, treasure_carried):
    state = list()
    for y in range(-2, 3):
        for x in range(-2, 3):
            if 0 <= position[0] + y < len(map) and 0 <= position[1] + x < len(map[0]) and not check_wall(x, y, position, map):
                sub_state = np.zeros(8)
                sub_state[map[position[0] + y][position[1] + x] - 1] = 1
                # print(sub_state, x, y, map[position[0]+y,position[1]+x])
                state.extend(sub_state)
                state.append(1) if robots.check_collision([position[0] + y, position[1] + x]) else state.append(0)
                state.append(1) if visited[position[0] + y, position[1] + x] == 1 else state.append(0)
            else:
                state.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    state.append(0.1*trash_carried)
    state.append(0.1*treasure_carried)
    state.extend([0, 0, 0, 0, 0, 0])  # Extend possible actions
    possible_actions = find_possible_actions(position, map, trash_carried, treasure_carried)
    s_a = np.tile(state, (len(possible_actions), 1))
    for i in range(len(possible_actions)):
        s_a[i][-5 + possible_actions[i]] = 1
    state = torch.tensor(s_a, dtype=torch.float, device=device)
    return state, possible_actions


def do_action(map, a, position, robots, trash_carried, treasure_carried, visited):
    reward = 0
    if a == 0:
        reward -= 0.1
        position[0] += 1
    if a == 1:
        reward -= 0.1
        position[1] += 1
    if a == 2:
        reward -= 0.1
        position[0] -= 1
    if a == 3:
        reward -= 0.1
        position[1] -= 1
    if a == 4 and map[position[0], position[1]] == 3:
        reward += 100
    if a == 4 and map[position[0], position[1]] == 7:
        map[position[0], position[1]] = 1
        treasure_carried += 1
        if treasure_carried > 10:
            print("WTF")
        reward += 0.2

    if a == 4 and map[position[0], position[1]] == 5:
        map[position[0], position[1]] = 1
        trash_carried += 1
        reward += 0.1

    if a == 5 and map[position[0], position[1]] == 6:
        print(trash_carried)
        reward += 1 * trash_carried
        trash_carried = 0
        print("dropped off trash")

    if a == 5 and map[position[0], position[1]] == 8:
        print(treasure_carried)
        reward += 10 * treasure_carried
        treasure_carried = 0
        print("dropped off treasure")

    if map[position[0], position[1]] == 4:
        # print("stepped in shit")
        reward -= 1
    if robots.check_collision(position):
        #print("Collided with robot")
        reward -= 1

    if visited[tuple(position)] == 0:
        reward += 0.2

    return map, reward, position, trash_carried, treasure_carried

main()
