import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
actions = np.array([-5, 0, 5])
model = nn.Sequential(
        nn.Linear(8, 1024),
        nn.BatchNorm1d(1024),
        #nn.Tanh(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        #nn.Tanh(),
        #nn.Tanh(),
        nn.Linear(512, 1),
       # nn.Hardsigmoid()
    )
adam = torch.optim.Adam(model.parameters(), lr=0.001)
loss_f = nn.MSELoss()
model = model.to(device)

def main():
    #hubert, cookie, x_pos, accumulated_rewards, X, y = do_run()
    #plot_hubert_cookie_pos(cookie, hubert, x_pos)
    #plot_accumulated_rewards(accumulated_rewards, x_pos)
    cycle(100_000)


def cycle(num_iterations):
    for i in range(num_iterations):
        hubert, cookie, x_pos, accumulated_rewards, X, y = do_run(True if i+1 < 1000 else False)
        train_model(X, y)
        if (i+1) % 100 == 0 and i+1 >= 1000:
            print("iteration:", i+1)
            plot_hubert_cookie_pos(cookie, hubert, x_pos)
            #print(model(X))
            plot_accumulated_rewards(accumulated_rewards, x_pos)


def train_model(X, y):
    model.train()

    # Forward
    out = model(X)
    out = out.to(device)
    # print(y-out)
    #_, preds = torch.max(out, 1)

    # Compute objective function
    loss = loss_f(out, y)

    # Clean the gradients
    model.zero_grad()

    # Accumulate partial deriviates wrt parameters
    loss.backward()

    # Step in the opposite direction og the gradient wrt optimizer
    adam.step()
    print(loss/len(X))

def find_a(x):
    out = model(x)
    #print(x)
    #print(out)
    return torch.argmax(out)


def do_run(randpolicy):
    model.train()

    i = 0
    size = int(100/0.2)
    X = torch.empty((size, 8), device=device)
    y = torch.empty((size, 1), device=device)

    positions_h = list()
    positions_c = list()
    accumulated_rewards = list()
    x_pos = list()
    rewards = list()



    total_reward = 0.0
    time = 0.0
    delta_time = 0.2
    time_passed_since_placed_cookie = 0.0
    cookie_pos = np.random.uniform(0, 10)
    pos = 5.0
    v = 0.0


    while time < 100:
        positions_h.append(pos)
        positions_c.append(cookie_pos)
        x_pos.append(time)

        x = torch.tensor([pos/10, cookie_pos/10, (pos-cookie_pos)/10, 1-time_passed_since_placed_cookie/5, v/5],
                         device=device, dtype=torch.float32)
        x = x.repeat(3, 1)
        one_hot = nn.functional.one_hot(torch.arange(0,3, device=device))
        x = torch.cat([x, one_hot], 1)
        print(x)
        a = np.random.randint(0,3) if randpolicy else find_a(x)
        new_v, new_pos, reward = find_new_velocity(v, a, pos)

        X[i] = x[a]

        #new_v, new_pos, reward = find_new_velocity(v, simple_policy(pos, cookie_pos), pos)
        #new_v, new_pos, reward = find_new_velocity(v, np.random.randint(0,3), pos)

        if check_cookie(new_v, new_pos, pos, cookie_pos):
            time_passed_since_placed_cookie = 0
            reward += 1
            cookie_pos = np.random.uniform(0, 10)

        time_passed_since_placed_cookie += 0.2
        if time_passed_since_placed_cookie >= 5:
            reward -= 0.5
            cookie_pos = np.random.uniform(0, 10)
            time_passed_since_placed_cookie = 0

        time += delta_time
        v = new_v
        pos = new_pos
        total_reward += reward

        rewards.append(reward)
        accumulated_rewards.append(total_reward)
        i += 1

    avg = 0
    for j, val in enumerate(reversed(rewards)):
        avg = val + 0.9*avg
        y[(size-1)-j] = torch.tensor([avg], device=device)
    #print(y)
    return positions_h, positions_c, x_pos, accumulated_rewards, X, y


def simple_policy(pos, cookie_pos):
    if pos < cookie_pos:
        return 2
    return 0


def less_simple_policy(pos, cookie_pos, v):
    if np.abs(v > 4):
        if v > 0:
            return 0
        return 2
    if np.abs(v) > 3:
        if (v > 0 and cookie_pos - pos > 0) or (v < 0 and cookie_pos - pos < 0):
            return 1
    if pos < cookie_pos:
        return 2
    return 0


def check_cookie(new_v, new_pos, old_pos, cookie):
    if (new_pos < cookie < old_pos) and np.abs(new_v) <= 4:
            return True
    if (old_pos < cookie < new_pos) and np.abs(new_v) <= 4:
            return True
    return False


def find_new_velocity(v, action, pos):
    rocket_fire = actions[action]
    friction = -np.abs(v)*v*0.05
    acceleration = rocket_fire + friction
    new_v = v + acceleration*0.2
    new_pos = pos + new_v*0.2
    if not (0 < new_pos < 10):
        return 0, int(new_pos/10)*10, -(v**2)*0.1
    return new_v, new_pos, 0

def plot_hubert_cookie_pos(cookie, hubert, x_pos):
    plt.plot(x_pos, hubert, label="Hubert")
    plt.plot(x_pos, cookie, label="Cookie")
    plt.title("Hubert and Cookie positions over time")
    plt.xlabel("Seconds")
    plt.ylabel("Position")
    plt.legend()
    plt.show()

def plot_accumulated_rewards(accumulated_rewards, x_pos):
    plt.plot(x_pos, accumulated_rewards)
    plt.title("Huberts accumulated rewards over time")
    plt.xlabel("Seconds")
    plt.ylabel("Accumulated Rewards")
    plt.show()


main()