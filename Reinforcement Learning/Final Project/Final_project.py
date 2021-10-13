import numpy as np
import matplotlib.pyplot as plt
# Bridge:
# 0-clear
# 1-Hole
# 2-Old log

# state:
# 0-clear
# 1-Hole
# 2-Nothing
# 3-Old log
# Kan ikke gå på hull

# La Hubert automatisk plukke opp en planke på index 0 så heller fjerne dette senere

# q_s_a:
# (Current, Left, Right, Log, shield, birds, battery_level, A)

# Shield
# 0 Ikke aktivert
# 1 Aktivert

# action:
# 0 Left
# 1 Right
# 2 Fikse hull
# 3 Reparere Gammel Log
# 4 Aktivere skjold
# 5 Do-Nothing
# 6 Charge battery
# Fikser hullet til høyre for der hubert står,

# Rewards:
# Nå slutten 1
# Fikse hull 1
# Fikse gammel log=1

# Birds:
# Birds beveger seg
# [x,y,z,v] x = position, y = type, z = start position, v = retning
# y = 1 = pigeon
# y = 2 = eagle
# v = -1 = left
# v = 1 = right

# Ting å legge til på slutten
# En do-nothing action
# Plukke opp planken

# State kan representeres som en tuppel
def plot_rewards(total_rewards, counts):
    reward_per_step = total_rewards / counts
    plt.plot(reward_per_step)
    plt.show()

def main():
    do_run()
    #q_s_a = np.zeros((3, 4, 4, 2, 2, 3, 6), dtype=float)
    #q_s_a = np.load("q_s_a.npy")
    #q_s_a = sarsa(100, q_s_a, 0.1, 0.1)
    #np.save("q_s_a", q_s_a)
    #print("finished")

def do_run():
    average_rewards_per_step_averaged = np.zeros(100)
    for i in range(1000):
        q_s_a = np.zeros((3, 4, 4, 2, 2, 3, 4, 7), dtype=float)
        q_s_a, total_rewards, counts = sarsa(100, q_s_a, 0.1, 0.1)
        average_rewards_per_step_averaged += (total_rewards/counts)*1/1000
    plt.plot(average_rewards_per_step_averaged)
    plt.ylabel("Average reward per timestep averaged")
    plt.xlabel("Iteration")
    plt.title("Finding the average reward per time step for several iterations then averaging for each iteration")
    np.save("q_s_a", q_s_a)
    plt.show()

def sarsa(num_episodes, q_s_a, alpha, epsilon):
    total_rewards = np.zeros(num_episodes)
    counts = np.zeros(num_episodes)
    for i in range(num_episodes):
        birds, bridge = generate_map(10)
        # bridge_ages = np.zeros(10)

        position = 0
        fixing = 0
        charging = 0
        state = update_state((0,0,0,0,0,0,3), bridge, 0, birds)
        actions = find_possible_actions(bridge, position, state, fixing, charging)
        action = e_soft(q_s_a[state], epsilon, actions)

        hubert_pos_tracker = np.repeat('_', len(bridge))
        hubert_pos_tracker[position] = 'H'


        while position < len(bridge)-1:
            bridge, new_position, reward, new_state, fixing, birds, charging = do_action(bridge, action, position,
                                                                                         state, fixing, birds, charging)
            if (abs(position-new_position) > 1):
                print("Error position")

            actions = find_possible_actions(bridge, new_position, new_state, fixing, charging)
            new_action = e_soft(q_s_a[new_state], epsilon, actions)

            q_s_a[state][action] += alpha*(reward+q_s_a[new_state][new_action]-q_s_a[state][action])
            #hubert_pos_tracker[position] = '_'
            #hubert_pos_tracker[new_position] = 'H'

            position = new_position
            state = new_state
            action = new_action

            counts[i] += 1
            total_rewards[i] += reward
            #print(hubert_pos_tracker)
            #print(bridge)
            #if i % 1000 == 0:
                #print(np.array2string(hubert_pos_tracker, separator="", suffix="", prefix=""))
                #print(np.array2string(bridge), "\n")
        # total_rewards[i] += count
        print(counts[i])
        if i % 10 == 0:
            print(counts[i])
    #plot_rewards(total_rewards, counts)
    return q_s_a, total_rewards, counts


def age_bridge(birds, bridge, bridge_ages, position):
    bridge_ages += 0.01
    for i in range(1, len(bridge)-1):
        if np.random.random() < bridge_ages[i]:
            bridge_ages[i] = 0
            # If nothing replace by hole or old log
            if bridge[i] == 0:
                if np.random.random() >= 0.5 and position != i:
                    bridge[i] = 1
                else:
                    bridge[i] = 2
                    birds.append((np.array([i, np.random.randint(0,2), i, 1])))
            elif bridge[i] == 2:
                bridge[i] = 1
                # Removes birds belonging to the nest.
                old_birds = birds
                birds = list()
                for _, bird in enumerate(old_birds):
                    #print(bird)
                    if bird[2] != i:
                        birds.append(bird)
                        #break
    return birds, bridge, bridge_ages


def generate_map(length):
    bridge = np.zeros(length, dtype=int)
    birds = list()
    for j in range(1, len(bridge) - 1):
        n = np.random.random()
        if n > 0.66:
            bridge[j] = 1
        elif n > 0.5:
            bridge[j] = 2
            b = np.random.random()
            if b > 0.66:
                birds.append(np.array([j, 1, j, 1]))
            else:
                birds.append(np.array([j, 2, j, -1]))
    return birds, bridge


def do_birds(birds, fixing, action, hubert_pos, state):
    reward = 0
    for i, bird in enumerate(birds):
        if ( not bird[0] == bird[2] or not bird[2] == hubert_pos):
            #print(bird[0], bird[2])
            dist = abs(bird[0] - bird[2])
            if dist > 2:
                print("Error dist > 2")
            if dist == 2:
                bird[3] *= -1 #Reverse direction
            bird[0] += bird[3] #Move one step in direction
        if action == 2 or (fixing > 0 and state[4] == 0):
            reward -= 0.2*bird[1]
    return birds, reward


# False = losing log
# Shield is deactivated if fixing is 0
# Shield is activated if shield is true
# Nothing is done if shield is false
def update_state(state, bridge, position, birds, log=True, fixing=0, shield=False, charging=False, uses_charge=False):

    new_state = list(state)
    new_state[0] = bridge[position]
    new_state[1] = bridge[position-1] if position-1 >= 0 else 2
    new_state[2] = bridge[position+1] if position+1 < len(bridge) else 2
    if not log:
        #print(log)
        new_state[3] = 0

    #Disable shield and looking for birds if not fixing.
    if fixing == 0:
        new_state[4] = 0
        new_state[5] = 0
        #if shield:
        #    print("error shield + fixing == 0")
    if shield:
        new_state[4] = 1

    if uses_charge:
        new_state[6] -= 1
        if new_state[6] < 0:
            print("Error newstate 6 < 0")

    if charging:
        new_state[6] = 3

    #State only says if birds are nearby if Hubert is fixing something.
    if new_state[5] == 1:
        for _, bird in enumerate(birds):
            if abs(bird[0] - position) <= 1:
                if bird[1] == 1 and new_state[5] != 2:
                    new_state[5] = 1
                if bird[1] == 2:
                    new_state[5] = 2

    if position == 0:
        new_state[3] = 1
    return tuple(new_state)


def find_possible_actions(bridge, position, state, fixing, charging):
    possible_actions = list()

    # Kan også gjøre ingenting mens man fixer eller lader
    if fixing > 0 or charging > 0:
        possible_actions.append(5)
        # Kan også aktivere skjold hvis man fikser
        if state[4] != 1 and fixing > 0 and state[6] > 0:
            possible_actions.append(4)
        return possible_actions


    # Kan gå til venstre hvis han ikke står helt til venstre
    if position != 0:
        possible_actions.append(0)

    # Kan fikse hull hvis har planke og hull til høyre
    if state[3] == 1 and (position + 1 < len(bridge) and bridge[position+1] == 1):
        possible_actions.append(2)
        #print("Can fix hole")

    # Kan gå til høyre hvis ikke hull og ikke på slutten. Helt til høyre regnes som slutten.
    if position + 1 < len(bridge) and bridge[position+1] != 1:
        possible_actions.append(1)

    # Kan reparere gammel planke hvis står på gammel planke og har batteri
    if bridge[position] == 2 and state[6] > 0:
        possible_actions.append(3)

    # Kan lade batteriet om han står i posisjon 0 og batteriet ikke er fullt
    if position == 0 and state[6] != 3:
        possible_actions.append(6)


    return possible_actions


def do_action(bridge, action, position, state, fixing, birds, charging):
    birds, reward = do_birds(birds, fixing, action, position, state)
    if fixing == 2:
        fixing = 0

    if fixing == 1:
        fixing += 1

    if charging == 2:
        charging = 0

    if charging == 1:
        charging += 1

    #Sjekk at fixing ikke er > 0 på andre actions:
    if fixing > 0 and (action != 4 and action != 5):
        print("Error doing something you can't while fixing")

    # Nådd terminal state ved å komme helt til høyre
    if action == 1 and position+1 == len(bridge)-1:
        reward += 1
        position = position +1
        state = update_state(state, bridge, position, birds)
        return bridge, position, reward, state, fixing, birds, charging

    # Fikse hull:
    if action == 2:
        bridge[position+1] = 0
        reward += 0.35
        state = update_state(state, bridge, position, birds, log=False)

    # Fikse gammel log
    if action == 3:
        #print("action == 3")
        fixing += 1
        bridge[position] = 0
        reward += 1
        state = update_state(state, bridge, position, birds, uses_charge=True)
        #print("Fixing log")

    # Høyre
    if action == 1:
        if state[3] == 1:
            reward += 0.1
        else:
            reward -= 0.1
        position = position+1
        state = update_state(state, bridge, position, birds)

    # Venstre
    if action == 0:
        if state[3] == 1:
            reward -= 0.1
        else:
            reward += 0.1
        position = position-1
        state = update_state(state, bridge, position, birds)

    # Aktivere skjold
    if action == 4:
        #print("action == 4")
        state = update_state(state, bridge, position, birds, shield=True, fixing=fixing, uses_charge=True)

    # Do nothing
    if action == 5:
        state = update_state(state, bridge, position, birds, fixing=fixing, charging=charging)

    # Charging
    if action == 6:
        reward += 0.1
        charging += 1
        state = update_state(state, bridge, position, birds, charging=True)

    return bridge, position, reward, state, fixing, birds, charging


def e_soft(s_a, epsilon, actions):
    if np.random.random() <= epsilon:
        action = np.random.choice(actions)
        return action

    max_action_value = np.NINF
    greedy_action = None
    for _, action in enumerate(actions):
        if s_a[action] > max_action_value:
            greedy_action = action
            max_action_value = s_a[action]

    return greedy_action




main()