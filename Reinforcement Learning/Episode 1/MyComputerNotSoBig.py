import numpy as np
import itertools
import matplotlib.pyplot as plt


class Factory:
    def __init__(self):
        self.rooms = np.array([0,0,0,0,0])

    def generate_waste(self):
        self.rooms[0] += 1
        self.rooms[1] += 3
        self.rooms[2] += 1
        self.rooms[4] += 2


class Hubert:

    def __init__(self, factory):
        self.maxCapacity = 10
        self.current_trash = 0
        self.position = 0
        self.factory = factory

    def move_left(self):
        if self.position == 0:
            self.position = 1
        else:
            self.position -= 1

    def move_right(self):
        if self.position == 4:
            self.position = 3
        else:
            self.position += 1

    def stay(self):
        if self.position == 3:
            self.factory.rooms[3] += self.current_trash
            self.current_trash = 0

        else:
            pickup = min(self.maxCapacity-self.current_trash, self.factory.rooms[self.position])
            self.current_trash += pickup
            self.factory.rooms[self.position] -= pickup


def main():
    seq_length = 6
    commands = map("".join, itertools.product("rls", repeat=seq_length))
    possible_sequences = int(np.math.pow(3, seq_length))
    weights = np.zeros(possible_sequences)
    i = 0
    for command in commands:
        weights[i] = routine(command)
        i += 1
    print_distribution(weights)

def routine(sequence):
    factory = Factory()
    hubert = Hubert(factory)
    for command in sequence:
        factory.generate_waste()
        if command == "r":
            hubert.move_right()
        elif command == "l":
            hubert.move_left()
        elif command == "s":
            hubert.stay()
        else:
            return NotImplemented

    return factory.rooms[3]

def print_distribution(weights):
    x_min = np.min(weights)
    print("min", x_min)
    x_max = np.max(weights)
    print("max", x_max)
    mean = np.mean(weights)
    print("mean", mean)
    weights_normalized = weights/x_max

    close_to_best = np.count_nonzero(weights_normalized >= 0.9)/len(weights_normalized)
    print(close_to_best)

    plt.hist(weights, density=True)
    plt.title("Frequency Distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Weight")
    plt.show()


main()