import numpy as np
import random


class Literal:
    def __init__(self, value, name):
        self.value = value
        self.name = name
        self.in_target = False
        self.in_hypothesis = True
        self.negation = None

    def __str__(self):
        return str(self.name) + "  " + str(self.value)

    def __repr__(self):
        return str(self.name) + " " + str(self.value)+", "


def generate_literals(n):
    literals = np.empty(2*n, dtype=Literal)
    for i in range(n):
        literals[2*i] = Literal(True, i)
        literals[2*i+1] = Literal(False, i)
        literals[2*i].negation = literals[2*i+1]
        literals[2*i+1].negation = literals[2*i]
        # print(literals[2*i])
        # print(literals[2*i+1])
    return literals


def generate_target(length, literals, distribution, noise, n):
    target = np.empty(length, dtype=Literal)
    for i in range(length):
        contradicting_literal = True
        while contradicting_literal:
            index = random.randint(0, n-1)
            value = distribution[index] + random.randint(-noise, noise)
            if not literals[2*index].in_target and not literals[2*index+1].in_target:
                if value < 50:
                    target[i] = literals[2*index+1]
                    target[i].in_target = True
                    contradicting_literal = False
                else:
                    target[i] = literals[2*index]
                    target[i].in_target = True
                    contradicting_literal = False
    return target


def generate_example(literals, distribution, noise, n):
    example = np.empty(n, dtype=Literal)
    for i in range(0, n):
        value = distribution[i] + random.randint(-noise, noise)
        if value < 50:
            example[i] = literals[2*i+1]
        else:
            example[i] = literals[2*i]
    return example


def generate_distribution(size):
    distribution = np.random.binomial(100, 0.5, size)
    # distribution = np.random.normal(50, 4, size)
    return distribution


def positive_example(example):
    pos_example = True
    for i in range(example.size):
        if example[i].negation.in_target:
            pos_example = False
    return pos_example


def update_hypothesis(example):
    if positive_example(example):
        for i in range(example.size):
            example[i].negation.in_hypothesis = False


def print_hypothesis(literals):
    print("Hypothesis: ")
    for literal in literals:
        if literal.in_hypothesis:
            print(literal)


# Returns fraction of correct guesses by hypothesis
def estimate_error(n_error_estimate, literals, distribution, noise, n):
    correct_guesses = 0
    for i in range(n_error_estimate):
        example = generate_example(literals, distribution, noise, n)
        if positive_example(example):
            correct_guess = True
            for j in range(example.size):
                if example[j].in_target and example[j].negation.in_hypothesis:
                    correct_guess = False
            if correct_guess:
                correct_guesses += 1
        else:
            correct_guesses += 1
    return float(correct_guesses/n_error_estimate)


def main():
    n = int(input("Literals"))
    literals = generate_literals(n)
    noise = 1
    distribution = generate_distribution(n)
    target = generate_target(10, literals, distribution, noise, n)
    epsilon = float(input("Epsilon"))
    delta = float(input("Delta"))
    n_error_estimate = int((20/epsilon)**2)

    print(distribution)
    print("Target: ")
    print(target)

    # Required size to have less than epsilon error with 1-delta confidence
    req_size = int(((2*n)/epsilon*(np.log(2*n)+np.log(1/delta))))
    print("Required Size: ", req_size)

    for i in range(req_size):
        example = generate_example(literals, distribution, noise, n)
        update_hypothesis(example)

    print_hypothesis(literals)
    accuracy = estimate_error(n_error_estimate, literals, distribution, noise, n)
    print("Estimated Accuracy ", accuracy)
