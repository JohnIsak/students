import numpy as np
import matplotlib.pyplot as plt

class State():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # FORWARD
    # z = 0 -> y+1
    # z = 1 -> x-1
    # z = 2 -> y-1
    # z = 3 -> x-1

    # LEFT
    # z = z+1

    # RIGHT
    # z = z-1
    def left(self):
        x = self.x
        y = self.y
        z = self.z
        if self.z == 0:
            x = self.x-1
            z = 1
        if self.z == 1:
            y = self.y-1
            z = 2
        if self.z == 2:
            x = self.x+1
            z = 3
        if self.z == 3:
            y = self.y+1
            z = 0
        return x, y, z
    def forward(self):
        x = self.x
        y = self.y
        z = self.z
        if self.z == 0:
            y = self.y+1
        elif self.z == 1:
            x = self.x-1
        elif self.z == 2:
            y = self.y-1
        elif self.z == 3:
            x = self.x+1
        return x, y, z
    def right(self):
        x = self.x
        y = self.y
        z = self.z
        if self.z == 0:
            x = self.x+1
            z = 3
        elif self.z == 1:
            y = self.y+1
            z = 2
        elif self.z == 2:
            x = self.x-1
            z = 1
        elif self.z == 3:
            y = self.y-1
            z = 0
        return x, y, z


def main():
    # the_hill = np.loadtxt("the_hill.txt")
    # lars_time(the_hill)
    # a_gradient = anne_lise_time(the_hill)
    # h_gradient = hubert_time(the_hill)
    # plot_optimal_routes(h_gradient, the_hill.copy(), "Huberts optimal route")
    # plot_optimal_routes(a_gradient, the_hill.copy(), "Anne-Lise's optimal route")
    the_backyard = np.loadtxt("the_backyard.txt")
    lars_time(the_backyard)
    #values_anne_lise = approximate_values(the_backyard, bellman_a)
    # np.save("values_anne_lise", values_anne_lise)
    values_anne_lise = np.load("values_anne_lise.npy")
    print(values_anne_lise[15,0,0])
    # values_hubert = approximate_values(the_backyard, bellman_h)
    # np.save("values_hubert", values_hubert)
    values_hubert = np.load("values_hubert.npy")
    plot_optimal_routes_2(values_hubert, the_backyard.copy(), "Hubert's optimal route")
    plot_optimal_routes_2(values_anne_lise, the_backyard.copy(), "Anne-Lise's optimal route")


def lars_time(map):
    time = np.sum(map[15,:])
    print(time)


def anne_lise_time(the_hill):
    values_anne_lise = np.zeros(the_hill.shape)

    def dynamic_shit(x,y):
        x = 0 if x < 0 else x
        x = 30 if x > 30 else x
        if values_anne_lise[x, y] != 0:
            return values_anne_lise[x, y]
        if y == 99:
            values_anne_lise[x, y] = -the_hill[x, y]
            return values_anne_lise[x, y]
        values_anne_lise[x, y] = -the_hill[x, y] + (dynamic_shit(x+1, y+1) + dynamic_shit(x, y+1) + dynamic_shit(x-1, y+1))/3
        return values_anne_lise[x, y]
    return values_anne_lise

def hubert_time(the_hill):
    values_hubert = np.zeros(the_hill.shape)

    def dynamic_shit_2(x, y):
        x = 0 if x < 0 else x
        x = 30 if x > 30 else x
        if values_hubert[x, y] != 0:
            return values_hubert[x, y]
        if y == 99:
            values_hubert[x, y] = -the_hill[x, y]
            return values_hubert[x, y]
        values_hubert[x, y] = -the_hill[x, y] + max(dynamic_shit_2(x+1, y+1),
                                                    max(dynamic_shit_2(x, y+1), dynamic_shit_2(x-1, y+1)))
        return values_hubert[x, y]

    dynamic_shit_2(15, 0)
    return values_hubert

def plot_optimal_routes(values, the_hill, str):
    values = values*(-1)
    time = np.zeros(1)

    def dynamic_shit_3(x, y):
        values[x, y] = values[15,0]
        time[0] += the_hill[x, y]
        the_hill[x, y] = 1
        if (y == 99):
            return
        min = values[x, y+1]
        x_n = x
        if(x != 29 and values[x+1, y+1] < min):
            min = values[x+1, y+1]
            x_n = x+1
        if(x != 0 and values[x-1, y+1] < min):
            x_n = x-1
        dynamic_shit_3(x_n, y+1)

    dynamic_shit_3(15, 0)
    print(str, time[0])
    plt.imshow(values)
    plt.title(str)
    plt.show()
    plt.imshow(the_hill)
    plt.title(str)
    plt.show()

# Defined these functions before the helper class unfortunately.
def bellman_a(the_backyard, values_anne_lise, x, y, z):
    v = -the_backyard[x, y]
    if z == 0:
        v += 1/3*values_anne_lise[x, y+1, z] if y+1 != 100 else (1/3)*values_anne_lise[x, y, z]
        v += 1/3*values_anne_lise[x-1, y, z+1] if x-1 != -1 else (1/3)*values_anne_lise[x, y, z]
        v += 1/3*values_anne_lise[x+1, y, 3] if x+1 != 31 else (1/3)*values_anne_lise[x, y, z]
    if z == 1:
        v += 1/3*values_anne_lise[x-1, y, z] if x-1 != -1 else (1/3)*values_anne_lise[x, y, z]
        v += 1/3*values_anne_lise[x, y-1, z+1] if y-1 != -1 else (1/3)*values_anne_lise[x, y, z]
        v += 1/3*values_anne_lise[x, y+1, z-1] if y+1 != 100 else (1/3)*values_anne_lise[x, y, z]
    if z == 2:
        v += 1/3*values_anne_lise[x, y-1, z] if y-1 != -1 else (1/3)*values_anne_lise[x, y, z]
        v += 1/3*values_anne_lise[x+1, y, z+1] if x+1 != 31 else (1/3)*values_anne_lise[x, y, z]
        v += 1/3*values_anne_lise[x-1, y, z-1] if x-1 != -1 else (1/3)*values_anne_lise[x, y, z]
    if z == 3:
        v += 1/3*values_anne_lise[x+1, y, z] if x+1 != 31 else (1/3)*values_anne_lise[x, y, z]
        v += 1/3*values_anne_lise[x, y+1, 0] if y+1 != 100 else (1/3)*values_anne_lise[x, y, z]
        v += 1/3*values_anne_lise[x, y-1, z-1] if y-1 != -1 else (1/3)*values_anne_lise[x, y, z]
    return v


def bellman_h(the_backyard, values_h, x, y, z):
    v = -the_backyard[x, y]
    if z == 0:
        v1 = values_h[x, y+1, z] if y+1 != 100 else values_h[x, y, z]
        v2 = values_h[x-1, y, z+1] if x-1 != -1 else values_h[x, y, z]
        v3 = values_h[x+1, y, 3] if x+1 != 31 else values_h[x, y, z]
    if z == 1:
        v1 = values_h[x-1, y, z] if x-1 != -1 else values_h[x, y, z]
        v2 = values_h[x, y-1, z+1] if y-1 != -1 else values_h[x, y, z]
        v3 = values_h[x, y+1, z-1] if y+1 != 100 else values_h[x, y, z]
    if z == 2:
        v1 = values_h[x, y-1, z] if y-1 != -1 else values_h[x, y, z]
        v2 = values_h[x+1, y, z+1] if x+1 != 31 else values_h[x, y, z]
        v3 = values_h[x-1, y, z-1] if x-1 != -1 else values_h[x, y, z]
    if z == 3:
        v1 = values_h[x+1, y, z] if x+1 != 31 else values_h[x, y, z]
        v2 = values_h[x, y+1, 0] if y+1 != 100 else values_h[x, y, z]
        v3 = values_h[x, y-1, z-1] if y-1 != -1 else values_h[x, y, z]
    v += max(v1, max(v2, v3))
    return v


def approximate_values(the_backyard, v_function):
    shape = the_backyard.shape + (4,)
    values_anne_lise = np.zeros(shape)
    iterations = 0
    while True:
        iterations += 1
        delta = 0.0
        for x in range(len(values_anne_lise)):
            # Going this way with y will make it converge a lot faster.
            for y in range(len(values_anne_lise[0])-1,-1,-1):
                for z in range(len(values_anne_lise[0, 0])):
                    v = values_anne_lise[x, y, z]
                    if(x != 15 or y != 99):
                        values_anne_lise[x, y, z] = v_function(the_backyard, values_anne_lise, x, y, z)
                        delta = max(delta, abs(v-values_anne_lise[x, y, z]))

        if delta < 0.00001:
            break
    return values_anne_lise

def plot_optimal_routes_2(values, the_backyard, str):
    values = values*(-1)
    time = np.zeros(1)
    values_show = np.mean(values, axis=2)

    def dynamic_shit_3(x, y, z):
        pos = State(x, y, z)
        values_show[x, y] = values_show[15, 0]
        time[0] += the_backyard[x, y]
        the_backyard[x, y] = 1
        if (y == 99 and x == 15):
            return
        min = np.Inf
        x_n, y_n, z_n = pos.forward()
        if (-1 < x_n < 31 and -1 < y_n < 100) and values[x_n, y_n, z_n] < min:
            min = values[x_n, y_n, z_n]
            x, y, z = x_n, y_n, z_n
        x_n, y_n, z_n = pos.left()
        if (-1 < x_n < 31 and -1 < y_n < 100) and values[x_n, y_n, z_n] < min:
            min = values[x_n, y_n, z_n]
            x, y, z = x_n, y_n, z_n
        x_n, y_n, z_n = pos.right()
        if (-1 < x_n < 31 and -1 < y_n < 100) and values[x_n, y_n, z_n] < min:
            x, y, z = x_n, y_n, z_n
        dynamic_shit_3(x, y, z)

    dynamic_shit_3(15, 0, 0)
    print(str, time[0])
    plt.imshow(values_show)
    plt.title(str)
    plt.show()
    plt.imshow(the_backyard)
    plt.title(str)
    plt.show()

main()