import numpy as np
import matplotlib.pyplot as plt
import skgeom as sg
from skgeom.draw import draw


def main():
    circles = np.array([(2, 9), (4, 5), (5, 5), (6, 5), (6, 2), (6, 12), (8, 10)
                           ,(11, 2), (11, 3), (11, 7)])
    asd = sg.Bbox2(0, 0, 13, 13)
    draw(asd)
    print("Hello")
    #current_point = (2,2)
    #rand_v = np.array([(np.random.random()-0.5)*2, (np.random.random()-0.5)*2])
    #new_point = current_point + rand_v
    #l = ((current_point[0], current_point[1]), (new_point[0], new_point[1]))
    #print(l.direction)
    #plot(circles)



main()

