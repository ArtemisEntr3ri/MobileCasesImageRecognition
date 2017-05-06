import numpy as np
import random


def dodaj1(x):
    return x+1

def dodaj5(x):
    return x+5


def nesto(x, funkcija):
    return funkcija(x)

if __name__ == "__main__":


    #print(nesto(5, dodaj5))

    x = np.array([5, 3, 2])

    num = np.dot(x[:-1], x[1:]) + sum(x[1:])
    print(num)
    print(0)



