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

    x=np.zeros((3, 1))
    print(x.shape)
    x = x.reshape(3,)
    print(x.shape)
    print(0)



