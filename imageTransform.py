
from scipy import misc
from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np

def transformation(image):
    return image

I = image.imread("D:\Image recognition - Spring 2017\Data science\Train\Automobile\Img52.jpg")
print(type(I), I.shape)
I = transformation(I)

white = np.full((I.shape[0], 300, 3), 255)

print(white[0:5, 0:5, ])
plt.imshow(white)
plt.show()
print(type(I))
new = np.concatenate((white, white), axis = 1)

tmp = np.concatenate((white, I), axis = 1)

print(tmp.shape)

plt.imshow(tmp)
plt.show()

print(tmp[0:10, 0:10, ])


misc.imsave('D:\Image recognition - Spring 2017\Data science\Deb\image.png', tmp);