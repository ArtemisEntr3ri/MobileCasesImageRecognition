import scipy.misc
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage import color

def create_data_set(directory, transformation, dataset_name, labels = 17):
    """From a folder with pictures creates list of touples (if labeled) or list of vectors (if not labeled).
    Every picture is transformed. Labels are binary vectors with size equall to number of categories.

    directory - putanja do foldera u kojem se nalaze svi folderi kategorija slika sa slikama, ili samo folder sa slikama
                ako se radi o testnom njihovom skupu

    transformation - funkcija koja prima matrice slika (tenzor) i vraca transfomiranu sliku u obliku vektora stupca
    dataset_name - ime kako zelimo spremiti dataset
    """

    # Index of category
    category = 0

    # List of pictures as vectors and labels
    image_list = []

    # Vector column length (check cause some images are 4 matrices) we discard them
    dimension_check = 0

    if labels > 0:
        for category_folder in glob.iglob(directory + '*\\'):
            print(category)
            for image_path in glob.iglob(category_folder + '*.*'):

                # Read image as matrices
                image = scipy.misc.imread(image_path)

                # Transform image to vector shape
                image = transformation(image)

                # Make image 1D vector column
                image = image.ravel()
                image = image.reshape(image.shape[0], 1)

                if dimension_check == 0:
                    dimension_check = image.shape[0]
                elif dimension_check != image.shape[0]:
                    continue

                # Set category
                label = np.zeros((labels, 1))
                label[category] = 1

                # Append touple to list
                image_list.append((image, label))
            # Next category
            category += 1
    # If there are no labels
    else:
        for image_path in glob.iglob(directory + '*\\*.*'):
            # Read image as matrices
            image = scipy.misc.imread(image_path)

            # Transform image to vector shape
            image = transformation(image)

            # Make image 1D vector column
            image = image.ravel()
            image = image.reshape(image.shape[0], 1)

            if dimension_check == 0:
                dimension_check = image.shape[0]
            elif dimension_check != image.shape[0]:
                continue

            # Append it to list
            image_list.append(image)

    with open('Objects\\' + dataset_name, 'wb') as f:
        pickle.dump(image_list, f, pickle.HIGHEST_PROTOCOL)

    return image_list

def identity(x):
    return x

def resize(x, height = 20, width = 10):
    x = scipy.misc.imresize(x, (height, width))
    x = color.rgb2gray(x)
    x = x/255
    return x



if __name__ == "__main__":

    image_path = 'C:\\Users\\Elio\\Documents\\Data Science A-Z\\20170407IdentifyingThemesfromMobileCaseImages\\' \
                 '2. Prepared Data\\Train\\Automobile\\Img52.jpg'

    # x = resize(scipy.misc.imread(image_path))
    # plt.imshow(x)
    # plt.show()
    # print(resize(scipy.misc.imread(image_path)))

    # x = scipy.misc.imread(image_path)
    #
    # print(x.shape)
    #
    # x = scipy.misc.imresize(x, (200, 200))
    #
    # print(x.shape)
    #
    # x = x.ravel()
    #
    # print(x.shape)


    directory = 'C:\\Users\\Elio\\Documents\\Data Science A-Z\\20170407IdentifyingThemesfromMobileCaseImages\\' \
                  '2. Prepared Data\\Train\\'

    x = create_data_set(directory, resize, 'debug_dataset2')

    print(0)