import scipy.misc
import glob
import numpy as np
import pickle

def create_data_set(directory, transformation, dataset_name, labels = 17):
    """From a folder with pictures creates list of touples (if labeled) or list of vectors (if not labeled).
    Every picture is transformed. Labels are binary vectors with size equall to number of categories."""

    # Index of category
    category = 0

    # List of pictures as vectors and labels
    image_list = []

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

            # Append it to list
            image_list.append(image)

    with open('Objects\\' + dataset_name, 'wb') as f:
        pickle.dump(image_list, f, pickle.HIGHEST_PROTOCOL)

    return image_list


def identity(x):
    return x

def resize(x, height = 40, width = 40):

    x = scipy.misc.imresize(x, (height, width))
    return x

if __name__ == "__main__":

    # image_path = 'C:\\Users\\Elio\\Documents\\Data Science A-Z\\20170407IdentifyingThemesfromMobileCaseImages\\' \
    #              '2. Prepared Data\\Train\\Automobile\\Img52.jpg'
    #
    #
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

    x = create_data_set(directory, identity, 'debug_dataset')

    print(len(x))
    print(x[5])

    print(0)