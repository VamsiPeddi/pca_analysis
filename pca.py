#################################
# File Author: Vamsi Peddi
# Year and Sem: Spring 2021
# Net ID: vpeddi
# Student ID: 9079650454
##################################


from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # TODO: add your code here
    dataset = np.load(filename)
    mean_array = np.mean(dataset, axis=0)

    dataset = dataset - mean_array

    return dataset


def get_covariance(dataset):
    # TODO: add your code here
    transposed = np.transpose(dataset)

    dot_prod = np.dot(transposed, dataset)

    S = (dot_prod)/(len(dataset)-1)

    return S


def get_eig(S, m):
    # TODO: add your code here

    #Getting m largest eigenvalues and eigenvectors
    n = len(S)
    vals, vector = eigh(S, eigvals=[n-m, n-1])

    #Reversing vals and making diagnal array
    vals = vals[::-1]
    vals = np.diag(vals)

    #Flipping vector array
    vector = np.fliplr(vector)

    return vals, vector


def get_eig_perc(S, perc):

    # Getting eigenvals greater than the variance
    start = 0
    vals, vector = eigh(S)
    n = len(vals)
    lambda_n = np.sum(vals)
    for i in vals:
        if(i/lambda_n) > perc:
            position = np.where(vals==i)
            start = position[0][0]
            break;

    # Reversing and making the vals array diagnal
    vals = vals[start:n]
    vals = vals[::-1]
    vals = np.diag(vals)

    # Flipping vector array and slicing it to get first two elements
    vector = np.fliplr(vector)
    vector = vector[:n,:2]

    return vals, vector


def project_image(img, U):
    # TODO: add your code here
    transpose = np.transpose(U)
    dot_prod = np.dot(img, U)
    image = np.dot(dot_prod, transpose)

    return image


def display_image(orig, proj):
    # TODO: add your code here
    # Reshaping the images to be 32x32
    image_orig = np.reshape(orig, (32,32))
    image_proj = np.reshape(proj, (32,32))
    image_orig = np.transpose(image_orig)
    image_proj = np.transpose(image_proj)

    # Creating a figure with 1 row and 2 subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,3))
    ax1.set_title('Original')
    ax2.set_title('Projection')
    
    # Rendering original image in first subplot
    pos = ax1.imshow(image_orig, aspect='equal')
    fig.colorbar(pos, ax=ax1)

    # Rendering projected image in second subplot
    neg = ax2.imshow(image_proj, aspect='equal')
    fig.colorbar(neg, ax=ax2)

    plt.show()
    return
