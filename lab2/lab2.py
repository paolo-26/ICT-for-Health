#!/usr/bin/env python3
"""
@author = Paolo Grasso
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans
import copy

NC = 3
RANDOM_STATE = 0

class Image(object):

    def __init__(self, filename):
        self.im = mpimg.imread(filename)  # Original image
        self.im_quant = self.quantize()  # 3-color image
        self.find_middle()
        self.prints()

    def quantize(self):
        im_2D = convert_to_2D(self.im)
        im_2D_quant = copy.deepcopy(im_2D)  # A copy of 2D image
        self.kmeans = KMeans(n_clusters=NC, random_state=RANDOM_STATE)
        self.kmeans.fit(im_2D)
        self.kmeans.cluster_centers_
        self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_.astype('uint8')  # 8-bit RGB colors

        for kc in range(NC):
            index = (self.kmeans.labels_ == kc)
            im_2D_quant[index,:] = self.centroids[kc,:]

        N1, N2, N3 = self.im.shape
        return convert_to_3D(im_2D_quant, N1, N2, N3)

    def find_middle(self):
        # Find the color with minimum value of R+G+B.
        # Note that [0 0 0] is black and [255 255 255] is black.
        sc = np.sum(self.centroids, axis=1)  # Sum of RGB values
        i_col = sc.argmin()  # Color with minimum brightness

        N1, N2, N3 = self.im.shape
        im_clust = self.kmeans.labels_.reshape(N1, N2)
        zpos = np.argwhere(im_clust == i_col)

        N_spots = int(input("How many spots? \n> "))

        # if N_spots == 1:
        #     center_mole = np.median(zpos, axis=0).astype(int)
        # else:
        #     kmeans2 = KMeans(n_clusters=N_spots, random_state=0)
        #     kmeans2.fit(zpos)
        #     centers = kmeans2.cluster_centers_.astype(int)
        #
        #     center_image = np.array([N1//2, N2//2])
        #     center_image.shape = (1,2)
        #     d = np.zeros((N_spots,1))
        #
        #     for k in range(N_spots):
        #         d[k] = np.linalg.norm(center_image-centers[k,:])
        #         center_mole = centers[d.argmin(),:]

    def prints(self):
        print_image(self.im, 'Original')
        print_image(self.im_quant, 'Converted - %d clusters' %NC)
        plt.show()

def convert_to_2D(img):
    N1, N2, N3 = img.shape
    return img.reshape((N1*N2, N3))

def convert_to_3D(img, N1, N2, N3):
    return img.reshape((N1, N2, N3))

def print_image(img,title=None):
    plt.figure()
    plt.imshow(img)
    plt.title(title)


if __name__ == '__main__':
    imag = Image('./moles/low_risk_1.jpg')
    print(' --- END --- ')
