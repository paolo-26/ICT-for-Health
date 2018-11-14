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
        self.im_or = mpimg.imread(filename)
        self.im_2D = self.convert_2D()
        self.im_2D_quant = copy.deepcopy(self.im_2D)
        self.kmeans = KMeans(n_clusters=NC, random_state=RANDOM_STATE)
        self.kmeans.fit(self.im_2D)
        self.kmeans.cluster_centers_
        self.kmeans.labels_
        self.run()

    def run(self):
        centroids = self.kmeans.cluster_centers_.astype('uint8')  # RGB colors

        for kc in range(NC):
            index = (self.kmeans.labels_ == kc)
            self.im_2D_quant[index,:] = centroids[kc,:]

        self.im_3D_quant = self.convert_3D()
        print_image(self.im_3D_quant, 'Converted - %d clusters' %NC)
        i_col = self.find_darkest(centroids)  # Find darkest color
        im_clust = self.kmeans.labels_.reshape(self.N1, self.N2)
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

    def find_darkest(self, centroids):
        sc = np.sum(centroids, axis=1)
        i_col = sc.argmin()
        return i_col

    def convert_2D(self):
        self.N1, self.N2, self.N3 = self.im_or.shape
        return self.im_or.reshape((self.N1*self.N2, self.N3))

    def convert_3D(self):
        return self.im_2D_quant.reshape((self.N1, self.N2, self.N3))


def print_image(img,title=None):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    imag = Image('./moles/low_risk_1.jpg')
    print(' --- END --- ')
