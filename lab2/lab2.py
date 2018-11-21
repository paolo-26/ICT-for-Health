#!/usr/bin/env python3
"""
@author = Paolo Grasso
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans
import copy
import math

NC = 3
RANDOM_STATE = 0

class Image(object):

    def __init__(self, filename):
        self.im = mpimg.imread(filename)  # Original image
        self.im_quant = self.quantize()  # 3-color image
        self.find_shape()
        #self.prints()
        self.refine()

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

    def find_shape(self):
        # Find the color with minimum value of R+G+B.
        # Note that [0 0 0] is black and [255 255 255] is black.
        sc = np.sum(self.centroids, axis=1)  # Sum of RGB values
        i_col = sc.argmin()  # Color with minimum brightness

        N1, N2, N3 = self.im.shape
        im_clust = self.kmeans.labels_.reshape(N1, N2)
        zpos = np.argwhere(im_clust == i_col)

        print_image(self.im_quant, 'Converted - %d clusters' %NC)
        plt.show()
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

        if N_spots == 1:
            center_mole = np.median(zpos,axis=0).astype(int)

        else:
            # use K-means to get the N_spots clusters of zpos
            kmeans2 = KMeans(n_clusters=N_spots, random_state=RANDOM_STATE)
            kmeans2.fit(zpos)
            centers = kmeans2.cluster_centers_.astype(int)
            # the mole is in the middle of the picture:
            center_image = np.array([N1//2, N2//2])
            center_image.shape = (1,2)
            d = np.zeros((N_spots,1))

            for k in range(N_spots):
                d[k] = np.linalg.norm(center_image-centers[k,:])

            center_mole = centers[d.argmin(),:]

        # 6: take a subset of the image that includes the mole
        cond = True
        area_old = 0
        step = 10# each time the algorithm increases the area by 2*step pixels
        # horizontally and vertically
        c0 = center_mole[0]
        c1 = center_mole[1]
        im_sel = (im_clust == i_col)# im_sel is a boolean NDarray with N1 rows and N2 columns
        im_sel = im_sel*1# im_sel is now an integer NDarray with N1 rows and N2 columns

        while cond:
            subset = im_sel[c0-step:c0+step+1,c1-step:c1+step+ 1]
            area = np.sum(subset)

            if area > area_old:
                step = step + 10
                area_old = area
                cond = True

            else:
                cond = False
                # subset is the serach area

        plt.matshow(subset)
        self.subset = subset
        #plt.show()
        #print("subset = \n", subset, "\n")

    def clearing(self, img):
        #for i in range(2):

        # Clean isolated pixels
        cleared = copy.deepcopy(img)

        for r in range(2, self.subset.shape[0]-1):
            for c in range(2, self.subset.shape[1]-1):
                mini = [[cleared[x][y] for x in range(r-1,r+2)]
                                       for y in range (c-1,c+2)]
                if np.sum(mini) <= 2:
                    cleared[r][c] = 0
                if np.sum(mini) >= 8:
                    cleared[r][c] = 1


        #    sub = copy.deepcopy(inside)
        return cleared

    def refine(self):

        sub = self.clearing(self.subset)
        plt.matshow(sub)
        plt.title('cleared')


        outside = copy.deepcopy(sub)
        for i in range(5):
            for r in range(2, self.subset.shape[0]-1):
                for c in range(2, self.subset.shape[1]-1):
                    #mini = [[sub[x][y] for x in range(r-1,r+2)] for y in range (c-1,c+2)]
                    if ((sub[r-1][c] == 0) and (sub[r+1][c] == 0) and
                        (sub[r][c-1] == 0) and (sub[r][c+1] == 0) and
                        (sub[r][c] == 0)):
                        outside[r][c] = 0
                    else:
                        outside[r][c] = 1
                    # if np.sum(mini) <= 3:
                    #     sub[r][c] = 0
                    # if np.sum(mini) >= 7:
                    #     sub[r][c] = 1
            sub = copy.deepcopy(outside)

        plt.matshow(outside)
        plt.title('outside')

        inside = copy.deepcopy(outside)

        for i in range(5):
            for r in range(2, self.subset.shape[0]-1):
                for c in range(2, self.subset.shape[1]-1):
                    #mini = [[sub[x][y] for x in range(r-1,r+2)] for y in range (c-1,c+2)]
                    if ((sub[r-1][c] == 1) and (sub[r+1][c] == 1) and
                        (sub[r][c-1] == 1) and (sub[r][c+1] == 1) and
                        (sub[r][c] == 1)):
                        inside[r][c] = 1
                    else:
                        inside[r][c] = 0
                    # if np.sum(mini) <= 3:
                    #     sub[r][c] = 0
                    # if np.sum(mini) >= 7:
                    #     sub[r][c] = 1
            sub = copy.deepcopy(inside)

        plt.matshow(inside)
        plt.title('Inside')


        subber = (inside != outside)
        contour = copy.deepcopy(subber)

        for i in range(2):
            for r in range(3, self.subset.shape[0]-2):
                for c in range(3, self.subset.shape[1]-2):
                    #mini = [[sub[x][y] for x in range(r-1,r+2)] for y in range (c-1,c+2)]
                    if ((subber[r-1][c] == 1) and (subber[r+1][c] == 1) and
                        (subber[r][c-1] == 1) and (subber[r][c+1] == 1) and
                        (subber[r][c] == 1)):
                        # (subber[r-2][c] == 1) and (subber[r+2][c] == 1) and
                        # (subber[r][c-2] == 1) and (subber[r][c-2] == 1)):
                        contour[r][c] = 1
                    else:
                        contour[r][c] = 0
                    # if np.sum(mini) <= 3:
                    #     sub[r][c] = 0
                    # if np.sum(mini) >= 7:
                    #     sub[r][c] = 1
            subber = copy.deepcopy(contour)

        contour = contour*1
        # for r in range(1, contour.shape[0]):
        #     for c in range(1, contour.shape[1]):
        #         if contour[r][c] == True:
        #             contour[r][c] = 0
        #         if contour[r][c] == False:
        #             contour[r][c] = 1
        plt.matshow(contour)
        plt.title('Contour')



        defi = [[subber[x][y] for y in range(2,subber.shape[0]-1)] for x in range(2,subber.shape[1]-1)]
        # for r in range(subber.shape[0]):
        #     subber[r][0] = 0
        #     subber[r][subber.shape[0]-1] = 0
        #
        # for c in range(subber.shape[1]):
        #     subber[0][c] = 0
        #     subber[subber.shape[1]-1][c] = 0

        plt.matshow(defi)
        plt.show()

        self.area = np.sum(inside)
        #
        # # Perimeter
        # per = 0
        # for i in range(2):
        #     for r in range(2, self.subset.shape[0]-1):
        #         for c in range(2, self.subset.shape[1]-1):
        #             if sub[r][c] == 1:
        #                 neigh = 0
        #                 if sub[r-1][c] == 1:
        #                     neigh += 1
        #                 if sub[r+1][c] == 1:
        #                     neigh += 1
        #                 if sub[r][c-1] == 1:
        #                     neigh +=1
        #                 if sub[r][c+1] == 1:
        #                     neigh +=1
        #                     print(neigh)
        #                 per += (4-neigh)
        #
        self.perimeter = np.sum(defi)
        self.circle_perimeter = 2 * math.pi * math.sqrt(self.area/math.pi)

    def info(self):
        print("Area: %d" % self.area)
        print("Perimeter: %d" % self.perimeter)
        print("Circle perimeter: %.2f" % self.circle_perimeter)
        ratio = (self.perimeter/self.circle_perimeter)
        print("Ratio = %.2f" % ratio)

    def prints(self):
        print_image(self.im, 'Original')
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
    imag = Image('./moles/melanoma_1.jpg')
    imag.info()
    print(' --- END --- ')
