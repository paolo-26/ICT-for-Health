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
import pandas as pd

NC = 3
RANDOM_STATE = 0
filename = 'low_risk_1'
F = './moles/'+filename+'.jpg'


class Image(object):

    def __init__(self, filename):
        """ Auto-run the whole code
        """
        self.filename = filename
        self.im = mpimg.imread(filename)  # Original image
        self.im_quant = self.solve_k_means()  # 3-colour image
        self.centre_image()  #2-colour image
        self.polish()

    def save_to_file(self):
        """ Save processed image
        """
        plt.matshow(self.processed_image)
        plt.savefig(self.filename.replace('.jpg', '.png'))

    def solve_k_means(self):
        """ Apply k-means clustering
        """
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

    def centre_image(self):
        """ Find the center of the mole and reshape the image
            to a 2-color version
        """
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

        if N_spots == 0:
            print("Try to change cluster number")

        if N_spots == 1:
            center_mole = np.median(zpos,axis=0).astype(int)

        else:
            # use K-means to get the N_spots clusters of zpos
            kmeans2 = KMeans(n_clusters=N_spots, random_state=RANDOM_STATE)
            kmeans2.fit(zpos)
            centers = kmeans2.cluster_centers_.astype(int)
            # the mole is in the middle of the picture:
            center_image = np.array([N1//2, N2//2])
            center_image.shape = (1, 2)
            d = np.zeros((N_spots, 1))

            for k in range(N_spots):
                d[k] = np.linalg.norm(center_image-centers[k, :])

            center_mole = centers[d.argmin(), :]

        # Take a subset of the image that includes the mole.
        c0 = center_mole[0]
        c1 = center_mole[1]
        RR, CC = im_clust.shape
        stepmax = min([c0, RR-c0, c1, CC-c1])
        cond = True
        area_old = 0
        surf_old = 1
        step = 10  # each time the algorithm increases the area by 2*step pixels
        # horizontally and vertically
        # im_sel is a boolean NDarray with N1 rows and N2 columns
        im_sel = (im_clust == i_col)
        im_sel = im_sel * 1  # im_sel is now an integer NDarray with N1 rows and N2 columns
        while cond:
            subset = im_sel[c0 - step:c0 + step + 1, c1 - step:c1 + step + 1]
            area = np.sum(subset)
            Delta = np.size(subset) - surf_old
            surf_old = np.size(subset)
            if area > area_old + 0.01*Delta:
                step = step + 10
                area_old = area
                cond = True
                if step > stepmax:
                    cond = False
            else:
                cond = False
                # subset is the search area
        print_image(subset,'2 colours')
        # if filename == 'low_risk_1' or filename == 'melanoma_23':
        #     plt.savefig('grainy.pdf')
        self.subset = subset

    def clean_in(self, img, r1=4):  # Clean from outside to inside
        """ Start from the centre of the mole, going to the sides.
            Fill all 0-pixels that have more than N neighbors.
            Return cleaned image.
        """
        dim = self.subset.shape[0]
        cleared = copy.deepcopy(img)
        for r in range(2, round(dim/2)):
            for c in range(2, round(dim/2)):
                mini = [[cleared[x][y] for x in range(r-1, r+2)]
                                       for y in range (c-1, c+2)]
                if cleared[r][c] == 0:
                    if np.sum(mini) >= r1:
                        cleared[r][c] = 1

        for r in range(2, round(dim/2)):
            for c in range(round(dim/2), dim-1)[::-1]:
                mini = [[cleared[x][y] for x in range(r-1, r+2)]
                                       for y in range (c-1, c+2)]
                if cleared[r][c] == 0:
                    if np.sum(mini) >= r1:
                        cleared[r][c] = 1

        for r in range(round(dim/2), dim-1)[::-1]:
            for c in range(round(dim/2), dim-1)[::-1]:
                mini = [[cleared[x][y] for x in range(r-1, r+2)]
                                       for y in range (c-1, c+2)]
                if cleared[r][c] == 0:
                    if np.sum(mini) >= r1:
                        cleared[r][c] = 1

        for r in range(round(dim/2), dim-1)[::-1]:
            for c in range(2, round(dim/2)):
                mini = [[cleared[x][y] for x in range(r-1, r+2)]
                                       for y in range (c-1, c+2)]
                if cleared[r][c] == 0:
                    if np.sum(mini) >= r1:
                        cleared[r][c] = 1

        return cleared

    def clean_out(self, img, r1=4):  # Clean from inside to outside
        """ Start from the corners of the image, going to the centre.
            Clean all 1-pixels that have less than N neighbors.
            Return cleaned image.
        """
        dim = self.subset.shape[0]
        cleared = copy.deepcopy(img)
        for r in range(2, round(dim/2))[::-1]:
            for c in range(2, round(dim/2))[::-1]:
                mini = [[cleared[x][y] for x in range(r-1, r+2)]
                                       for y in range (c-1, c+2)]
                if cleared[r][c] == 1:
                    if np.sum(mini) <= r1:
                        cleared[r][c] = 0

        for r in range(2, round(dim/2))[::-1]:
            for c in range(round(dim/2), dim-1):
                mini = [[cleared[x][y] for x in range(r-1, r+2)]
                                       for y in range (c-1, c+2)]
                if cleared[r][c] == 1:
                    if np.sum(mini) <= r1:
                        cleared[r][c] = 0

        for r in range(round(dim/2), dim-1):
            for c in range(round(dim/2), dim-1):
                mini = [[cleared[x][y] for x in range(r-1, r+2)]
                                       for y in range (c-1, c+2)]
                if cleared[r][c] == 1:
                    if np.sum(mini) <= r1:
                        cleared[r][c] = 0

        for r in range(round(dim/2), dim-1):
            for c in range(2, round(dim/2))[::-1]:
                mini = [[cleared[x][y] for x in range(r-1, r+2)]
                                       for y in range (c-1, c+2)]
                if cleared[r][c] == 1:
                    if np.sum(mini) <= r1:
                        cleared[r][c] = 0

        return cleared

    def fill_horiz(self, img):
        """ Fill horizontal holes that are preceded and succeeded by
            at least N = dim/5 pixels.
            Return cleaned image.
        """
        flag = 0
        dim = img.shape[0]-1
        par = round(dim/5)
        for r in range(1, dim):
            start = 0
            stop = 0
            for c in range(par, round(dim)):
                v = [img[r][x] for x in range(c-par, c)]
                if img[r][c] == 0 and np.sum(v) == len(v):
                    start = c
                    break
            if start != 0:
                for c in range(start, dim):
                    if img[r][c] == 1:
                        stop = c
                        try:
                            w = [img[r][x] for x in range(stop, stop+par)]
                            if np.sum(w) >= len(w) - 2:
                                for c in range(start, stop):
                                    img[r][c] = 1
                                break
                            else:
                                stop = 0
                        except:
                            pass
        return img


    def polish(self):
        """ Call all the cleaning functions and find area and perimeter.
        """
        sub = self.subset

        print("- Cleaning inside...")
        sub = self.clean_in(sub)
        print_image(sub, 'Inside cleaning')

        print("- Cleaning outside...")
        sub = self.clean_out(sub)
        print_image(sub, 'Outside cleaning')

        print("- Filling holes...")
        sub = self.fill_horiz(sub)
        print_image(sub, 'Holes filling')

        print("- Cleaning outside...")
        sub = self.clean_out(sub, r1=5)

        print("- Filling holes...")
        sub = self.fill_horiz(sub)

        self.im_area = copy.deepcopy(sub)  # Area image
        inside = copy.deepcopy(self.im_area)

        for r in range(2, self.subset.shape[0]-1):
            for c in range(2, self.subset.shape[1]-1):
                if ((sub[r-1][c] == 1) and (sub[r+1][c] == 1) and
                    (sub[r][c-1] == 1) and (sub[r][c+1] == 1) and
                    (sub[r][c] == 1)):
                    inside[r][c] = 1
                else:
                    inside[r][c] = 0

        sub = copy.deepcopy(inside)
        contour = (inside != self.im_area)
        contour = contour*1  # Contour image
        print_image(contour, 'Perimeter')
        self.processed_image = (contour+self.im_area)**2  # Contour+area image
        print_image(self.processed_image, 'Processed')
        # if filename == 'medium_risk_10' or filename == 'melanoma_23':
        #     plt.savefig('processed3.pdf')
        self.perimeter = np.sum(contour)  # Perimeter
        self.area = np.sum(self.im_area)-self.perimeter  # Area
        self.circle_perimeter = 2 * math.pi * math.sqrt(self.area/math.pi)


    def info(self):
        """ Print results
        """
        print("Filename: %s" % self.filename)
        print("Area: %d" % self.area)
        print("Perimeter: %d" % self.perimeter)
        print("Circle perimeter: %.2f" % self.circle_perimeter)
        self.ratio = (self.perimeter/self.circle_perimeter)
        print("Ratio = %.2f" % self.ratio)

    def prints(self):
        print_image(self.im, 'Original')


class Results(object):
    """ Load and save results on a csv file
    """
    def __init__(self, filename):
        self.data = pd.read_csv('results.csv', index_col=0)
        print(self.data)

    def add(self, name, area, perimeter, ratio):
        try:
            self.data.loc[name] = [area, perimeter, round(ratio, 6)]
        except:
            print("Error")

        self.data = self.data.sort_index()
        with open('results.csv', 'w') as f:
            self.data.to_csv(f)


def convert_to_2D(img):
    N1, N2, N3 = img.shape
    return img.reshape((N1*N2, N3))

def convert_to_3D(img, N1, N2, N3):
    return img.reshape((N1, N2, N3))

def print_image(img,title=None):
    """ Print matrix images with title
    """
    plt.figure()
    plt.imshow(img, cmap="Greys")
    plt.title(title)
    plt.savefig(title+'.pdf')

def main():
    c = Results('results.csv')
    imag = Image(F)
    imag.info()
    plt.show()
    imag.save_to_file()
    c.add(filename, imag.area, imag.perimeter, imag.ratio)
    print('\n --- END --- ')

if __name__ == '__main__':
    main()
