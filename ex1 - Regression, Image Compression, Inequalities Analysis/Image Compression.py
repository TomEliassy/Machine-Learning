from numpy import diag,zeros
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
import scipy.misc as misc

n = 512


def diag_assign (s , mat):
    for i in range (len(s)):
        mat[i][i] = s[i]
    return mat

def find_rank(s):
    num_of_non_zero = 0
    for i in range (len(s)):
        if (s[i] is not 0):
            num_of_non_zero += 1

    return num_of_non_zero


img = misc.ascent()

u, s, vh = svd(img, True)

M_rank = find_rank(img)

# lists for saving the results
compress_ratio_lst = []
Frobenius_dist_lst = []

for k in range(512):

    # exctract the 512-k largest sigular values
    k_sigular_values = s[:k]

    # produce a diagonal matrix
    zeros_mat = zeros(shape=(len(u) , len(vh)))
    Sk = diag_assign(k_sigular_values , zeros_mat)

    # calculate the multiplication of the matrix
    Mk = u @ Sk @ vh

    # calculate the compression ratio
    compress_ratio_lst.append( ((2*k*n + k)/(2*n*M_rank + n)))


    # calculate the Frobenius distance between the original and the reconstructed images
    Frobenius_dist_lst.append(norm(img - Mk))

    if k in [0, 10 , 20 , 40 , 510]:
        title ('Image Compression \n k = ' + str(k))
        xlabel('Compression ratio = ' + str(compress_ratio_lst[k]) +' \n Norm = ' + str(Frobenius_dist_lst[k]))
        imshow(Mk)
        show()

figure(1)
plot(compress_ratio_lst , range(512) , label = "Compression Ratio")
title("Compression Ratio")
ylabel("k")
xlabel("comp. ratio")
show()

figure(2)
plot(Frobenius_dist_lst , range(512) , label = "Frobenius Distance")
title("Frobenius distance")
ylabel("k")
xlabel("Frob. dist.")
show()








