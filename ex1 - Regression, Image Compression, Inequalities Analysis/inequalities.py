import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.pylab import plot, legend, xlabel, ylabel, ylim, show, figure,title
import math

#-------------------------------------------------Generating the data--------------------------------------------------


data = np.random.binomial(1,0.25,(100000,1000))
epsilons = [0.5,0.25,0.1,0.01,0.001]



#-----------------------------------------------------Section A--------------------------------------------------------
def calc_x_bar (tosses):
    means = []
    acc = np.cumsum(tosses)
    for j in range (1, len(tosses)+1):
        means.append(acc[j-1]/j)

    return means # len(means) ==1000

def calc_estimation():
    figure(1)

    # 1st row
    means = calc_x_bar(data[0])
    plot(range(1,1001), means, 'c', label=('1st row'))
    # 2nd row
    means = calc_x_bar(data[1])
    plot(range(1,1001), means, 'm', label=('2nd row'))
    # 3rd row
    means = calc_x_bar(data[2])
    plot(range(1,1001), means, 'y' , label=('3rd row'))
    # 4th row
    means = calc_x_bar(data[3])
    plot(range(1,1001), means, 'k' , label=('4th row'))
    # 5th row
    means = calc_x_bar(data[4])
    plot(range(1,1001), means, 'r', label=('5th row'))

    title("Mean of 1,000 tosses")
    legend(loc='best')
    xlabel("m")
    ylabel("p estimate")
    ylim(0,0.6)
    show()

calc_estimation()



#------------------------------------------------------Section B-------------------------------------------------------
# As mentioned in class, the upper bound of the variance for a Bernoulli
# indicator with probability p is 1/4
def Chebyshev_calc (m , epsilon):
    upper_bounds = []
    for j in range (1 , m+1):
        upper_bounds.append(min(1/(4 * j * (epsilon**2)),1))
    return upper_bounds

def Hoeffding (m , epsilon):
    upper_bounds = []
    for j in range (1 , m+1):
        upper_bounds.append(min(2 * math.exp((-2) * j * (epsilon **2)),1))
    return upper_bounds

def calc_lists (epsilon):
    Chev_lst = Chebyshev_calc(1000 , epsilon)
    Hoff_lst = Hoeffding(1000 , epsilon)
    return Chev_lst , Hoff_lst


def plot_upper_bounds ():
    for epsilon in [0.5,0.25,0.1,0.01,0.001]:
        Chev_lst , Hoff_lst = calc_lists (epsilon)
        plot_graphs(Chev_lst , Hoff_lst , epsilon , False)


def plot_graphs (Chev_lst , Hoff_lst , epsilon , percentage , percent_list = None):
    figure(1)
    if percentage:
        plot(range(1,1001) , percent_list , label = "percentage")
    plot(range(1,1001) , Chev_lst , label = "Chebyshev")
    plot(range(1,1001) , Hoff_lst , label = "Hoeffding")
    title("Hoeffding and Chebyshev Boundaries \n for epsilon = " + str(epsilon))
    legend(loc = 'upper right')
    xlabel("m")
    ylabel("upper bound")
    show()



plot_upper_bounds()


#-----------------------------------------------------------Section C--------------------------------------------------

def inequality_calc(tosses , epsilon , prob):
    x_bar = calc_x_bar(tosses)

    return [(1 if (math.fabs(item - prob) >= epsilon) else 0) for item in x_bar]


def find_precentage (database, epsilons , prob):
    for epsilon in epsilons:
        num_of_rows_hold = 0
        percent_list = [0] * 1000
        for i in range (1,100001):
            row_i =  inequality_calc(database[i-1] , epsilon , prob)
            for j in range(len(row_i)):
                if row_i[j]:
                    percent_list[j] += 1

            if ( i% 5000 == 0):
                print(i)
        percent_list = [i / 100000 for i in percent_list]
        Chev_lst , Hoff_lst = calc_lists(epsilon)
        plot_graphs(Chev_lst , Hoff_lst , epsilon , True , percent_list)




find_precentage(data , epsilons , 0.25)




