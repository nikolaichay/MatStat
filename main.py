import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import poisson, laplace, cauchy, uniform, norm, gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
from math import erf


NORMAL, CAUCHY, POISSON, UNIFORM, LAPLACE = "Нормальное распределение", "Распределение Коши", "Распределение Пуассона", "Равномерное распределение", "Рапределение Лапласса"
DENSITY, SIZE = "Плотность распределения", "Размер: "

def get_distribit(name, size):
    match name:
        case 'normal':
            return norm.rvs(size = size)
        case 'cauchy':
            return cauchy.rvs(0, 1, size = size)
        case 'poisson':
            return poisson.rvs(10, size = size)
        case 'uniform':
            return uniform.rvs(size = size, loc= -math.sqrt(3), scale = 2 * math.sqrt(3))
        case 'laplace':
            return laplace.rvs(0, 1/(math.sqrt(2)), size = size)
        
def get_density(name, array):
    match name:
        case 'normal':
            return [1 / (np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2) for x in array]
        case 'cauchy':
            return [1 / (np.pi * (x**2 + 1)) for x in array]
        case 'laplace':
            return [1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.fabs(x)) for x in array]
        case 'poisson':
            return [10 ** float(x) * np.exp(-10) / np.math.gamma(x) for x in array]
        case 'uniform':
            return [1 / (2 * np.sqrt(3)) if abs(x) <= np.sqrt(3) else 0 for x in array]

def get_func(name, x):
    match name:
        case 'normal':
            return 0.5 * (1 + erf(x / np.sqrt(2)))
        case 'cauchy':
            return np.arctan(x) / np.pi + 0.5
        case 'laplace':
            if x <= 0:
                return 0.5 * np.exp(np.sqrt(2) * x)
            else:
                return 1 - 0.5 * np.exp(-np.sqrt(2) * x)
        case 'poisson':
            return poisson.cdf(x, 10)
        case 'uniform':
            if x < -np.sqrt(3):
                return 0
            elif np.fabs(x) <= np.sqrt(3):
                return (x + np.sqrt(3)) / (2 * np.sqrt(3))
            else:
                return 1

name_dist = ['normal','cauchy','poisson','uniform','laplace']
#lab_1
sizes = [10 , 50 , 1000]
np.random.seed(10)
def norm_distribution():
    for s in sizes:
        density = norm()
        histogram = get_distribit(name_dist[0],s)
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram, bins = 15, density  = True, histtype = 'stepfilled', color = "skyblue")
        x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
        ax.plot(x, density.pdf(x), 'r--', lw = 0.7)
        ax.set_xlabel(NORMAL)
        ax.set_ylabel(DENSITY)
        ax.set_title(SIZE + str(s))
        plt.grid()
        #plt.show()
        plt.close(fig)
        fig.savefig('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab1/' + 'Normal' + str(s) +'.png' )
    return

norm_distribution()

def poisson_distribution():
    for s in sizes:
        density = poisson(10)
        histogram = get_distribit(name_dist[2],s)
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram, bins = 15, density  = True, histtype = 'stepfilled', color = "skyblue")
        x = np.arange(poisson.ppf(0.01,10), poisson.ppf(0.99,10))
        ax.plot(x, density.pmf(x), 'r--', lw = 0.7)
        ax.set_xlabel(POISSON)
        ax.set_ylabel(DENSITY)
        ax.set_title(SIZE + str(s))
        plt.grid()
        #plt.show()
        plt.close(fig)
        fig.savefig('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab1/' + 'Poisson' + str(s) +'.png' )
    return
np.random.seed(10)
poisson_distribution()

def cauchy_distribution():
    for s in sizes:
        density = cauchy(0, 1)
        histogram = get_distribit(name_dist[1],s)
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram,bins = 15, density = True, histtype = 'stepfilled', color = "skyblue")
        x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
        ax.plot(x, density.pdf(x), 'r--', lw = 0.7)
        ax.set_xlabel(CAUCHY)
        ax.set_ylabel(DENSITY)
        ax.set_title(SIZE + str(s))
        plt.grid()
        #plt.show()
        plt.close(fig)
        fig.savefig('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab1/' + 'Cauchy' + str(s) +'.png' )
    return

np.random.seed(10)
cauchy_distribution()

def uniform_distribution():
    for s in sizes:
        density = uniform(loc = -math.sqrt(3), scale=2 * math.sqrt(3))
        histogram = get_distribit(name_dist[3],s)
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram,bins = 15, density = True, histtype = 'stepfilled', color = "skyblue")
        x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
        ax.plot(x, density.pdf(x), 'r--', lw = 0.7)
        ax.set_xlabel(UNIFORM)
        ax.set_ylabel(DENSITY)
        ax.set_title(SIZE + str(s))
        plt.grid()
        #plt.show()
        plt.close(fig)
        fig.savefig('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab1/' + 'Uniform' + str(s) +'.png' )
    return
np.random.seed(10)
uniform_distribution()

def laplace_distribution():
    for s in sizes:
        density = laplace(0, 1/(math.sqrt(2)))
        histogram = get_distribit(name_dist[4],s)
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram,bins = 15, density = True, histtype = 'stepfilled', color = "skyblue")
        x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
        ax.plot(x, density.pdf(x), 'r--', lw = 0.7)
        ax.set_xlabel(LAPLACE)
        ax.set_ylabel(DENSITY)
        ax.set_title(SIZE + str(s))
        plt.grid()
        #plt.show()
        plt.close(fig)
        fig.savefig('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab1/' + 'Laplace'+str(s) +'.png' )
    return
np.random.seed(10)
laplace_distribution()
# lab_2

def trimmed_mean(distribit):
    r = int(len(distribit) / 4)
    sorted_distribit = np.sort(distribit)
    sum = 0.0
    for i in range(r + 1, len(distribit) - r):
        sum += sorted_distribit[i]
    return sum / (len(distribit) - 2 * r)

def mean_get(distribit):
    return np.mean(distribit)
def med_get(distribit):
    return np.median(distribit)
def zr_get(distribit):
    return (max(distribit)+min(distribit))/2
def zq_get(distribit):
    return (np.quantile(distribit, 1/4)+np.quantile(distribit,3/4))/2
def ztr_get(distribit):
    return trimmed_mean(distribit)

def static_value():
    sizes = [10 , 100 , 1000]
    repeat_count = 1000
    for name in name_dist:
        for s in sizes:
            mean, med, z_r, z_q, z_tr =[], [], [], [], []
            for i in range (repeat_count):
                distribit = get_distribit(name,s)
                mean.append(mean_get(distribit))
                med.append(med_get(distribit))
                z_r.append(zr_get(distribit))
                z_q.append(zq_get(distribit))
                z_tr.append(ztr_get(distribit))
            with open('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab2/' + 'res2' + name + str(s) + '.tex', "w") as file:
                file.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
                file.write("\\hline\n")
                file.write(str(name) + " n= " + str(s) + "& $\overline{x}$ & mediana & $z_r$ & $z_Q$ & $z_{tr}$ \\\ \\hline\n")
                file.write("E(z) & " + f"{np.around(np.mean(mean), decimals=4)} & "
                                    f"{np.around(np.mean(med), decimals=4)} & "
                                    f"{np.around(np.mean(z_r), decimals=4)} & "
                                    f"{np.around(np.mean(z_q), decimals=4)} & "
                                    f"{np.around(np.mean(z_tr), decimals=4)} \\\ \\hline\n")
                file.write("D(z) & " + f"{np.around(np.mean(np.multiply(mean, mean)) - np.mean(mean) * np.mean(mean), decimals=4)} & "
                                    f"{np.around(np.mean(np.multiply(med, med)) - np.mean(med) * np.mean(med), decimals=4)} & "
                                    f"{np.around(np.mean(np.multiply(z_r, z_r)) - np.mean(z_r) * np.mean(z_r), decimals=4)} & "
                                    f"{np.around(np.mean(np.multiply(z_q, z_q)) - np.mean(z_q) * np.mean(z_q), decimals=4)} & "
                                    f"{np.around(np.mean(np.multiply(z_tr, z_tr)) - np.mean(z_tr) * np.mean(z_tr), decimals=4)} \\\ \\hline\n")
                file.write("\\end{tabular}")
static_value()
# lab3
def tunkey_box():
    sizes = [20, 100]
    for name in name_dist:
        plt.figure()
        arr_20 = get_distribit(name, sizes[0])
        arr_100 = get_distribit(name, sizes[1])
        for a in arr_20:
            if a <= -50 or a >= 50:
                arr_20 = np.delete(arr_20, list(arr_20).index(a))
        for a in arr_100:
            if a <= -50 or a >= 50:
                arr_100 = np.delete(arr_100, list(arr_100).index(a))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp = ax.boxplot((arr_20, arr_100), patch_artist=True, vert=False, labels=["n = 20", "n = 100"])
        for whisker in bp['whiskers']:
            whisker.set(color="black", alpha=0.3, linestyle=":", linewidth=1.5)
        for flier in bp['fliers']:
            flier.set(marker="D", markersize=4)
        for box in bp['boxes']:
            box.set_facecolor('red')
            box.set(alpha=0.6)
        for median in bp['medians']:
            median.set(color='black')
        plt.ylabel("n")
        plt.xlabel("X")
        plt.title(name)
        # plt.show()
        plt.savefig("C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab3//" + name + ".jpg")
tunkey_box()

def count_emissions():
    sizes = [20, 100]
    repeats = 1000
    data = []
    for name in name_dist:
        count = 0
        for s in sizes:
            for i in range(repeats):
                arr = get_distribit(name, s)
                min = np.quantile(arr, 0.25) - 1.5 * (np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
                max = np.quantile(arr, 0.75) + 1.5 * (np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
                for k in range(0, s):
                    if arr[k] > max or arr[k] < min:
                        count += 1
            count /= repeats
            data.append(name + " n = $" + str(s) + "$ & $" + str(np.around(count / s, decimals=3)) + "$")
    with open('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab3/' + 'res3.tex', "w") as f:
        f.write("\\begin{tabular}{|c|c|}\n")
        f.write("\\hline\n")
        for row in data:
            f.write(row + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")
count_emissions()

def emperic():
    sizes = [20, 60, 100]
    for name in name_dist:
        if name == 'poisson':
            interval = np.arange(6, 15, 1)
        else:
            interval = np.arange(-4, 4, 0.01)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(wspace=0.5)
        fig.suptitle(name)
        for j in range(len(sizes)):
            arr = get_distribit(name, sizes[j])
            for a in arr:
                if name == 'poisson' and (a < 6 or a > 14):
                    arr = np.delete(arr, list(arr).index(a))
                elif name != 'poisson' and (a < -4 or a > 4):
                    arr = np.delete(arr, list(arr).index(a))
            ax[j].set_title("n = " + str(sizes[j]))
            if name == 'poisson':
                ax[j].step(interval, [get_func(name, x) for x in interval], color='red')
            else:
                ax[j].plot(interval, [get_func(name, x) for x in interval], color='red')
            if name == 'poisson':
                arr_ex = np.linspace(6, 14)
            else:
                arr_ex = np.linspace(-4, 4)
            ecdf = ECDF(arr)
            y = ecdf(arr_ex)
            ax[j].step(arr_ex, y, color='blue', linewidth=0.5)
            plt.savefig("C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab4/" + name + "_emperic.jpg")

def kernel():
    sizes = [20, 60, 100]
    for name in name_dist:
        for s in sizes:
            if name == 'poisson':
                x = np.arange(6, 15, 1)
            else:
                x = np.arange(-4,4,0.01)
            data = get_distribit(name,s)
            pdf = get_density(name, x)    
            scales = [0.5, 1.0, 2.0]
            fig, ax = plt.subplots(1, len(scales), figsize=(12, 4))
            fig.suptitle(f'{name}, n = {len(data)}')
            for i, scale in enumerate(scales):
                sns.kdeplot(data, ax=ax[i], bw_method='silverman', bw_adjust=scale, color = 'red')
                ax[i].set_xlim([x[0], x[-1]])
                ax[i].set_ylim([0, 1])
                ax[i].plot(x, get_density(name,x),color = 'blue')
                ax[i].set_title(f'h={str(scale)}*$h_n$')
            plt.savefig("C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/resLab4/" + name + str(s) + "_kernel.jpg")
            plt.close(fig)

if __name__ == "__main__":
    #norm_distribution()
    #cauchy_distribution()
    #poisson_distribution()
    #uniform_distribution()
    #laplace_distribution()
    static_value()
    #tunkey_box()
    #count_emissions()
    #emperic()
    #kernel()