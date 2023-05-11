import scipy.stats as stats
import scipy.optimize as opt
from scipy.stats import laplace, uniform, norm
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tabulate import tabulate
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import math 

def multivariateNormal(size, ro):
    return stats.multivariate_normal.rvs([0, 0], [[1.0, ro], [ro, 1.0]], size=size)

def mixMultivariateNormal(size, ro):
    return 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + 0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)

#lab5

def quadrantCoeff(x, y):
    med_x = np.median(x)
    med_y = np.median(y)
    n1 = np.array([x >= med_x and y >= med_y for x, y in zip(x, y)]).sum()
    n2 = np.array([x < med_x and y >= med_y for x, y in zip(x, y)]).sum()
    n3 = np.array([x < med_x and y < med_y for x, y in zip(x, y)]).sum()
    n4 = np.array([x >= med_x and y < med_y for x, y in zip(x, y)]).sum()
    return ((n1 + n3) - (n2 + n4)) / len(x)

def coeffCount(get_sample, size, ro, repeats):
    pearson, quadrant, spirman = [], [], []
    for i in range(repeats):
        sample = get_sample(size, ro)
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spirman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrantCoeff(x, y))
    return pearson, spirman, quadrant

def createTable(pearson, spirman, quadrant, size, ro, repeats):
    if ro != -1:
        rows = [["rho = " + str(ro), 'r', 'r_{S}', 'r_{Q}']]
    else:
        rows = [["size = " + str(size), 'r', 'r_{S}', 'r_{Q}']]
    p = np.median(pearson)
    s = np.median(spirman)
    q = np.median(quadrant)
    rows.append(['E(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = np.median([pearson[k] ** 2 for k in range(repeats)])
    s = np.median([spirman[k] ** 2 for k in range(repeats)])
    q = np.median([quadrant[k] ** 2 for k in range(repeats)])
    rows.append(['E(z^2)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = statistics.variance(pearson)
    s = statistics.variance(spirman)
    q = statistics.variance(quadrant)
    rows.append(['D(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    return tabulate(rows, [], tablefmt="latex")

def createEllipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    radX = np.sqrt(1 + pearson)
    radY = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=radX * 2, height=radY * 2, facecolor='none', **kwargs)

    scaleX = np.sqrt(cov[0, 0]) * n_std
    meanX = np.mean(x)

    scaleY = np.sqrt(cov[1, 1]) * n_std
    meanY = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scaleX, scaleY).translate(meanX, meanY)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def printEllipse(size, ros):
    fig, ax = plt.subplots(1, 3)
    strSize = "n = " + str(size)
    titles = [strSize + r', $ \rho = 0$', strSize + r', $\rho = 0.5 $', strSize + r', $ \rho = 0.9$']
    for i in range(len(ros)):
        num, ro = i, ros[i]
        sample = multivariateNormal(size, ro)
        x, y = sample[:, 0], sample[:, 1]
        createEllipse(x, y, ax[num], edgecolor='navy')
        ax[num].grid()
        ax[num].scatter(x, y, s=5)
        ax[num].set_title(titles[num])
    # plt.show()
    fig.savefig('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab5/' + 'fig5_' + str(size) + '.png')

def task5():
    sizes = [20, 60, 100]
    ros = [0, 0.5, 0.9]
    REPETITIONS = 1000

    for s in sizes: 
        for ro in ros:
            pearson, spirman, quadrant = coeffCount(multivariateNormal, s, ro, REPETITIONS)
            with open('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab5/' + 'res5_' + str(s)+'_p_'+str(ro) + '.tex', "w") as file:
                file.write('\n' + str(s) + '\n' + str(createTable(pearson, spirman, quadrant, s, ro, REPETITIONS)))
        pearson, spearman, quadrant = coeffCount(mixMultivariateNormal, s, 0, REPETITIONS)
        with open('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab5/' + 'res5Mix_' + str(s) + '.tex', "w") as file:
            file.write('\n' + str(s) + '\n' + str(createTable(pearson, spirman, quadrant, s, -1, REPETITIONS)))
        printEllipse(s, ros)
    return

#lab6

def func(x):
    return 2 + 2 * x

def noiseFunc(x):
    y = []
    for i in x:
        y.append(func(i) + stats.norm.rvs(0, 1))
    return y

def LMM(parameters, x, y):
    alpha_0, alpha_1 = parameters
    sum = 0
    for i in range(len(x)):
        sum += abs(y[i] - alpha_0 - alpha_1*x[i])
    return sum 

def getMNKParams(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1

def getMNMParams(x, y):
    beta_0, beta_1 = getMNKParams(x, y)
    result = opt.minimize(LMM, [beta_0, beta_1], args=(x, y), method='SLSQP')
    coefs = result.x
    alpha_0, alpha_1 = coefs[0], coefs[1]
    return alpha_0, alpha_1

def MNK(x, y, file):
    beta_0, beta_1 = getMNKParams(x, y)
    file.write(' beta_0 = ' + str(beta_0) + ' beta_1 = ' + str(beta_1))
    y_new = [beta_0 + beta_1 * x_ for x_ in x]
    return y_new

def MNM(x, y, file):
    alpha_0, alpha_1 = getMNMParams(x, y)
    file.write(' alpha_0= ' + str(alpha_0) + ' alpha_1 = ' + str(alpha_1))
    y_new = [alpha_0 + alpha_1 * x_ for x_ in x]
    return y_new

def getDist(y_model, y_regr):
    dist_y = sum([(y_model[i] - y_regr[i])**2 for i in range(0,len(y_model))])
    return dist_y

def getDist2(y_model, y_regr):
    dist_y = sum([(abs(y_model[i] - y_regr[i])) for i in range(0,len(y_model))])
    return dist_y

def plotLiRegression(text, x, y, file):
    fig, ax = plt.subplots(1, 1)
    y_mnk = MNK(x, y, file)
    y_mnm = MNM(x, y, file)
    y_dist_mnk = getDist(y, y_mnk)
    y_dist_mnm = getDist2(y, y_mnm)
    file.write(' mnk distance= ' +  str(y_dist_mnk))
    file.write(' mnm distance= ' + str(y_dist_mnm))
    plt.scatter(x, y, label='Выборка', color='black', marker = ".", linewidths = 0.7)
    plt.plot(x, func(x),  label='Модель', color='blue')
    plt.plot(x, y_mnk, label="МНК", color='red')
    plt.plot(x, y_mnm, label="МНМ", color='cyan')
    plt.xlim([-1.8, 2])
    plt.grid()
    plt.legend()
    # plt.show()
    fig.savefig('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab6/' + 'fig6_' +text + '.png')
    

def task6():
    file = open('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab6/' + 'var_6' + '.tex', "w")
    x = np.arange(-1.8, 2, 0.2)
    y = noiseFunc(x)
    plotLiRegression('NoPerturbations', x, y,file)

    x = np.arange(-1.8, 2, 0.2)
    y = noiseFunc(x)
    y[0] += 10
    y[-1] -= 10
    plotLiRegression('Perturbations', x, y,file)
    file.close()
    return

#lab7  

def getK(size):
    return math.ceil(1.72 * (size) ** (1/3))

def calculate(distribution, p, k, file1):
    mu = np.mean(distribution)
    sigma = np.std(distribution)

    file1.write(' mu = ' + str(np.around(mu, decimals=2)))
    file1.write(' sigma = ' + str(np.around(sigma, decimals=2)))
    
    limits = np.linspace(-1.1, 1.1, num=k - 1)
    chi_2 = stats.chi2.ppf(p, k-1)
    file1.write(' chi_2 = ' + str(chi_2))
    return limits

def getN_P(distribution, limits, size):
    p_list = np.array([])
    n_list = np.array([])
    
    for i in range(-1, len(limits)):
        if i != -1:
            prev_cdf_val = stats.norm.cdf(limits[i])
        else:
            prev_cdf_val = 0
        if i != len(limits) - 1:
            cur_cdf_val = stats.norm.cdf(limits[i+1])
        else: 
            cur_cdf_val = 1 
        p_list = np.append(p_list, cur_cdf_val - prev_cdf_val)
        if i == -1:
            n_list = np.append(n_list, len(distribution[distribution <= limits[0]]))
        elif i == len(limits) - 1:
            n_list = np.append(n_list, len(distribution[distribution >= limits[-1]]))
        else:
            n_list = np.append(n_list, len(distribution[(distribution <= limits[i + 1]) & (distribution >= limits[i])]))

    result = np.divide(np.multiply((n_list - size * p_list), (n_list - size * p_list)), p_list * size)
    return n_list, p_list, result

def createTable1(n_list, p_list, result, size, limits,name):
    cols = ["i", "limits", "n_i", "p_i", "np_i", "n_i - np_i", "/frac{(n_i-np_i)^2}{np_i}"]
    rows = []
    for i in range(0, len(n_list)):
        if i == 0:
            boarders = ['-inf', (np.around(limits[0], decimals=2))]
        elif i == len(n_list) - 1:
            boarders = [np.around(limits[-1], decimals=2), 'inf']
        else:
            boarders = [np.around(limits[i - 1], decimals=2), np.around(limits[i], decimals=2)]

        rows.append([i + 1, boarders, n_list[i], np.around(p_list[i], decimals=4), np.around(p_list[i] * size, decimals=2),
                 np.around(n_list[i] - size * p_list[i], decimals=2), np.around(result[i], decimals=2)])

    rows.append([len(n_list), "-", np.sum(n_list), np.around(np.sum(p_list), decimals=4),
             np.around(np.sum(p_list * size), decimals=2),
             -np.around(np.sum(n_list - size * p_list), decimals=2),
             np.around(np.sum(result), decimals=2)])
    with open('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab7/' + 'res7_'+ name + str(size) + '.tex', "w") as file:
            file.write(tabulate(rows, cols, tablefmt="latex"))

def solver(size, distribution, p, alpha, name, file1):
    k = getK(size)
    limits = calculate(distribution, p, k, file1)
    n_list, p_list, result = getN_P(distribution, limits, size)
    createTable1(n_list, p_list, result, size, limits, name)
    return

def task7():
    file1 = open('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab7/' + 'var' + '.tex', "w")
    sizes = [20, 100]
    alpha = 0.05
    p = 1 - alpha
    #normal
    solver(sizes[1], np.random.normal(0, 1, size=sizes[1]), p, alpha,'Normal', file1)
    #laplace
    solver(sizes[0], stats.laplace.rvs(size=sizes[0], scale=1 / math.sqrt(2), loc=0), p, alpha,'Laplace', file1)
    #uniform
    solver(sizes[0], stats.uniform.rvs(size=sizes[0], loc=-math.sqrt(3), scale=2 * math.sqrt(3)), p, alpha,'Uniform', file1)
    file1.close()
    return


#lab8

def mean(data):
    return np.mean(data)

def dispersion_exp(sample):
    return mean(list(map(lambda x: x*x, sample))) \
           - (mean(sample))**2

def normal(size):
    return np.random.standard_normal(size=size)


def printRes(x_set : list, m_all : float, s_all : list, name, file1):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig.set_figheight(8)
    fig.set_figwidth(26)
    m_left20 = [m_all[0][0], m_all[0][0]]
    m_right20 = [m_all[0][1], m_all[0][1]]
    m_left100 = [m_all[1][0], m_all[1][0]]
    m_right100 = [m_all[1][1], m_all[1][1]]
    
    d_left20 = [m_all[0][0] - s_all[0][1], m_all[0][0] - s_all[0][1]]
    d_right20 = [m_all[0][1] + s_all[0][1], m_all[0][1] + s_all[0][1]]
    d_left100 = [m_all[1][0] - s_all[1][1], m_all[1][0] - s_all[1][1]]
    d_right100 = [m_all[1][1] + s_all[1][1], m_all[1][1] + s_all[1][1]]

    # draw hystograms
    #for hyst=20
    ax1.set_ylim(0, 1.4)
    ax1.hist(x_set[0], density=True, histtype='stepfilled', alpha=0.2, label='N(0, 1) hyst n=20', color = "red")
    ax1.legend(loc='best', frameon=True)
    ax1.plot(m_left20, [0, 0.8], 'mo-', label='min_mu, max_mu')
    ax1.plot(m_right20, [0, 0.8], 'mo-')
    ax1.plot(d_left20, [0, 0.8], 'bo-', label='min_mu—max_sigma, max_mu+max_sigma')
    ax1.plot(d_right20, [0, 0.8], 'bo-')
    ax1.legend()

    file1.write(" m twin20: %.2f, %.2f" % (m_all[0][0], m_all[0][1]))
    file1.write(" m twin100: %.2f, %.2f" % (m_all[1][0], m_all[1][1]))
    file1.write(" sigma twin20: %.2f, %.2f" % (d_left20[0], d_right20[0]))
    file1.write(" sigma twin100: %.2f, %.2f" % (d_left100[0], d_right100[0]))
    
    #for 100
    ax2.set_ylim(0, 1.4)
    ax2.hist(x_set[1], density=True, histtype='stepfilled', alpha=0.2, label='N(0, 1) hyst n=100', color = "red")
    ax2.legend(loc='best', frameon=True)
    ax2.plot(m_left100, [0, 0.8], 'mo-', label='min_mu, max_mu')
    ax2.plot(m_right100, [0, 0.8], 'mo-')
    ax2.plot(d_left100, [0, 0.8], 'bo-', label='min_mu—max_sigma, max_mu+max_sigma')
    ax2.plot(d_right100, [0, 0.8], 'bo-')
    ax2.legend()

    # draw intervals of m
    ax3.set_ylim(0.9, 1.4)
    ax3.plot(m_all[0], [1, 1], 'mo-', label='sigma interval n = 20')
    ax3.plot(m_all[1], [1.1, 1.1], 'bo-', label='sigma interval n = 100')
    ax3.legend()
    
    # draw intervals of sigma
    ax4.set_ylim(0.9, 1.4)
    ax4.plot(s_all[0], [1, 1], 'mo-', label='sigma interval n = 20')
    ax4.plot(s_all[1], [1.1, 1.1], 'bo-', label='sigma interval n = 100')
    ax4.legend()
    
    plt.rcParams["figure.figsize"] = (25,5)
    # plt.show()
    fig.savefig('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab8/' + 'fig8_'+ name + '.png')
    return
def solve (x_set : list, n_set : list, file1):
    alpha = 0.05
    m_all = list()
    s_all = list()
    text = "normal"
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]
        
        m = mean(x)
        s = np.sqrt(dispersion_exp(x))
        
        m1 = [m - s*(stats.t.ppf(1 - alpha/2, n-1))/np.sqrt(n-1), m + s*(stats.t.ppf(1 - alpha/2, n-1))/np.sqrt(n-1)]
        s1 = [s*np.sqrt(n)/np.sqrt(stats.chi2.ppf(1 - alpha/2, n-1)), s*np.sqrt(n)/np.sqrt(stats.chi2.ppf(alpha/2, n-1))]
        
        m_all.append(m1)
        s_all.append(s1)
        
        file1.write(" t: %i" % (n))
        file1.write(" m: %.2f, %.2f" % (m1[0], m1[1]))
        file1.write(" sigma: %.2f, %2.f" % (s1[0], s1[1]))
        
    printRes(x_set, m_all, s_all,'normal', file1)
    return
    
def solveAsymp(x_set : list, n_set : list, file1):
    alpha = 0.05
    m_all = list()
    s_all = list()
    text = "asymp"
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = mean(x)
        s = np.sqrt(dispersion_exp(x))

        m_as = [m - stats.norm.ppf(1-alpha / 2)/np.sqrt(n), m + stats.norm.ppf(1 - alpha / 2)/np.sqrt(n)]
        e = (sum(list(map(lambda el: (el-m)**4, x)))/n)/s**4 - 3
        s_as = [s/np.sqrt(1+stats.norm.ppf(1-alpha / 2)*np.sqrt((e+2)/n)), s/np.sqrt(1-stats.norm.ppf(1-alpha / 2)*np.sqrt((e+2)/n))]

        m_all.append(m_as)
        s_all.append(s_as)

        file1.write(" m asymptotic :%.2f, %.2f" % (m_as[0], m_as[1]))
        file1.write(" sigma asymptotic: %.2f, %.2f" % (s_as[0], s_as[1]))
    printRes(x_set, m_all, s_all,'asymp' , file1)
    return

def task8():
    file1 = open('C:/Users/nikol/OneDrive/Рабочий стол/labMatStat/Lab2/resLab8/' + 'var' + '.tex', "w")
    n_set = [20, 100]
    x_20 = normal(20)
    x_100 = normal(100)
    x_set = [x_20, x_100]
    solve(x_set, n_set, file1)
    solveAsymp(x_set, n_set, file1)
    file1.close()
    return

def main():
    # task5()
    task6()
    task7()
    task8()

if __name__ == "__main__":
    main()