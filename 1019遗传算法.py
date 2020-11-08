#调整颜色显示：coolwarm---改为--PiYG(matplotlib.cm是颜色模块)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm          #matplotlib.cm是matplotlib库中内置的色彩映射函数。
from mpl_toolkits.mplot3d import Axes3D     #是Matplotlib里面专门用来画三维图的工具包

DNA_SIZE = 24       #编码长度
POP_SIZE = 200      #POP表示种群，该种群由200个个体组成。
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 50
X_BOUND = [-3, 3]   #X限制在-3~3范围
Y_BOUND = [-3, 3]


def F(x, y):                          #定义一个二元函数，本算法利用遗传算法解决该函数的最优化问题
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)


def plot_3d(ax):
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.PiYG)      #cm.PiYG调用彩色映射
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()

#适应度评估
def get_fitness(pop):
    x, y = translateDNA(pop)  #解码后的x,y；均为-3~3内的实数；
    pred = F(x, y)            #将x,y带入函数F计算结果赋值给pred
    return (pred - np.min(pred))  + 1e-3# 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]
'''
pred是将可能解带入函数F中得到的预测值，因为后面的选择过程需要根据个体适应度确定每个个体被保留下来的概率，而概率不能是负值，
所以减去预测中的最小值把适应度值的最小区间提升到从0开始，但是如果适应度为0，其对应的概率也为0，表示该个体不可能在选择中保留
下来，这不符合算法思想，遗传算法不绝对否定谁也不绝对肯定谁，所以最后加上了一个很小的正数。
有了求最大值的适应度函数求最小值适应度函数也就容易了，python代码如下:'''
# def get_fitness(pop):
# 	x,y = translateDNA(pop)
# 	pred = F(x, y)
# 	return -(pred - np.max(pred)) + 1e-3



#解码过程
'''
DNA_SIZE = 24(一个实数DNA长度)，两个实数x,y一共用48位二进制编码，同时将x,y编码到同一个48位的二进制串里，奇数列为x的编码表示，偶数列为y的编码表示。
解码后x,y均为-3~3内的实数
'''
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 1::2]  # 偶数列表示X，其中pop[:, 1::2],第一个逗号前的：表示行遍历0~最后即全行；逗号后1：：2表示列遍历下标1开始到最后，步长为2
    y_pop = pop[:, ::2]  # 奇数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y

#交叉、变异
def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲;pop[随机数]--选取pop的第’随机数‘行
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE*2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转，^为按位异或运算符：当两对应的二进位相异时，结果为1

#选择：适者生存----适应度越高，被选择的机会越高，而适应度低的，被选择的机会就低
'''
有了评估的适应度函数，下面可以根据适者生存法则将优秀者保留下来了。选择则是根据新个体的适应度进行，
但同时不意味着完全以适应度高低为导向（选择top k个适应度最高的个体，容易陷入局部最优解），因为单纯
选择适应度高的个体将可能导致算法快速收敛到局部最优解而非全局最优解，我们称之为早熟。作为折中，遗传
算法依据原则：适应度越高，被选择的机会越高，而适应度低的，被选择的机会就低。
'''
def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax)

    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    '''
    DNA_SIZE = 24  POP_SIZE = 200   因此上式生成一个200行24*2列的随机矩阵（矩阵元素为0或1）
    '''
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o');
        plt.show();
        plt.pause(0.1)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        fitness = get_fitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群

    print_info(pop)
    plt.ioff()
    plot_3d(ax)

