import matplotlib.pyplot as plt

# https://matplotlib.org/stable/users/explain/quick_start.html#quick-start

def easy_plot():
    plt.plot([10, 9, 8, 7, 6, 5, 4.5, 4.1, 3.9, 3.8], label='train')
    plt.plot([11, 10, 9, 8, 7, 6, 5.3, 5, 4.7, 4.5], label='val')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('DPEDE')
    plt.title('Train/Val DPEDE')
    plt.show()              # 每次 plt.show() 之后都会清空之前的数据


def plot():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 4, 8, 16, 32, 64, 128], label='pow')
    ax.plot([1, 4, 9, 16, 25, 36, 49, 64], label='quadratic')
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8], label='linear')
    ax.set_title('Title')
    ax.set_ylabel('Index')
    ax.set_xlabel('Iteration')
    ax.legend()             # 图例
    ax.grid()               # 网格
    ax.set_yscale('log')    # 尺度
    plt.show()


def mutli_plot():
    fig, axs = plt.subplots(2, 3)   # 2 x 3
    axs[0][0].plot([1, 2, 3])
    axs[0][0].set_title('pic1')
    axs[0][1].set_title('pic2')
    axs[0][2].set_title('pic3')
    axs[1][0].set_title('pic4')
    axs[1][1].set_title('pic5')
    axs[1, 2].set_title('pic6')     # [row, col] or [row][col]
    plt.show()


if __name__ == '__main__':
    # easy_plot()
    # plot()
    mutli_plot()
