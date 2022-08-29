import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

def label(vocab, vec):
    l = [w for w in vec if w in vocab]
    return np.array([vocab[v] for v in l]), l

def color(color, list):
    return [color] * len(list)

def bias_diff_vector(vocab, def_sets, key_pair):
    w1 = []
    w2 = []
    done = []

    for k in def_sets:
        p = def_sets.get(k)
        for i, w in enumerate(p):

            if w not in done:
                try:
                    if i == 0:
                        w1.append(vocab[w])
                        done.append(w)
                    elif i == 1:
                        w2.append(vocab[w])
                        done.append(w)
                except KeyError as e:
                    pass

    set_vectors = np.array(np.average(w1, axis=0) - np.average(w2, axis=0))
    return set_vectors.reshape(1, -1)

# Axis are 1st and 2nd eigenvalue
def visualize(data, vocab, color, title):
    path = title.replace("-", "").replace(" ", "_").replace("/", "_").lower()

    # Computing the correlation matrix
    df = pd.DataFrame(data)
    X_corr = df.corr()

    # Computing eigen values and eigen vectors
    values, vectors = np.linalg.eig(X_corr)

    # Sorting the eigen vectors coresponding to eigen values in descending order
    args = (-values).argsort()
    values = vectors[args]
    vectors = vectors[:, args]

    # Taking first 2 components which explain maximum variance for projecting
    new_vectors = vectors[:, :2]
    # Projecting it onto new dimesion with 2 axis
    neww_X = np.dot(data, new_vectors)
    custom_lines = [Line2D([0], [0], color="yellow", lw=4),
                    Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="m", lw=4),
                    Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="black", lw=4),
                    Line2D([0], [0], color="red", lw=4)]
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    ax.legend(custom_lines,
              ['African Female', 'African Male', 'European Female', 'European Male', 'Random', 'Target Vector'])

    for i in range(len(vocab)):
        plt.scatter(neww_X[i, 0], neww_X[i, 1], linewidths=1, color=color[i])
    plt.xlabel("PC1", size=15)
    plt.ylabel("PC2", size=15)

    # plt.xlim([-45, 40])
    # plt.ylim([-45, 40])
    plt.title(title, size=20)

    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(neww_X[i, 0], neww_X[i, 1]))

    plt.savefig('plots/' + path + '.png')

    plt.show()

# Customize axis
def new_visualize(data, x, y, vocab, color, title):
    path = title.replace("-", "").replace(" ", "_").replace("/", "_").lower()
    #Computing the correlation matrix
    df = pd.DataFrame(data)
    axis = np.concatenate((x, y)).T
    #Projecting it onto new dimesion with 2 axis
    neww_X = np.dot(df, axis)

    custom_lines = [Line2D([0], [0], color="yellow", lw=4),
                    Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="m", lw=4),
                    Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="black", lw=4),
                    Line2D([0], [0], color="red", lw=4)]

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    ax.legend(custom_lines, ['African Female', 'African Male', 'European Female', 'European Male', 'Random', 'Target Vector'])

    for i in range(len(vocab)):
        plt.scatter(neww_X[i, 0], neww_X[i, 1], linewidths=1, color=color[i])
    plt.xlabel("man-woman", size=15)
    plt.ylabel("black-white", size=15)
    plt.axvline(x=0)
    plt.axhline(y=0)
    # plt.xlim([-45, 40])
    # plt.ylim([-45, 40])
    plt.title(title, size=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(neww_X[i, 0], neww_X[i, 1]))

    plt.savefig('plots/' + path + '.png')

    plt.show()

# def pca_analysis():

def newnew_visualize(data, x, y, vocab, color, title):
    path = title.replace("-", "").replace(" ", "_").replace("/", "_").lower()
    #Computing the correlation matrix
    df = pd.DataFrame(data)
    axis = np.concatenate((x, y)).T
    #Projecting it onto new dimesion with 2 axis
    neww_X = np.dot(df, axis)

    custom_lines = [Line2D([0], [0], color="yellow", lw=4),
                    Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="m", lw=4),
                    Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="black", lw=4),
                    Line2D([0], [0], color="red", lw=4)]

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    ax.legend(custom_lines, ['African Female', 'African Male', 'European Female', 'European Male', 'Random', 'Target Vector'])

    for i in range(len(vocab)):
        plt.scatter(neww_X[i, 0], neww_X[i, 1], linewidths=1, color=color[i])
    plt.xlabel("man-woman", size=15)
    plt.ylabel("black-white", size=15)
    plt.axvline(x=0)
    plt.axhline(y=0)
    # plt.xlim([-45, 40])
    # plt.ylim([-45, 40])
    plt.title(title, size=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(neww_X[i, 0], neww_X[i, 1]))

    plt.savefig('plots/' + path + '.png')

    plt.show()

def pca_visualize(data, color, title):

    x = np.array([1, 2, 3, 4, 5])
    y_inter = np.array([0.8913592376, 0.9357647056, 0.9849266002, 0.8960669404, 0.8973288227])
    y_sum = np.array([0.8387024311, 0.8488860144, 0.8493966292, 0.8824599757, 0.9100571199])
    y_mean = np.array([0.8836582668, 0.8968833511, 0.8807107411, 0.8787855276, 0.8790999953])

    # path = title.replace("-", "").replace(" ", "_").replace("/", "_").lower()

    custom_lines = [Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="green", lw=4)]

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    ax.legend(custom_lines,
              ['Polysemy', 'Sum', 'Mean'])

    plt.plot(x, y_inter, 'ro:', x, y_sum, 'bo:', x, y_mean, 'go:')
    plt.xlabel("PCA Rank", size=15)
    plt.ylabel("MAC score", size=15)
    # plt.xlim([-45, 40])
    # plt.ylim([-45, 40])
    plt.title("Intersection evalset / Intersection subspace", size=20)


    # plt.savefig('plots/pca/' + path + '.png')

    plt.show()

# pca_visualize(None, None, None)

