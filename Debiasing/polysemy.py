import numpy as np
from sklearn.decomposition import PCA

# get difference vector of each bias definite sets
def bias_diff_vector(vocab, def_sets, key_pair):
    w1 = []
    w2 = []
    done = []

    for k, v in enumerate(def_sets):
        for i, w in enumerate(v):
            if w in str(key_pair)[1:-1] and w not in done:
                try:
                    if i == 0:
                        w1.append(vocab[w])
                        done.append(w)
                    else:
                        w2.append(vocab[w])
                        done.append(w)
                except KeyError as e:
                    pass

    set_vectors = np.array(np.average(w1, axis=0) - np.average(w2, axis=0))
    return set_vectors.reshape(1, -1)



def vocabCheck(vocab, word):
    if word.lower() in vocab.keys():
        return True
    else:
        return False


def normalizeMatrix(mat):
    matNorm = np.zeros(mat.shape)
    d = (np.sum(mat ** 2, 1) ** (0.5))
    matNorm = (mat.T / d).T

    return matNorm



def getCentVec(contextVecs):
    sample, rank, dim = contextVecs.shape
    contexts = np.reshape(contextVecs, (sample * rank, dim))
    pca = PCA(n_components=1)
    pca.fit(contexts)

    return pca.components_[0]


def polysemy(contextVecs, K, dim, itermax, senseNum):
    minErr = float('inf')
    kmeansIterMax = itermax
    n = senseNum
    for ranIdx in range(5):
        tempSenseVecs = np.random.normal(size=(K, dim))
        tempSenseVecs = normalizeMatrix(tempSenseVecs)
        postErr = 0
        curErr = float('inf')
        iterNum = 0
        while True:
            if iterNum > kmeansIterMax:
                break
            iterNum += 1
            postErr = curErr
            # cluster subspaces
            errSum = 0
            coeffs = np.dot(contextVecs, tempSenseVecs.T)
            d = np.sum(coeffs ** 2, 1)

            errs = 1 - np.max(d, axis=1)
            ## remove numerical negative term
            errs = (errs + np.absolute(errs)) / 2
            errSum = sum(errs)
            labels = np.argmax(d, axis=1)

            # update centers of subspaces
            # logging.info("%f\t%d" % (errSum/sample, iterNum))
            curErr = errSum

            if abs(postErr - curErr) < 1e-5 * 3:
                break
            for s in range(n):
                clusterIdx = np.where(labels == s)
                if len(labels[clusterIdx]) == 0:
                    continue
                tempSenseVecs[s] = getCentVec(contextVecs[clusterIdx])
        if curErr <= minErr:
            minErr = curErr
            senseVecs = tempSenseVecs

    lastErr = minErr
    lastSenseVecs = senseVecs

    if np.isnan(lastSenseVecs).any():
        print("isnan")
    elif np.isinf(lastSenseVecs).any():
        print("isinf")
    else:
        print("None")

    return lastSenseVecs


def label(vocab, vec):
    l = [w for w in vec if w in vocab]
    return np.array([vocab[v]/np.linalg.norm(vocab[v]) for v in l]), l
