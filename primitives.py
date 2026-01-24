import brain
import copy
import numpy as np
import utils

def dotSegment(dst: np.ndarray, begin: tuple[int, int], end: tuple[int, int], value: int, dot_step: int) -> np.ndarray:
    """
    Dot a straight line between two points in an numpy array with the specified value.
    """
    m = copy.deepcopy(dst)

    u = np.array([end[0], end[1]]) - np.array([begin[0], begin[1]])
    v = u / np.linalg.norm(u)

    if (min(v)):
        v *= min(v)

    step = np.linalg.norm(v) / np.linalg.norm(u)
    i = 0

    for t in np.arange(0, 1 + 0.1 * step, step):
        if (i % dot_step == 0):
            m[round(begin[0] + t * u[0]), round(begin[1] + t * u[1])] = value

        i += 1

    return m

def segments(dst: np.ndarray, pairs: list[tuple[tuple[int, int], tuple[int, int]]], value: int, start: bool, finish: bool) -> np.ndarray:
    """
    Dot straight lines between a list of two points in an numpy array with the specified value considering coloring or not start and finish points
    """
    m = copy.deepcopy(dst)

    for pair in pairs:
        if (utils.validIndex(m, pair[0]) and utils.validIndex(m, pair[1])):
            s = m[pair[0][0]][pair[0][1]]
            f = m[pair[1][0]][pair[1][1]]

            m = dotSegment(m, pair[0], pair[1], value, 1)

            if (start):
                m[pair[0][0]][pair[0][1]] = s

            if (finish):
                m[pair[1][0]][pair[1][1]] = f

    return m

def map(x: list[np.ndarray], mapping: dict) -> list[np.ndarray]:
    """
    Return numpy arrays which cells ared mapped from a dictionnary
    """
    
    result = []
    
    for v in x:
        y = copy.deepcopy(v)

        for i in range(0, y.shape[0]):
            for j in range(0, y.shape[1]):
                y[i, j] = mapping.get(y[i, j], y[i, j])

        result.append(y)

    return result

def associate(mapping: dict, a: int, b: int) -> dict:
    """
    Associate a value with another value in a dictionnary
    """
    m = copy.deepcopy(mapping)

    m[a] = b

    return m

def inferColorMapping(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    Infer a color mapping from the given pairs of numpy arrays as a dictionnary
    """
    mapping = {}

    for pair in pairs:
        h, w = pair[0].shape

        for i in range(h):
            for j in range(w):
                a = int(pair[0][i, j])
                b = int(pair[1][i, j])

                if (a in mapping):
                    if (mapping[a] != b):
                        return mapping
                else:
                    mapping[a] = b

    return mapping

def sameElement(pairs: list[tuple[tuple[int, int], tuple[int, int]]], first: bool) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Return a list of index pairs in an numpy array where the elements are the same considering first or second element of a pair
    """
    
    results = []

    for p in pairs:
        if (first):
            if (p[0][0] == p[1][0]):
                results.append(p)
        else:
            if (p[0][1] == p[1][1]):
                results.append(p)

    return results

def regionPairs(regions: list[list[tuple[int, int]]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Return a list of paired regions from a list of regions
    """
    
    pairs = []

    for i1 in range(0, len(regions)):
        for j1 in range(i1 + 1, len(regions)):
            for i2 in range(0, len(regions[i1])):
                for j2 in range(0, len(regions[j1])):
                    pairs.append((regions[i1][i2], regions[j1][j2]))

    return pairs

def fliplr(x: list[np.ndarray]) -> list[np.ndarray]:
    """
    Flip left/right a list of numpy array
    """
    
    return [np.fliplr(y) for y in x]

def flipud(x: list[np.ndarray]) -> list[np.ndarray]:
    """
    Flip up/down a list of numpy array
    """
    
    return [np.flipud(y) for y in x]
