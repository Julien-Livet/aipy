import brain
import copy
import numpy as np

def validIndex(a: np.ndarray, at: tuple[int, int]) -> bool:
    return (0 <= at[0] and at[0] < a.shape[0] and 0 <= at[1] and at[1] < a.shape[1])

def neighbors(loc: tuple[int, int], size: tuple[int, int], diagonals: bool) -> list[tuple[int, int]]:
    n = []

    for di in range(-1, 2):
        for dj in range(-1, 2):
            if (di == 0 and dj == 0):
                continue

            i = loc[0] + di
            j = loc[1] + dj

            if (not (0 <= i and i < size[0] and 0 <= j and j < size[1])):
                continue

            if (not diagonals and abs(di) == abs(dj)):
                continue

            n.append((i, j))

    return n

def region(a: np.ndarray, at: tuple[int, int], diagonals: bool) -> list[tuple[int, int]]:
    if (not validIndex(a, at)):
        return []

    s = brain.Set()
    stack = brain.Set()
    stack.add(at)

    v = a[at[0]][at[1]]

    indices = []

    while (len(stack)):
        loc = stack[0]
        stack.erase(loc)

        if (not loc in s):
            s.add(loc)

            if (abs(a[loc[0]][loc[1]] - v) < brain.eps):
                indices.append(loc)

                for n in neighbors(loc, (a.shape[0], a.shape[1]), diagonals):
                    stack.add(n)

    return indices

def dotSegment(dst: np.ndarray, begin: tuple[int, int], end: tuple[int, int], value: int, dot_step: int) -> np.ndarray:
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
    m = copy.deepcopy(dst)

    for pair in pairs:
        if (validIndex(m, pair[0]) and validIndex(m, pair[1])):
            s = m[pair[0][0]][pair[0][1]]
            f = m[pair[1][0]][pair[1][1]]

            m = dotSegment(m, pair[0], pair[1], value, 1)

            if (start):
                m[pair[0][0]][pair[0][1]] = s

            if (finish):
                m[pair[1][0]][pair[1][1]] = f

    return m

def map(x: np.ndarray, mapping: dict):
    y = copy.deepcopy(x)

    for i in range(0, y.shape[0]):
        for j in range(0, y.shape[1]):
            y[i, j] = mapping.get(y[i, j], y[i, j])

    return y

def associate(mapping: dict, a: int, b: int):
    m = copy.deepcopy(mapping)

    m[a] = b

    return m

def inferColorMapping(pairs: list[tuple[np.ndarray, np.ndarray]]):
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
    pairs = []

    for i1 in range(0, len(regions)):
        for j1 in range(i1 + 1, len(regions)):
            for i2 in range(0, len(regions[i1])):
                for j2 in range(0, len(regions[j1])):
                    pairs.append((regions[i1][i2], regions[j1][j2]))

    return pairs

