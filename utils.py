import numpy as np

def regionSet(m: np.ndarray, diagonals: bool) -> set:
    s = set()

    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            r = tuple(sorted(region(m, (i, j), diagonals)))

            if (not r in s):
                s.add(r)

    return s

def validIndex(a: np.ndarray, at: tuple[int, int]) -> bool:
    """
    Check if the index is within the bounds of a numpy array
    """
    
    return (0 <= at[0] and at[0] < a.shape[0] and 0 <= at[1] and at[1] < a.shape[1])

def neighbors(loc: tuple[int, int], size: tuple[int, int], diagonals: bool) -> list[tuple[int, int]]:
    """
    Return a list of the neighboring indices of a given index considering diagonal neihbors or not
    """
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
    """
    Return a list of indices in the same region as the given index considering diagonal neihbors or not
    """
    
    if (not validIndex(a, at)):
        return []

    s = set()
    stack = set()
    stack.add(at)

    v = a[at[0]][at[1]]

    indices = []

    while (len(stack)):
        l = list(stack)
        loc = l[0]
        del l[0]
        stack = set(l)

        if (not loc in s):
            s.add(loc)

            if (abs(a[loc[0]][loc[1]] - v) < np.finfo(float).eps):
                indices.append(loc)

                for n in neighbors(loc, (a.shape[0], a.shape[1]), diagonals):
                    stack.add(n)

    return indices
