import brain
from connection import Connection
import copy
import json
from neuron import Neuron
import numpy as np
import primitives
import pydot
import pytest
import time
import urllib.request
import utils

@pytest.fixture(autouse = True)
def print_test_duration(request):
    start_time = time.time()
    yield
    duration = time.time() - start_time
    print(f"\nTest {request.node.name} took {duration:.2f} seconds to execute.")

digitNeurons = []

for i in range(0, 10):
    digitNeurons.append(Neuron(str(i), lambda i = i: i, [], int))

addNeuron = Neuron("add", lambda x, y: x + y, [int, int], int)
mulNeuron = Neuron("mul", lambda x, y: x * y, [int, int], int)
intToStrNeuron = Neuron("intToStr", lambda x: str(x), [int], str)

def test1():
    conn0 = Connection(digitNeurons[0], [])

    assert(conn0.toStr() == "0")
    assert(conn0.output() == 0)
    assert(conn0.depth() == 0)
    assert(conn0.cost() == 0)

    conn1 = Connection(addNeuron, [2, 3])

    assert(conn1.toStr() == "add(2, 3)")
    assert(conn1.output() == 5)
    assert(conn1.depth() == 0)
    assert(conn1.cost() == 2)

    conn2 = Connection(mulNeuron, [conn1, 4])

    assert(conn2.toStr() == "mul(add(2, 3), 4)")
    assert(conn2.output() == 20)
    assert(conn2.depth() == 1)
    assert(conn2.cost() == 4)

    assert(conn2.inputTypes() == [int, int, int])
    assert(conn2.output((3, 5, 4)) == 32)

def test2():
    c1 = Connection(addNeuron, [1, 2])
    c2 = Connection(addNeuron, [3, 4])
    c3 = Connection(mulNeuron, [c1, c2])
    c4 = Connection(addNeuron, [c3, c3])

    assert(c4.output() == (1 + 2) * (3 + 4) + (1 + 2) * (3 + 4))

    dot = "digraph ConnectionTree {\n"
    dot += "node [shape=circle, style=filled, fillcolor=lightgray];\n"

    index = 0

    s, index = c4.dot(index)
    dot += s

    dot += "}\n"

    (graph,) = pydot.graph_from_dot_data(dot)
    graph.write_png('dot/test2_a.png')

    c4.applyInputs([4, 5, 6, 7, 8, 9, 10, 11])

    dot = "digraph ConnectionTree {\n"
    dot += "node [shape=circle, style=filled, fillcolor=lightgray];\n"

    index = 0

    s, index = c4.dot(index)
    dot += s

    dot += "}\n"

    (graph,) = pydot.graph_from_dot_data(dot)
    graph.write_png('dot/test2_b.png')

    assert(c4.output() == (4 + 5) * (6 + 7) + (8 + 9) * (10 + 11))

def test_dot():
    neurons = []
    neurons += digitNeurons
    neurons.append(addNeuron)
    neurons.append(mulNeuron)
    neurons.append(intToStrNeuron)

    b = brain.Brain(neurons)

    s = b.neuronDot()

    (graph,) = pydot.graph_from_dot_data(s)
    graph.write_png('dot/dot_neurons.png')

    conn1 = Connection(addNeuron, [2, 3])
    conn2 = Connection(mulNeuron, [conn1, 4])

    dot = "digraph ConnectionTree {\n"
    dot += "node [shape=circle, style=filled, fillcolor=lightgray];\n"

    index = 0

    s, index = conn2.dot(index)
    dot += s

    dot += "}\n"

    (graph,) = pydot.graph_from_dot_data(dot)
    graph.write_png('dot/dot_connection.png')

def test_str():
    neurons = []
    neurons += digitNeurons
    neurons.append(addNeuron)
    neurons.append(mulNeuron)
    neurons.append(intToStrNeuron)

    b = brain.Brain(neurons)

    target = "11"

    connections = b.learn([target])

    assert(len(connections))

    print(connections[0].toStr())

    assert(not brain.heuristic(connections[0].output(), target))

def test_expsin():
    neurons = []

    neurons.append(Neuron("add", lambda x, y: x + y, [np.ndarray, np.ndarray], np.ndarray))
    neurons.append(Neuron("mul", lambda x, y: x * y, [np.ndarray, np.ndarray], np.ndarray))
    neurons.append(Neuron("exp", lambda x: np.exp(x), [np.ndarray], np.ndarray))
    neurons.append(Neuron("sin", lambda x: np.sin(x), [np.ndarray], np.ndarray))
    neurons.append(Neuron("id", lambda x: x, [np.ndarray], np.ndarray))

    x = np.random.rand(10)

    neurons.append(Neuron("x", lambda x = x: x, [], np.ndarray))

    b = brain.Brain(neurons)

    s = b.neuronDot()

    (graph,) = pydot.graph_from_dot_data(s)
    graph.write_png('dot/dot_expsin.png')

    connections = b.learn([np.exp(x) * np.sin(x)])

    s = connections[0].toStr()

    assert(s == "mul(exp(x), sin(x))" or s == "mul(sin(x), exp(x))")

def trainTestPairs(folder: str, task: str) -> tuple:
    url = urllib.request.urlopen("https://raw.githubusercontent.com/arcprize/ARC-AGI-2/refs/heads/main/data/" + folder + "/" + task + ".json")
    data = json.loads(url.read().decode())

    train = data["train"]
    trainPairs = []

    for v in train:
        trainPairs.append((np.array(v["input"]), np.array(v["output"])))

    test = data["test"]
    testPairs = []

    for v in test:
        testPairs.append((np.array(v["input"]), np.array(v["output"])))

    return (trainPairs, testPairs)

def inputOutputPairs(pairs):
    inputs = []
    outputs = []

    for p in pairs:
        inputs.append(p[0])
        outputs.append(p[1])

    return (inputs, outputs)

def regionSet(m: np.ndarray, diagonals: bool) -> set:
    s = set()

    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            r = tuple(sorted(utils.region(m, (i, j), diagonals)))

            if (not r in s):
                s.add(r)

    return s

def updateRegionNeurons(regionNeurons: dict, pairs: list[np.ndarray]):
    regionMap = dict()

    for input_ in pairs:
        s = regionSet(input_, False)

        regions = dict()

        for i in range(0, 10):
            regions[i] = []

        for r in s:
            regions[input_[r[0][0], r[0][1]]].append(r)

        for k, v in regions.items():
            l = regionMap.get(k, [])
            l.append(v)
            regionMap[k] = l

    for k, v in regionMap.items():
        function = lambda v = v: v

        emptyCount = 0

        for l in v:
            if (len(l) == 0):
                emptyCount += 1

        if (regionNeurons.get(k, None) == None):
            if (emptyCount != len(v)):
                regionNeurons[k] = Neuron("region" + str(k), function, [], list[list[list[tuple[int, int]]]])
        else:
            if (emptyCount != len(v)):
                regionNeurons[k].function = function
            else:
                del regionNeurons[k]

def processTask(folder: str, task: str, activatedNeuronNames: list[str]):
    taskPairs = trainTestPairs(folder, task)
    trainPairs = inputOutputPairs(taskPairs[0])
    testPairs = inputOutputPairs(taskPairs[1])

    falseNeuron = Neuron("False", lambda: False, [], bool)
    trueNeuron = Neuron("True", lambda: True, [], bool)
    fliplrNeuron = Neuron("fliplr", primitives.fliplr, [list[np.ndarray]], list[np.ndarray])
    flipudNeuron = Neuron("flipud", primitives.flipud, [list[np.ndarray]], list[np.ndarray])
    mappingNeuron = Neuron("mapping", lambda: dict(), [], dict())
    associateNeuron = Neuron("associate", utils.associate, [dict, int, int], dict)
    mapNeuron = Neuron("map", primitives.map, [list[np.ndarray], dict], list[np.ndarray])
    inferColorMappingNeuron = Neuron("inferColorMapping", primitives.inferColorMapping, [list[tuple[np.ndarray, np.ndarray]]], dict)
    pairsNeuron = Neuron("pairs", lambda v, taskPairs = taskPairs: [] if len(taskPairs[0]) != len(v) else [(v[i], taskPairs[0][i][1]) for i in range(0, len(taskPairs[0]))],
                         [list[np.ndarray]], list[tuple[np.ndarray, np.ndarray]])
    trainPairsNeuron = Neuron("trainPairs", lambda taskPairs = taskPairs: taskPairs[0], [], list[tuple[np.ndarray, np.ndarray]])
    inputNeuron = Neuron("input", lambda trainPairs = trainPairs: trainPairs[0], [], list[np.ndarray])
    segmentsNeuron = Neuron("segments", primitives.segments, [list[np.ndarray], list[list[tuple[tuple[int, int], tuple[int, int]]]], int, bool, bool], list[np.ndarray])
    sameElementNeuron = Neuron("sameElement", primitives.sameElement, [list[list[tuple[tuple[int, int], tuple[int, int]]]], bool], list[list[tuple[tuple[int, int], tuple[int, int]]]])
    regionPairsNeuron = Neuron("regionPairs", primitives.regionPairs, [list[list[list[tuple[int, int]]]]], list[list[tuple[tuple[int, int], tuple[int, int]]]])

    regionNeurons = dict()
    updateRegionNeurons(regionNeurons, trainPairs[0])

    neurons = []

    if (len(activatedNeuronNames) == 0 or "digitNeurons" in activatedNeuronNames):
        #neurons += digitNeurons

        digits = set()

        for m in trainPairs[0]:
            for v in m.reshape(m.size):
                digits.add(v)

        for m in trainPairs[1]:
            for v in m.reshape(m.size):
                digits.add(v)

        existingDigits = []

        for i in range(0, 10):
            existingDigits.append(i in digits)

        for i in range(0, len(digitNeurons)):
            if (existingDigits[i]):
                neurons.append(digitNeurons[i])

    if (len(activatedNeuronNames) == 0 or "boolNeurons" in activatedNeuronNames):
        neurons.append(falseNeuron)
        neurons.append(trueNeuron)

    if (len(activatedNeuronNames) == 0 or "253bf280" in activatedNeuronNames):
        neurons.append(sameElementNeuron)
        neurons.append(regionPairsNeuron)
        neurons.append(segmentsNeuron)

    if (len(activatedNeuronNames) == 0 or "mapNeurons" in activatedNeuronNames):
        neurons.append(mapNeuron)
        #neurons.append(associateNeuron)
        #neurons.append(mappingNeuron)
        neurons.append(inferColorMappingNeuron)
        neurons.append(pairsNeuron)
        neurons.append(trainPairsNeuron)

    if (len(activatedNeuronNames) == 0 or "flipNeurons" in activatedNeuronNames):
        neurons.append(fliplrNeuron)
        neurons.append(flipudNeuron)

    if (len(activatedNeuronNames) == 0 or "regionNeurons" in activatedNeuronNames):
        for k, v in regionNeurons.items():
            neurons.append(v)

    neurons.append(inputNeuron)

    b = brain.Brain(neurons)

    connections = b.learn([trainPairs[1]], [list[np.ndarray]], level = 3)

    if (len(connections) == 0):
        return -1

    connection = connections[0]

    print(connection.toStr())

    if (brain.heuristic(connection.output(), trainPairs[1])):
        return -2

    updateRegionNeurons(regionNeurons, testPairs[0]);

    inputNeuron.function = lambda testPairs = testPairs: testPairs[0]

    if (brain.heuristic(connection.output(), testPairs[1])):
        return -3

    return 0

def test_task3c9b0459(): #Flip left/right and flip up/down
    assert(processTask("training", "3c9b0459", ["flipNeurons"]) == 0)

def test_task0d3d703e(): #Color mapping
    assert(processTask("training", "0d3d703e", ["digitNeurons", "mapNeurons"]) == 0)

def test_task253bf280(): #Draw colored segment between pixels that have same x or y coordinates
    assert(processTask("training", "253bf280", ["digitNeurons", "boolNeurons", "regionNeurons", "253bf280"]) == 0)

def test_tasks_arc_agi():
    return

    folder = "training"

    url = urllib.request.urlopen("https://raw.githubusercontent.com/arcprize/ARC-AGI-2/refs/heads/main/data/" + folder + ".txt")
    data = url.read().decode()
    tasks = data.split("\n")

    for task in tasks:
        processTask(folder, task)

    folder = "eval"

    url = urllib.request.urlopen("https://raw.githubusercontent.com/arcprize/ARC-AGI-2/refs/heads/main/data/" + folder + ".txt")
    data = url.read().decode()
    tasks = data.split("\n")

    for task in tasks:
        processTask(folder, task)

