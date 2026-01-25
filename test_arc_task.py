import brain
from connection import Connection
import copy
import json
import math
from neuron import Neuron
import numpy as np
import pytest
import subprocess
import time
import urllib.request
import utils

@pytest.fixture(autouse = True)
def print_test_duration(request):
    start_time = time.time()
    yield
    duration = time.time() - start_time
    print(f"\nTest {request.node.name} took {duration:.2f} seconds to execute.")

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

def bestPrimitives(folder: str, task: str, connectionStr: str, cost: float) -> tuple[list[str], list[str]]:
    url = urllib.request.urlopen("https://raw.githubusercontent.com/arcprize/ARC-AGI-2/refs/heads/main/data/" + folder + "/" + task + ".json")
    data = json.loads(url.read().decode())

    command = "Here is an ARC AGI task.\n"
    command += json.dumps(data["train"]) + "\n"
    command += "Describe the transformation performed in ONE abstract sentence (ignoring the exact dimensions, coordinates and numbers).\n"
    command += "Do not mention any Python functions."

    modelName = "gemma3"

    cmd = ["ollama", "run", "gemma3", command]
    result = subprocess.run(cmd, capture_output = True, text = True)
    firstSentence = result.stdout.replace("\n", "")
    cmd = ["ollama", "run", "gemma3:27b", command]
    result = subprocess.run(cmd, capture_output = True, text = True)
    secondSentence = result.stdout.replace("\n", "")

    command = "Here is an ARC AGI task.\n"
    command += json.dumps(data["train"]) + "\n"
    command += "A. " + firstSentence + "\n"
    command += "B. " + secondSentence + "\n"
    command += "Without any explanation, give me only the letter that best matches the description of the given task from the two previous proposals."
    cmd = ["ollama", "run", "gemma3:27b", command]
    result = subprocess.run(cmd, capture_output = True, text = True)
    sentence = firstSentence if result.stdout.strip() == "A" else secondSentence
    #print(sentence)

    file = open("primitives.py")
    content = file.read()
    file.close()

    functions = []

    lines = content.split("\n")

    for line in lines:
        if (line.startswith("def ")):
            functions.append(line[len("def "):line.find("(")])

    import primitives

    scores = {}

    for function in functions:
        cmd = ["python", "-c", "import primitives; help(primitives." + function + ")"]
        result = subprocess.run(cmd, capture_output = True, text = True)

        functionLines = result.stdout.split("\n")

        for i in range(0, 2):
            del functionLines[0]
            del functionLines[-1]

        functionContent = "\n".join(functionLines)

        command = "ARC AGI task description:\n"
        command += sentence + "\n"
        command += "Primitive:\n"
        command += functionContent + "\n"
        command += "WITHOUT ANY EXPLANATION, GIVE ME ONLY THE NUMBER that corresponds to the relevance of this primitive to the given task (a score of 0.0 indicates a useless function, a score of 1.0 indicates an essential function).\n"

        cmd = ["ollama", "run", modelName, command]
        result = subprocess.run(cmd, capture_output = True, text = True)
        scores[function] = float(result.stdout)

    scores = dict(list(reversed(sorted(scores.items(), key = lambda x: x[1]))))
    #print(scores)
    #functions = list(scores.keys())
    scores = list({k: v for k, v in scores.items() if v > 0}.items())

    functions = []

    if (len(scores)):
        functions = [scores[0][0]]
        lastScore = scores[0][1]

        for i in range(1, len(scores)):
            if (abs(lastScore - scores[i][1]) > 0.25):
                break

            lastScore = scores[i][1]
            functions.append(scores[i][0])
    #print(functions)
    definitions = []

    #...

    return (functions, definitions)

def updateRegionNeurons(regionNeurons: dict, pairs: list[np.ndarray]):
    regionMap = dict()

    for input_ in pairs:
        s = utils.regionSet(input_, False)

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

def processTask(folder: str, task: str) -> int:
    taskPairs = trainTestPairs(folder, task)
    trainPairs = inputOutputPairs(taskPairs[0])
    testPairs = inputOutputPairs(taskPairs[1])

    falseNeuron = Neuron("False", lambda: False, [], bool)
    trueNeuron = Neuron("True", lambda: True, [], bool)
    inputNeuron = Neuron("input", lambda trainPairs = trainPairs: trainPairs[0], [], list[np.ndarray])
    pairsNeuron = Neuron("pairs", lambda v, taskPairs = taskPairs: [] if len(taskPairs[0]) != len(v) else [(v[i], taskPairs[0][i][1]) for i in range(0, len(taskPairs[0]))], [list[np.ndarray]], list[tuple[np.ndarray, np.ndarray]])
    trainPairsNeuron = Neuron("trainPairs", lambda taskPairs = taskPairs: taskPairs[0], [], list[tuple[np.ndarray, np.ndarray]])

    digitNeurons = []

    for i in range(0, 10):
        digitNeurons.append(Neuron(str(i), lambda i = i: i, [], int))

    connectionStr = "input"
    cost = brain.heuristic(trainPairs[0], trainPairs[1])

    while (cost):
        regionNeurons = dict()
        updateRegionNeurons(regionNeurons, trainPairs[0])

        neurons = [inputNeuron, falseNeuron, trueNeuron, pairsNeuron, trainPairsNeuron]

        for k, v in regionNeurons.items():
            neurons.append(v)

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

        p, definitions = bestPrimitives(folder, task, connectionStr, cost)

        import primitives

        for primitive in p:
            function = getattr(primitives, primitive)
            annotations = copy.deepcopy(function.__annotations__)
            outputType = copy.deepcopy(annotations["return"])
            del annotations["return"]
            inputTypes = list(annotations.values())

            neurons.append(Neuron(primitive, function, inputTypes, outputType))

        b = brain.Brain(neurons)

        connections = b.learn([trainPairs[1]], [list[np.ndarray]], level = 3)

        if (len(connections) == 0):
            continue

        connection = connections[0]
        connectionStr = connection.toStr()

        cost = brain.heuristic(connection.output(), trainPairs[1])

        if (cost):
            continue

        updateRegionNeurons(regionNeurons, testPairs[0]);

        inputNeuron.function = lambda testPairs = testPairs: testPairs[0]
        cost = brain.heuristic(connection.output(), testPairs[1])

        if (cost):
            continue

    print(connectionStr)

def test_task3c9b0459(): #Flip left/right and flip up/down
    processTask("training", "3c9b0459")

def test_task0d3d703e(): #Color mapping
    processTask("training", "0d3d703e")

def test_task253bf280(): #Draw colored segment between pixels that have same x or y coordinates
    processTask("training", "253bf280")
