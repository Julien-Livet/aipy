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

def bestPrimitives(folder: str, task: str, connectionStr: str, cost: float) -> list[str]:
    command = "Here is the full list of available functions defined in this Python file:\n"

    file = open("primitives.py")
    content = file.read()
    file.close()

    command += content + "\n"
    command += "---\n"
    command += "Here is an ARC AGI task:\n"

    url = urllib.request.urlopen("https://raw.githubusercontent.com/arcprize/ARC-AGI-2/refs/heads/main/data/" + folder + "/" + task + ".json")

    command += url.read().decode() + "\n"
    command += "---\n"
    command += "For the previous ARC task, assign each function a relevance score between 0.0 and 1.0 indicating the probability that it is useful for solving the task.\n"
    command += "Currently, for a cost of " + str(cost) + ", the best expression is: " + connectionStr + ".\n"
    command += "The answer must strictly adhere to the following format:\n"
    command += "{\n"
    command += """  "functionName1": 0.9,\n"""
    command += """  "functionName2": 0.6\n"""
    command += "}\n"

    cmd = ["ollama", "run", "gemma3:27b", command.replace('"', '\\"')]
    result = subprocess.run(cmd, capture_output = True, text = True)
    data = json.loads(result.stdout.replace("```json", "").replace("```", ""))
    scores = list(reversed(sorted(data.items(), key = lambda x: x[1])))

    functions = [scores[0][0]]
    lastScore = scores[0][1]

    for i in range(1, len(scores)):
        if (abs(lastScore - scores[i][1]) >= 0.25):
            break

        functions.append(scores[i][0])
    print(functions)
    return functions

def processTask(folder: str, task: str) -> int:
    taskPairs = trainTestPairs(folder, task)
    trainPairs = inputOutputPairs(taskPairs[0])
    testPairs = inputOutputPairs(taskPairs[1])

    inputNeuron = Neuron("input", lambda trainPairs = trainPairs: trainPairs[0], [], list[np.ndarray])
    pairsNeuron = Neuron("pairs", lambda v, taskPairs = taskPairs: [] if len(taskPairs[0]) != len(v) else [(v[i], taskPairs[0][i][1]) for i in range(0, len(taskPairs[0]))], [list[np.ndarray]], list[tuple[np.ndarray, np.ndarray]])
    trainPairsNeuron = Neuron("trainPairs", lambda taskPairs = taskPairs: taskPairs[0], [], list[tuple[np.ndarray, np.ndarray]])

    connectionStr = "input"
    cost = brain.heuristic(trainPairs[0], trainPairs[1])

    while (cost):
        neurons = [inputNeuron, pairsNeuron, trainPairsNeuron]

        p = bestPrimitives(folder, task, connectionStr, cost)

        import primitives

        for primitive in p:
            try:
                function = getattr(primitives, primitive)
                annotations = function.__annotations__
                outputType = copy.copy(annotations["return"])
                del annotations["return"]
                inputTypes = list(annotations.values())

                neurons.append(Neuron(primitive, function, inputTypes, outputType))
            except:
                pass

        b = brain.Brain(neurons)

        connections = b.learn([trainPairs[1]], [list[np.ndarray]], level = 2)

        if (len(connections) == 0):
            continue

        connection = connections[0]
        connectionStr = connection.toStr()

        cost = brain.heuristic(connection.output(), trainPairs[1])

        if (cost):
            continue

        inputNeuron.function = lambda testPairs = testPairs: testPairs[0]
        cost = brain.heuristic(connection.output(), testPairs[1])

        if (cost):
            continue

    print(connectionStr)

def test_task3c9b0459(): #Flip left/right and flip up/down
    processTask("training", "3c9b0459")

def test_task0d3d703e(): #Color mapping
    processTask("training", "0d3d703e")
