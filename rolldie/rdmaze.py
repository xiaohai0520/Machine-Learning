import sys
import math

noOfNodesGenerated = 1
noOfStatesVisited = 0
solution = True
currentState = None
startGlobal = None
goalGlobal = None
mazeEuclidean = None
mazeManhattan = None
mazeManWithDie = None

h = ["Euclidean", "Manhattan"]


def main():
    global noOfNodesGenerated
    global noOfStatesVisited
    global solution
    global currentState
    global mazeEuclidean
    global mazeManhattan
    global mazeManWithDie
    fileName = sys.argv[1]
    """ Read lines from file"""
    lines = read_fileRows(fileName)
    """" create maze for three type of heuristic """
    mazeEuclidean = Maze(lines)
    mazeManhattan = Maze(lines)
    mazeManWithDie = Maze(lines)

    maze1 = Maze(lines)
    maze1.createMaze(lines)
    maze1.print_graph(maze1.maze)

    print("-------------------Solving for Euclidean-------------------")
    maze1.aStarSearch("Euclidean")
    if (solution == True):
        maze1.findSolutionPath(currentState, "Euclidean")

    print("Number of nodes generated : ", noOfNodesGenerated)
    print("Number of nodes visited : ", noOfStatesVisited)

    solution = False

    noOfNodesGenerated = 1
    noOfStatesVisited = 0
    print("-------------------Solving for Manhattan-------------------")
    maze1.aStarSearch("Manhattan")
    if (solution == True):
        maze1.findSolutionPath(currentState, "Manhattan")
    print("Number of nodes generated : ", noOfNodesGenerated)
    print("Number of nodes visited : ", noOfStatesVisited)

    solution = False

    noOfNodesGenerated = 1
    noOfStatesVisited = 0
    print("-------------------Solving for ManWithDie-------------------")
    maze1.aStarSearch("ManWithDie")
    if (solution == True):
        maze1.findSolutionPath(currentState, "ManWithDie")
    print("Number of nodes generated : ", noOfNodesGenerated)
    print("Number of nodes visited : ", noOfStatesVisited)


def read_fileRows(fileName):
    """ Read file"""
    rows = []
    inFile = open(fileName, 'rU')
    for line in inFile:
        rows.append(list(line.rstrip()))
    inFile.close()
    return rows


class Die:
    def __init__(self):
        self.top = 0
        self.north = 0
        self.bottom = 0
        self.east = 0
        self.west = 0
        self.south = 0
        """Initial state associated with start node"""

    def moveUp(self):
        """move the dice up and change the orientation acordingly"""
        oldNorth = self.north
        oldBottom = self.bottom
        oldTop = self.top
        oldSouth = self.south
        oldWest = self.west
        oldEast = self.east
        self.west = oldWest
        self.east = oldEast
        self.top = oldSouth
        self.bottom = oldNorth
        self.south = oldBottom
        self.north = oldTop

    def moveDown(self):
        """move the dice down and change the orientation acordingly"""
        oldNorth = self.north
        oldBottom = self.bottom
        oldTop = self.top
        oldSouth = self.south
        oldWest = self.west
        oldEast = self.east
        self.west = oldWest
        self.east = oldEast
        self.top = oldNorth
        self.north = oldBottom
        self.south = oldTop
        self.bottom = oldSouth

    def moveRight(self):
        """move the dice right and change the orientation acordingly"""
        oldEast = self.east
        oldBottom = self.bottom
        oldTop = self.top
        oldWest = self.west
        oldNorth = self.north
        oldSouth = self.south
        self.north = oldNorth
        self.south = oldSouth
        self.bottom = oldEast
        self.east = oldTop
        self.top = oldWest
        self.west = oldBottom

    def moveLeft(self):
        """move the dice left and change the orientation acordingly"""
        oldEast = self.east
        oldBottom = self.bottom
        oldTop = self.top
        oldWest = self.west
        oldNorth = self.north
        oldSouth = self.south
        self.north = oldNorth
        self.south = oldSouth
        self.west = oldTop
        self.top = oldEast
        self.east = oldBottom
        self.bottom = oldWest

    def copyState(self, parentDie):
        """Copy the current state into the die"""
        self.top = parentDie.top
        self.north = parentDie.north
        self.south = parentDie.south
        self.east = parentDie.east
        self.west = parentDie.west
        self.bottom = parentDie.bottom


class Maze:
    openList = []
    closeList = []
    allNodes = []
    die = Die()  # die created with initial configuration set up for start node

    # need to create Die's object and associate with start node

    def __init__(self, linesInText):
        self.rows = len(linesInText)
        self.columns = len(linesInText[0])
        self.maze = [['_' for row in range(self.columns)] for row in
                     range(self.rows)]  # creating a 2D matrix; initialized to '_''
        self.start = None
        self.goal = None
        self.blocks = []

    def createMaze(self, linesInText):
        """ Initialize the maze"""
        global mazeEuclidean
        global mazeManhattan
        global mazeManWithDie
        global goalGlobal
        i = 0
        for r in range(self.rows):
            for c in range(self.columns):
                self.maze[r][c] = linesInText[r][c]
                node = Node(r, c)  # making each cell in the maze a node
                mazeEuclidean.maze[r][c] = linesInText[r][c]
                mazeManhattan.maze[r][c] = linesInText[r][c]
                mazeManWithDie.maze[r][c] = linesInText[r][c]
                if (linesInText[r][c] == 'S'):
                    self.start = node  # assigning newly node to self.start
                    # initializing start node with initial config of die
                    self.start.die.top = 1
                    self.start.die.north = 2
                    self.start.die.east = 3
                    self.start.die.west = 4
                    self.start.die.south = 5
                    self.start.die.bottom = 6
                    self.start.g = 0  # start node has g = 0

                elif (linesInText[r][c] == 'G'):
                    self.goal = node  # assigning newly node to self.goal
                elif (linesInText[r][c] == '*'):
                    self.blocks.append(node)  # appending nodes with '*' to blocks[]

        startGlobal = self.start
        goalGlobal = self.goal
        self.start.h_euclidean = self.start.setHEuclidean(self.start)
        self.start.h_manhattan = self.start.setHManhattan(self.start)
        self.start.h_manWithDie = self.start.setHManWithDie(self.start)

    def print_graph(self, mazeMatrix):
        """print the graph"""
        for i in range(len(mazeMatrix)):
            for j in range(len(mazeMatrix[i])):
                print(mazeMatrix[i][j], ' ', end="")
            print("")

    def checkIfPresentInOpenList(self, node):
        """"Check if presenr in openList"""
        x, y = node.x, node.y
        for item in openList:
            if (item.x == x and item.y == y):
                return True
        return False

    def checkNodeForBlocks(self, node):
        """check for blocks"""
        x, y = node.x, node.y
        for b in self.blocks:
            if (b.x == x and b.y == y):
                return True
        return False

    def possibleMoves(self, currentNode, currentDie):
        """generate all possible nodes"""
        childDie = Die()
        childDie1 = Die()
        childDie2 = Die()
        childDie3 = Die()
        childDie.copyState(currentDie)
        childDie1.copyState(currentDie)
        childDie2.copyState(currentDie)
        childDie3.copyState(currentDie)

        xCordinate = currentNode.x
        yCordinate = currentNode.y
        allPossibleMoves = []

        """genearated new node when die is moved up and changes the state accordingly"""
        if (xCordinate != 0):  # die will not change state here
            childDie.moveUp()
            if (childDie.top != 6):
                upMove = Node(xCordinate - 1, yCordinate)
                if (self.checkNodeForBlocks(upMove) == False):
                    upMove.setDieState(childDie)
                    allPossibleMoves.append(upMove)

        """genearated new node when die is moved down and changes the state accordingly"""
        if (xCordinate != self.rows - 1):
            childDie1.moveDown()
            if (childDie1.top != 6):
                downMove = Node(xCordinate + 1, yCordinate)
                if (self.checkNodeForBlocks(downMove) == False):
                    downMove.setDieState(childDie1)
                    allPossibleMoves.append(downMove)

        """genearated new node when die is moved left and changes the state accordingly"""
        if (yCordinate != 0):
            childDie2.moveLeft()
            if (childDie2.top != 6):
                leftMove = Node(xCordinate, yCordinate - 1)
                if (self.checkNodeForBlocks(leftMove) == False):
                    leftMove.setDieState(childDie2)
                    allPossibleMoves.append(leftMove)

        """genearated new node when die is moved right and changes the state accordingly"""
        if (yCordinate != self.columns - 1):
            childDie3.moveRight()
            if (childDie3.top != 6):
                rightMove = Node(xCordinate, yCordinate + 1)
                if (self.checkNodeForBlocks(rightMove) == False):
                    rightMove.setDieState(childDie3)
                    allPossibleMoves.append(rightMove)
        return allPossibleMoves

    def checkIfGoal(self, node):
        """check if the goal"""
        global goalGlobal
        if (node.x == goalGlobal.x and node.y == goalGlobal.y):
            return True
        else:
            return False

    def returnIfPresentInOpenList(self, openList, node):
        """check if present in the openlist with its comination of orienttion and cordiantes"""
        x, y, dieState = node.x, node.y, node.die
        for item in openList:
            if (item.x == x and item.y == y and dieState.top == item.die.top and dieState.north == item.die.north):
                return item
        return None

    def checkIfPresentInClosedList(self, closeList, node):
        """check if present in the openlist with its comination of orienttion and cordiantes"""
        x, y, dieState = node.x, node.y, node.die
        for item in closeList:
            if (item.x == x and item.y == y and dieState.top == item.die.top and dieState.north == item.die.north):
                return True
        return False

    def setHeuristicsForNodes(self, allNodes):
        """set heuristics"""
        nodesWithHeuristics = []
        for node in allNodes:
            node.h_euclidean = node.setHEuclidean(node)
            node.h_manhattan = node.setHManhattan(node)
            node.h_manWithDie = node.setHManWithDie(node)
            nodesWithHeuristics.append(node)
        return nodesWithHeuristics

    def findSolutionPath(self, solutionNode, type):
        """print solution path on the maze"""
        global mazeEuclidean
        global mazeManhattan
        global mazeManNRoll
        tracBackNodes = []
        temp = solutionNode
        print("Tracing back")
        while (solutionNode != None):
            tracBackNodes.append(solutionNode)
            temp = solutionNode.parent
            solutionNode = solutionNode.parent
        index = len(tracBackNodes)
        index = index - 1
        if (type == "Manhattan"):
            tempMaze = mazeManhattan.maze
        elif (type == "Euclidean"):
            tempMaze = mazeEuclidean.maze
        elif (type == "ManWithDie"):
            tempMaze = mazeManWithDie.maze
        """reverse the list and print"""
        for node in tracBackNodes:
            node = tracBackNodes[index]
            tempMaze[node.x][node.y] = node.die.top
            index = index - 1
            self.print_graph(tempMaze)
            print('Orientation of die at position: ', node.x, ',', node.y)
            print('Top face: ', node.die.top)
            print('North face: ', node.die.north)
            print('South face: ', node.die.south)
            print('East face: ', node.die.east)
            print('West face: ', node.die.west)
            print('Bottom face: ', node.die.bottom)
            print('_______________________________________')

    def getIndexOfChildOList(self, openList, presentNode):
        x, y = presentNode.x, presentNode.y
        i = 0
        for node in openList:
            if (node.x == x and node.y == y):
                return i
            else:
                i = i + 1

    def aStarSearch(self, type):
        global noOfNodesGenerated
        global noOfStatesVisited
        global solution
        global currentState
        solution = True
        closeList = []
        openList = []  # Q.PriorityQueue()
        start = self.start
        openList.append(start)
        current = self.start
        found = False
        # noOfNodesGenerated = noOfNodesGenerated + 1
        while (len(openList) > 0 and found == False):
            if (type == "Euclidean"):
                openList = (
                sorted(openList, key=lambda node: node.fEuclidean))  # sorts the openList list at each iteration
            elif (type == "Manhattan"):
                openList = (sorted(openList, key=lambda node: node.fManhattan))
            elif (type == "ManWithDie"):
                openList = (sorted(openList, key=lambda node: node.fManWithDie))
            current = openList[0]  # getting node with smallest 'f' value
            openList.pop(0)  # removing that node from openList
            noOfStatesVisited = noOfStatesVisited + 1
            closeList.append(current)  # appending node to closedList as we explore possible states from node
            if (self.checkIfGoal(current) == True and current.die.top == 1):  # if I have found my goal
                found = True
                print("Solution Found")
                currentState = current
                break

            listOfPossibleMoves = self.possibleMoves(current, current.die)
            withHeuristics = self.setHeuristicsForNodes(listOfPossibleMoves)
            for node in withHeuristics:
                child = node
                gScore = child.calculateG(current)
                if (type == "Euclidean"):
                    fScore = child.calculateFEuclidean(gScore, child.h_euclidean)  #
                elif (type == "Manhattan"):
                    fScore = child.calculateFManhattan(gScore, child.h_manhattan)  #
                elif (type == "ManWithDie"):
                    fScore = child.calculateFManWithDie(gScore, child.h_manWithDie)

                if (self.checkIfPresentInClosedList(closeList, child) == True):
                    continue
                # if the same state of the node is in openList
                presentChild = self.returnIfPresentInOpenList(openList, child)
                if (presentChild != None):
                    if (type == "Euclidean"):
                        existingF = presentChild.fEuclidean
                    elif (type == "Manhattan"):
                        existingF = presentChild.fManhattan
                    elif (type == "ManWithDie"):
                        existingF = presentChild.fManWithDie

                if ((presentChild == None) or (fScore < existingF)):
                    child.parent = current
                    child.g = gScore
                    if (type == "Euclidean"):
                        child.fEuclidean = fScore
                    elif (type == "Manhattan"):
                        child.fManhattan = fScore
                    elif (type == "ManWithDie"):
                        child.fManWithDie
                    if (presentChild != None):
                        idx = self.getIndexOfChildOList(openList, presentChild)
                        openList.pop(idx)  # error isssssss posssssssible
                    openList.append(child)
                    noOfNodesGenerated = noOfNodesGenerated + 1
                    # currentState = child

            if (len(openList) == 0):
                solution = False
                print("No Solution")
                break


class Node:
    die = Die()

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.h = 0
        self.fEuclidean = 0
        self.fManhattan = 0
        self.fManWithDie = 0
        self.g = 0
        self.parent = None
        self.h_euclidean = 0
        self.h_manhattan = 0
        self.h_manWithDie = 0

    # self.die = die

    def setHEuclidean(self, node):
        self.h_euclidean = self.calculateH(node, "Euclidean", goalGlobal)
        return self.h_euclidean

    def setHManhattan(self, node):
        self.h_manhattan = self.calculateH(node, "Manhattan", goalGlobal)
        return self.h_manhattan

    def setHManWithDie(self, node):
        # global goalGlobal
        self.h_manWithDie = self.calculateManWithDie(node, goalGlobal, node.die)
        return self.h_manWithDie

    def calculateManWithDie(self, current, goal, dieState):
        currentDieState1 = Die()
        currentDieState1.copyState(dieState)
        currentDieState2 = Die()
        currentDieState2.copyState(dieState)
        x1, x2 = current.x, goal.x
        y1, y2 = current.y, goal.y
        tempDx = math.fabs(x1 - x2)
        dx = int(tempDx)
        x = x1 - x2
        tempDy = math.fabs(y1 - y2)
        dy = int(tempDy)
        y = y1 - y2
        if (x > 0):
            for i in range(dx):
                currentDieState1.moveDown()
        elif (x < 0):
            for i in range(dx):
                currentDieState1.moveUp()
        if (y > 0):
            for i in range(dy):
                currentDieState1.moveLeft()
        if (y < 0):
            for i in range(dy):
                currentDieState1.moveRight()

        currentTopValue1 = currentDieState1.top

        if (y > 0):
            for i in range(dy):
                currentDieState2.moveLeft()
        if (y < 0):
            for i in range(dy):
                currentDieState2.moveRight()
        if (x > 0):
            for i in range(dx):
                currentDieState2.moveDown()
        elif (x < 0):
            for i in range(dx):
                currentDieState2.moveUp()
        currentTopValue2 = currentDieState2.top

        if (currentTopValue1 < currentTopValue2):
            currentTopValue = currentTopValue1
        else:
            currentTopValue = currentTopValue2

        manHattanDist = math.fabs(x1 - x2) + math.fabs(y1 - y2)
        if (currentTopValue == 1):
            return (0 + manHattanDist)
        elif (currentTopValue == 6):
            return (6 + manHattanDist)
        else:
            return (4 + manHattanDist)

    def setParent(self, parent):
        self.parent = parent

    def calculateH(self, current, type, goal):  # passing starting node and goal node as parameters
        x1, y1 = current.x, current.y
        x2, y2 = goal.x, goal.y

        if (type == "Manhattan"):
            manhattan_dist = math.fabs(x1 - x2) + math.fabs(y1 - y2)
            return manhattan_dist

        elif (type == "Euclidean"):
            euclidean_dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
            return euclidean_dist

        """elif(type == "ManWithDie"):
            man_with_die_dist = self.calculateManWithDie(current, goal, current.die)
            return man_with_die_dist"""

    def calculateG(self, parent):
        self.g = parent.g + 1
        return self.g

    def setDieState(self, dieState):
        self.die = dieState

    def getDieState(self):
        return self.die

    def calculateFEuclidean(self, g, euclidean):
        self.fEuclidean = g + euclidean
        return self.fEuclidean

    def calculateFManhattan(self, g, manhattan):
        self.fManhattan = g + manhattan
        return self.fManhattan

    def calculateFManWithDie(self, g, manNRoll):
        self.fManWithDie = g + manNRoll
        return self.fManWithDie


"""	def getManhattanDistance(self, node):
		global goalGlobal
		x2, y2 = goalGlobal.x, goalGlobal.y
		x1, y1 = node.x, node.y
		manhattan_dist = math.fabs(x1 - x2) + math.fabs(y1 - y2)
		return manhattan_dist
"""

main()