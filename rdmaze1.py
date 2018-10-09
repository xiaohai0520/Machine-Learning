import math
import sys


#  die class
class die:
    def __init__(self):
        self.up = self.down = self.north = self.south = self.east = self.west = 0

    def moveNorth(self):
        self.up, self.down, self.north,self.south, self.east, self.west =\
            self.south, self.north, self.up, self.down, self.east, self.west

    def moveSouth(self):
        self.up, self.down, self.north, self.south, self.east, self.west =\
            self.north, self.south,self.down,self.up,self.east, self.west

    def moveEast(self):
        self.up, self.down, self.north, self.south, self.east, self.west =\
            self.west, self.east,self.north,self.south,self.up, self.down

    def moveWest(self):
        self.up, self.down, self.north, self.south, self.east, self.west = \
            self.east, self.west, self.north, self.south, self.down, self.up

    def transferFrom(self,prev):
        self.up, self.down, self.north, self.south, self.east, self.west = \
            prev.up, prev.down, prev.north, prev.south, prev.east, prev.west


#maze class
class maze:


    def __init__(self):
        self.mazeModel = []         #maze array
        self.obstacles = []         #maze block
        self.not_visited = []       #point will be visited
        self.visited = []           #point already visited
        self.start = None           #start point
        self.goal = None            #end point
        self.getResult = False
        self.goalNode = None
        self.nodeGenre = 0
        self.nodeVisit = 0

    #save maze into the maze model
    def readMaze(self,fileName):
        with open(fileName) as f:
            for line in f:
                self.mazeModel.append(list(line.strip()))
            f.close()

    #setting about the maze
    def createMaze(self):

        global finalGoal
        row = len(self.mazeModel)
        # print('row',row)
        col = len(self.mazeModel[0])
        # print('col', col)
        #find the start goal block
        for i in range(row):
            for j in range(col):
                # print(i,j)
                if self.mazeModel[i][j] == 'S':
                    # print('start', i, j)
                    self.start = Node(i,j)
                    self.start.die.up = 1
                    self.start.die.down = 6
                    self.start.die.north = 2
                    self.start.die.south = 5
                    self.start.die.east = 3
                    self.start.die.west = 4

                elif self.mazeModel[i][j] == 'G':
                    # print('goal' ,i,j)
                    self.goal = Node(i,j)
                elif self.mazeModel[i][j] == '*':
                    self.obstacles.append(Node(i,j))

    #draw the maze
    def outputMaze(self):

        for i in range(len(self.mazeModel)):
            for j in range(len(self.mazeModel[0])):
                print(self.mazeModel[i][j], ' ', end="")
            print("")


    def returnSameInNV(self,cur):

        for node in self.not_visited:
            if node.x == cur.x and node.y == cur.y and node.die.up == cur.die.up and node.die.north == cur.die.north:
                return node
        return None

    def A_Star_Search(self,type):
        global VisitNo
        global GenerNo


        #add root into queue
        self.not_visited = [self.start]
        self.visited = []
        found = False
        while(self.not_visited and not found):
            # print(self.not_visited)
            # print(self.start.f)
            self.not_visited.sort(key=lambda x : x.f)
            #get the first node
            cur = self.not_visited.pop(0)

            # print(cur.x,cur.y)

            self.nodeVisit += 1
            #add into visited
            self.visited.append(cur)
            #make sure if goal
            if self.checkGoal(cur):
                found = True   #change found
                self.goalNode = cur #save the goal state
                self.getResult = True
                break


            #find the next possible moves return an node array
            nextMoves = self.findNextMoves(cur,cur.die)

            #cal h for these node
            movesWithH = self.setNodesH(nextMoves,type)   #add type


            #check each node
            for node in movesWithH:

                node.calG(cur)

                node.calF()
                # print(node.x,node.y,self.inVisit(node))
                # print(self.visited)
                if self.inVisit(node):
                    continue

                #find the same node in the not visit
                #if no same, set pre and add to the not visited
                #if have same, compare the f save the little f
                while True:

                    sameNode = self.returnSameInNV(node)


                    if not sameNode:
                        node.prev = cur
                        self.not_visited.append(node)
                        self.nodeGenre += 1
                        break
                    else:

                        if node.f < sameNode.f:
                            index = self.findID(sameNode,self.not_visited)
                            self.not_visited.pop(index)
                        else:
                            break


            if not self.not_visited:
                solution = False
                break



    def outputPath(self,solutionNode):

        tracback= []
        # print(solutionNode)
        while solutionNode:
            tracback.append(solutionNode)
            solutionNode = solutionNode.prev

        tracback = tracback[::-1]

        for node in tracback:
            # print(node.x, node.y)
            print('-------------------------------------------')
            self.mazeModel[node.x][node.y] = node.die.up
            self.outputMaze()



    #find the index of the same node
    def findID(selfs,target,nodes):
        for i,node in enumerate(nodes):
            if node.x == target.x and node.y == target.y:
                return i


    #make sure visit or not
    def inVisit(self, cur):
        for node in self.visited:
            if node.x == cur.x and node.y == cur.y and node.die.up == cur.die.up and node.die.north == cur.die.north:
                return True
        return False

    #check if goal   x y and 1
    def checkGoal(self,node):
        if node.x == self.goal.x and node.y == self.goal.y and node.die.up == 1:
            return True
        return False


    #check not block
    def checkObstacle(self,node):

        for obstacle in self.obstacles:
            if obstacle.x == node.x and obstacle.y == node.y:
                return True
        return False

    #set h for each node
    def setNodesH(self,nodes,type):
        res = []
        for node in nodes:
            node.calH(self.goal,type)
            res.append(node)
        return res


    #find next step  all possible
    def findNextMoves(self,cur,curDie):

        #it can move to four directions
        res = [] #result array

        nextDieNorth = die() #move to north
        nextDieNorth.transferFrom(curDie)  #get the cur state

        nextDieSouth = die()
        nextDieSouth.transferFrom(curDie)

        nextDieWest = die()
        nextDieWest.transferFrom(curDie)

        nextDieEast = die()
        nextDieEast.transferFrom(curDie)

        x = cur.x
        y = cur.y

        # move to north
        if x != 0:
            nextDieNorth.moveNorth()
            if nextDieNorth.up != 6: # not 6
                northMove = Node(x-1,y)
                if not self.checkObstacle(northMove): #check can move not block
                    northMove.die = nextDieNorth
                    res.append(northMove)

        #move to south
        if x != len(self.mazeModel) - 1 :
            nextDieSouth.moveSouth()
            if nextDieSouth.up != 6:
                southMove = Node(x + 1,y)
                if not self.checkObstacle(southMove):
                    southMove.die = nextDieSouth
                    res.append(southMove)

        #move to west
        if y != 0:
            nextDieWest.moveWest()
            if nextDieWest.up != 6:
                westMove = Node(x,y-1)
                if not self.checkObstacle(westMove):
                    westMove.die = nextDieWest
                    res.append(westMove)

        #move to east
        if y != len(self.mazeModel[0]) - 1:
            nextDieEast.moveEast()
            if nextDieEast.up != 6:
                eastMove = Node(x,y+1)
                if not self.checkObstacle(eastMove):
                    eastMove.die = nextDieEast
                    res.append(eastMove)

        return res




class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.die = die()
        self.prev = None

        self.h = 0          # to goal cost
        self.g = 0          # already cost
        self.f = 0          # guess cost   f = g + h



    def calH(self,goal,type):   #cal the h
        if type == "Manhattan":
            self.h = math.sqrt(math.pow(self.x - goal.x, 2) + math.pow(self.y - goal.y, 2))
        elif type == "Euclidean":
            self.h = math.fabs(self.x - goal.x) + math.fabs(self.y - goal.y)
        elif type == "RollDie":
            self.calHwithDie(goal)


    def calG(self,prev):    #CALCULATE g
        self.g = prev.g + 1
        return self.g

    def calF(self):     #CALCULATE f
        self.f = self.g + self.h
        return self.f

    def calHwithDie(self,goal):
        cur1 = die()
        cur2 = die()
        cur1.transferFrom(self.die)
        cur2.transferFrom(self.die)
        dx = int(math.fabs(self.x - goal.x))
        dy = int(math.fabs(self.y - goal.y))

        x = self.x - goal.x
        y = self.y - goal.y

        if x > 0:
            for i in range(dx):
                cur1.moveSouth()
        elif x < 0:
            for i in range(dx):
                cur1.moveNorth()
        if y > 0:
            for i in range(dy):
                cur1.moveWest()
        elif y < 0:
            for i in range(dy):
                cur1.moveEast()

        up1 = cur1.up

        if y > 0:
            for i in range(dy):
                cur2.moveWest()
        elif y < 0:
            for i in range(dy):
                cur2.moveEast()
        if x > 0:
            for i in range(dx):
                cur2.moveSouth()
        elif x < 0:
            for i in range(dx):
                cur2.moveNorth()
        up2 = cur2.up

        up = min(up1,up2)
        m = math.fabs(self.x - goal.x) + math.fabs(self.y - goal.y)
        if up == 1:
            self.h = m
        elif up == 6:
            self.h = m + 6
        else:
            self.h = m + 4


def main():


    filename = sys.argv[1]
    # filename = 'puzzle1.txt'

    distancetypes = ['Euclidean','Manhattan','RollDie']
    for distanceType in distancetypes:

        testmaze = maze()
        testmaze.readMaze(filename)
        testmaze.createMaze()
        print('\n----------------{}------------------'.format(distanceType))

        testmaze.outputMaze()

        testmaze.A_Star_Search(distanceType)


        if testmaze.getResult:
            testmaze.outputPath(testmaze.goalNode)

        print("\nNumber of nodes generated : ", testmaze.nodeGenre)
        print("Number of nodes visited : ",testmaze.nodeVisit)


main()