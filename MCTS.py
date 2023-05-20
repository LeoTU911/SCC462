# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:11:51 2023

@author: Chunhui TU
"""
import numpy as np
import math
import random
import time
import utils # make sure that MCTS.py, experiment.py, utils.py and main.py are under the same directory and set a correct work dir



def generate_Map(mapSize, nPoint):
    """
    Create a map of a specified size, 
    generate a specified number of random bonus/punishment points, 
    and randomly generate start point
    """
    
    # create an empty 1D array
    emptySpace = [0 for i in range(mapSize[0]*mapSize[1])]

    # Select n random elements from the emptySpace
    randomIndices = random.sample(range(len(emptySpace)), nPoint)
    # Random start point
    randomStart   = random.sample(range(len(emptySpace)), 1)[0]
    
    # avoid start point is at the same position of random b/p point 
    while randomStart in randomIndices:
        randomStart = random.sample(range(len(emptySpace)), 1)

    # convert their 1D position to 2D position
    randomPoints = []
    for index in randomIndices:
        i = index // mapSize[0]
        j = index % mapSize[0]
        randomPoints.append((i,j))
    
    # birth point
    i_b = randomStart // mapSize[0]
    j_b = randomStart % mapSize[0]
    birthPoint = (i_b, j_b)
       
    return randomPoints, birthPoint

    
# generate random bonus/penalty marks
def generate_Random_Marks(nList, ratio):
    
    # Generate a complete list of rewards and penalties
    penaltyList = np.arange(-0.1, -1.1, -0.02).tolist()
    bonusList   = np.arange(0.1, 1, 0.02).tolist()
    
    # confirm the actual random rewards list
    nBonus   = math.ceil(nList * ratio)
    nPenalty = nList - nBonus
    rBonus   = random.sample(bonusList, nBonus)
    rPenalty = random.sample(penaltyList, nPenalty)
    
    # final list
    randomList = rBonus + rPenalty
    
    return randomList


class GameState:
    def __init__(self, mapSize, randomPoints, randomMarkList, rbirthPoint, birthPoint=None):
        self.board          = [[0 for _ in range(mapSize[0])] for _ in range(mapSize[1])]
        self.mapSize        = mapSize
        self.randomPoints   = randomPoints
        self.randomMarkList = randomMarkList
        self.rbirthPoint    = rbirthPoint
        self.currentPlayer  = 1
        self.score          = 0
        
        # Initialize birth point
        # if birthPoint is not specified manually, use the randomly generated birth point
        if birthPoint is None:
            self.currentPoint = rbirthPoint
        else:
            self.currentPoint = birthPoint
    
    def get_Possible_Actions(self):
        
        x, y = self.currentPoint
        possibleActions = []

        # populate the list of actions based on the current game state
        # make sure to exclude actions that move beyond the boundary of the game map

        # check north
        if y > 0:
            possibleActions.append(['N', (x, y-1)])
        # check south
        if y < 9:
            possibleActions.append(['S', (x, y+1)])
        # check west
        if x > 0 :
            possibleActions.append(['W', (x-1, y)])
        # check east
        if x < 9:
            possibleActions.append(['E', (x+1, y)])

        return possibleActions
    
    def get_Next_State(self, action):
        # new position
        x = action[1][0]
        y = action[1][1]
        
        nextState = GameState(mapSize        = self.mapSize,
                              randomPoints   = self.randomPoints,
                              randomMarkList = self.randomMarkList,
                              rbirthPoint    = self.rbirthPoint,
                              birthPoint     = (x,y))
        
        nextState.board = [row[:] for row in self.board]
        nextState.board[x][y] = self.currentPlayer
        nextState.currentPlayer = -self.currentPlayer
        
        if nextState.currentPoint in self.randomPoints:
            # random score
            score = random.sample(self.randomMarkList, 1)[0]
            self.score += score
        
        return nextState

    
    def get_Winner(self):
        
        win = False
        
        if self.score >= 0.5:
            win = True
        
        return win
    
    def is_Game_Over(self):
        
        gameOver = False
        
        if self.score <= -0.5:
            gameOver = True
        elif self.get_Winner():
            gameOver = True
        elif all(all(row) for row in self.board):
            gameOver = True
        
        return gameOver


class Node:
    def __init__(self, state, parent=None):
        self.state    = state
        self.parent   = parent
        self.children = []
        self.visits   = 0
        self.wins     = 0

    def add_child(self, childState):
        childNode = Node(state=childState, parent=self)
        self.children.append(childNode)
        return childNode

    def expand(self):
        untriedActions = self.state.get_Possible_Actions()
        if untriedActions:
            action    = random.choice(untriedActions)
            nextState = self.state.get_Next_State(action)
            childNode = self.add_child(childState=nextState)
            
            return childNode
        return None

    def simulate(self):
        currentState = self.state
        while not currentState.is_Game_Over():
            action = random.choice(currentState.get_Possible_Actions())
            currentState = currentState.get_Next_State(action)
            #print(currentState.currentPoint)
        return currentState.get_Winner()

    def backpropagate(self, result):
        node = self
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

    def get_Best_Child(self):
        bestChild = max(self.children, key=lambda c: c.wins / c.visits)
        return bestChild.state
        

def monte_Carlo_Tree_Search(initial_State, num_Iterations=100, select_method='UCB1', C_p=0.9):
    
    # monitor the resources consumption
    # include CPU, MEM and time
    # record start time of total
    startTime_Total = time.time()
    monitor_thread, stop_event, cpu_percent_list, memory_usage_list = utils.monitor_Resource_Usage()    
    
    # start MCTS
    rootNode = Node(initial_State)

    for _ in range(num_Iterations):
        node = rootNode

        # Selection
        while node.children:
            # UCB1
            if select_method == 'UCB1':
                node = max(node.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * math.log(node.visits) / c.visits))
            else: 
            # UCT
                node = max(node.children, key=lambda c: c.wins / c.visits + 2 * C_p * math.sqrt(2 * math.log(node.visits) / c.visits))
        
        # Expansion
        childNode = node.expand()
        #print('childNode:', childNode.state.currentPoint)
        if childNode:
            result = childNode.simulate()
            childNode.backpropagate(result)
            bestChild_State = rootNode.get_Best_Child()
    
    
    # calculate resources consumption
    endTime_Total = time.time()
    # Record the resource usage after 1 second as a benchmark
    time.sleep(1)
    # stop monitoring
    stop_event.set()
    # join the monitoring thread to wait for it to finish
    monitor_thread.join()
    
    # calculate the overall resources consumption
    timeSpent_Total = endTime_Total - startTime_Total
    timeSpent_Loop  = timeSpent_Total/num_Iterations
    cpu_avg, mem_avg = utils.calculate_Resource_Usage(cpu_percent_list, memory_usage_list, timeSpent_Total)
    resources = [round(timeSpent_Total, 3), round(timeSpent_Loop, 3),  cpu_avg, mem_avg]
    
    return bestChild_State, resources


################################ END ##########################################
    