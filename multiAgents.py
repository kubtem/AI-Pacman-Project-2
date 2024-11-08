# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            closestFoodDistance = min(foodDistances)
        else:
            closestFoodDistance = 0  

        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        closestGhostDistance = min(ghostDistances)

        score = successorGameState.getScore() - 1.5 * closestFoodDistance

        if closestGhostDistance > 0:
            score += 2.0 / closestGhostDistance  

        return score
    

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
    
        def minimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return max(
                    minimax(1, depth, gameState.generateSuccessor(agentIndex, action))
                    for action in gameState.getLegalActions(agentIndex)
                )
            else:
                nextAgent = agentIndex + 1  
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0  
                    depth += 1
                return min(
                    minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action))
                    for action in gameState.getLegalActions(agentIndex)
                )

        bestScore, bestAction = float("-inf"), None
        for action in gameState.getLegalActions(0):
            score = minimax(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore, bestAction = score, action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                v = float("-inf")
                for action in gameState.getLegalActions(agentIndex):
                    v = max(v, alphaBeta(1, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if v > beta:
                        return v 
                    alpha = max(alpha, v)
                return v
            else:
                v = float("inf")
                nextAgent = agentIndex + 1  
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0  
                    depth += 1
                for action in gameState.getLegalActions(agentIndex):
                    v = min(v, alphaBeta(nextAgent, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if v < alpha:
                        return v  
                    beta = min(beta, v)
                return v

        bestScore, bestAction = float("-inf"), None
        alpha, beta = float("-inf"), float("inf")
        for action in gameState.getLegalActions(0):
            score = alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if score > bestScore:
                bestScore, bestAction = score, action
            alpha = max(alpha, score)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        def expectimax(state, agentIndex, depth):
            if state.isWin() or state.isLose() or (agentIndex == 0 and depth == self.depth):
                return self.evaluationFunction(state)

            if agentIndex == 0:
                return max(expectimax(state.generateSuccessor(agentIndex, action), 1, depth)
                           for action in state.getLegalActions(agentIndex))
            else:

                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():  
                    nextAgent = 0
                    depth += 1

                actions = state.getLegalActions(agentIndex)
                if len(actions) == 0:  
                    return self.evaluationFunction(state)

                return sum(expectimax(state.generateSuccessor(agentIndex, action), nextAgent, depth)
                           for action in actions) / len(actions)

        legalMoves = gameState.getLegalActions(0)
        scores = [expectimax(gameState.generateSuccessor(0, action), 1, 0) for action in legalMoves]
        bestScore = max(scores)
        bestActions = [action for action, score in zip(legalMoves, scores) if score == bestScore]

        return random.choice(bestActions)


from util import manhattanDistance

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()

    ghostPenalty = 0
    for ghost in ghostStates:
        distance = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            ghostPenalty += 200 / (distance + 1)
        else:
            if distance > 0:
                ghostPenalty -= 10 / distance

    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodList]
    closestFoodDist = min(foodDistances) if foodDistances else 1

    capsuleDistances = [manhattanDistance(pacmanPos, capPos) for capPos in capsules]
    closestCapsuleDist = min(capsuleDistances) if capsuleDistances else 1

    evaluation = (
        currentScore
        + 10 / closestFoodDist
        + 20 / closestCapsuleDist
        + ghostPenalty
        - 4 * len(foodList)
        - 15 * len(capsules)
    )

    return evaluation



# Abbreviation
better = betterEvaluationFunction
