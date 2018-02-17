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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPosition).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        
        # ----------------------------------------------------------------------- #
        # Let us assume initial weight for Food and Ghost
        WEIGHT_FOOD = 10.0
        WEIGHT_GHOST = 10.0

        # The initial score successor game states 
        score = successorGameState.getScore()

        # Distances to the Ghost from Pacman's new position
        distanceToGhost = manhattanDistance(newPosition, newGhostStates[0].getPosition())
        if distanceToGhost > 0:
            # decrease points if pacman is getting close to the ghost
            score -= WEIGHT_GHOST / distanceToGhost

        # Distances to the Food from Pacman's new position
        distancesToFood = [manhattanDistance(newPosition, food) for food in newFood.asList()]
        if len(distancesToFood):
            # increase points if pacman eats any food
            score += WEIGHT_FOOD / min(distancesToFood)

        return score
        # ----------------------------------------------------------------------- #
        

def scoreEvaluationFunction(currentGameState):
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
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        
    # ----------------------------------------------------------------------- #
    # is this a terminal state
    def isTerminal(self, state, depth, agent):
        return depth == self.depth or state.isWin() or state.isLose() or state.getLegalActions(agent) == 0

    # is this agent pacman
    def isPacman(self, state, agent):
        return agent % state.getNumAgents() == 0
    # ----------------------------------------------------------------------- #

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        # ----------------------------------------------------------------------- #
        def minimax(state, depth, agent):
            # is the agent is only pacman?
            if agent == state.getNumAgents():  
                # set next depth and number of agent
                nextDepth = depth + 1
                numberOfAgent = 0
                return minimax(state, nextDepth, numberOfAgent)  

            # is it the terminal? 
            if self.isTerminal(state, depth, agent):
                # the evaluation for bottom states
                return self.evaluationFunction(state) 

            # find the "best" state of the successors based on min or max
            successors = (minimax(state.generateSuccessor(agent, action), depth, agent + 1) for action in state.getLegalActions(agent))
            return (max if self.isPacman(state, agent) else min)(successors)

        # return the best of pacman's possible moves
        return max(gameState.getLegalActions(0), key = lambda x: minimax(gameState.generateSuccessor(0, x), 0, 1))
        # ----------------------------------------------------------------------- #
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        # ----------------------------------------------------------------------- #
        def dispatch(state, depth, agent, a=float("-inf"), b=float("inf")):
            if agent == state.getNumAgents():  # next depth
                depth += 1
                agent = 0

            # is it the terminal? 
            if self.isTerminal(state, depth, agent):  
                # return evaluation for bottom states
                return self.evaluationFunction(state), None

            if self.isPacman(state, agent):
                return getValue(state, depth, agent, a, b, float('-inf'), max)
            else:
                return getValue(state, depth, agent, a, b, float('inf'), min)
                
        def getValue(state, depth, agent, a, b, ms, mf):
            bestScore = ms
            bestAction = None

            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                score,_ = dispatch(successor, depth, agent + 1, a, b)
                bestScore, bestAction = mf((bestScore, bestAction), (score, action))

                if self.isPacman(state, agent):
                    if bestScore > b:
                        return bestScore, bestAction
                    a = mf(a, bestScore)
                else:
                    if bestScore < a:
                        return bestScore, bestAction
                    b = mf(b, bestScore)

            return bestScore, bestAction

        score, action = dispatch(gameState, 0, 0)
        return action
        # ----------------------------------------------------------------------- #

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        # ----------------------------------------------------------------------- #
        def expectimax(state, depth, agent):
            # is the agent is only pacman?
            if agent == state.getNumAgents():
                # set next depth and number of agent
                nextDepth = depth + 1
                numberOfAgent = 0
                return expectimax(state, nextDepth, numberOfAgent)

            # is it the terminal? 
            if self.isTerminal(state, depth, agent):
                # the evaluation for bottom states
                return self.evaluationFunction(state)  

            successors = [expectimax(state.generateSuccessor(agent, action), depth, agent + 1) for action in state.getLegalActions(agent)]

            # the best move for the pacman from the maximum successors
            if self.isPacman(state, agent):
                return max(successors)

            # since ghost's movement is pretty unpredictable 
            # so the average moves for the ghost is considering as their likely behavior for all 
            else:
                return sum(successors)/len(successors)

        # the best possible moves of the pacman's is return after all
        return max(gameState.getLegalActions(0), key = lambda x: expectimax(gameState.generateSuccessor(0, x), 0, 1))
        # ----------------------------------------------------------------------- #
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
    # ----------------------------------------------------------------------- #
    # get new position of pacman and food on the board
    # also get ghost state and the scared time
    newPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # Let us assume the food and the ghost weights are as -
    WEIGHT_FOOD = 10.0
    WEIGHT_GHOST = 10.0
    WEIGHT_EDIBLE_GHOST = 100.0

    # game score
    score = currentGameState.getScore()

    # distance to ghosts
    ghostScore = 0
    for ghost in newGhostStates:
        # calculate distance between closest ghost and the pacman
        distance = manhattanDistance(newPosition, newGhostStates[0].getPosition())
        if distance > 0:
            # update ghost value 
            # if ghost is scared -> go and and catch him -> point will be added
            # otherwise -> run away! -> else pacman will die -> reduce point 
            if ghost.scaredTimer > 0:  
                ghostScore += WEIGHT_EDIBLE_GHOST / distance
            else:  
                ghostScore -= WEIGHT_GHOST / distance
    
    # update score for ghost
    score += ghostScore

    # distance to closest food
    distancesToFood = [manhattanDistance(newPosition, food) for food in newFood.asList()]
    if len(distancesToFood):
        # add point when foods are eaten by the pacman
        score += WEIGHT_FOOD / min(distancesToFood)

    return score
    # ----------------------------------------------------------------------- #

# Abbreviation
better = betterEvaluationFunction

