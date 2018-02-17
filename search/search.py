# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def generalGraphSearch(problem, structure):
    """
    defines a general algorithm to search a graph.
    
    :params: 
        ::(SearchProblem) problem:: 
            ::: the search problem
        ::(Stack, Queue, PriorityQueue) structure:: 
            ::: any data structure with .push() and .pop() methods
    :return:
        ::[(Stack, Queue, PriorityQueue)] paths:: 
            ::: the list of paths to the goal state, ignoring the first "Stop"
        ::[] path:: 
            ::: empty list if search fails
    """

    # push the data structure list in this format: [(state, actionTaken, cost)]
    # the pushed list into the structure: [(rootState, "Stop", 0), (newState, "North", 1)]
    structure.push([(problem.getStartState(), "Stop", 0)])

    # the list of visited nodes: empty list
    visited = []

    # While the structure is not empty, i.e. there are still elements to be searched,
    """
    Try to guess and figure out actual start, state and successors ...
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    while not structure.isEmpty():
        # get the path returned by the data structure's .pop() method
        paths = structure.pop()

        # the current state is the first element in the last tuple of the paths
        # i.e. [(rootState, "Stop", 0), (newState, "North", 1)][-1][0] = (newState, "North", 1)[0] = newState
        currentState = paths[-1][0]

        # if the current state is the goal state
        if problem.isGoalState(currentState):
            # return the actions to the goal state
            # which is the second element for each tuple in the paths, ignoring the first "Stop"
            return [path[1] for path in paths][1:]

        # if the current state has not been visited
        if currentState not in visited:
            # keep track on the current state as visited by appending to the visited list
            visited.append(currentState)

            # for all the successors of the current state
            for successor in problem.getSuccessors(currentState):
                # successor[0] = (state, action, cost)[0] = state
                # if the successor's state is unvisited
                if successor[0] not in visited:
                    # copy the parent's paths
                    successorPath = paths[:]
                    # set the paths of the successor node to the parent's paths + the successor node
                    successorPath.append(successor)
                    # push the successor's paths into the structure
                    structure.push(successorPath)

    # if search fails, return empty list 
    return []
    
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
    # ----------------------------------------------------------------------- #
    # Initialize an empty Stack
    stack = util.Stack()
    
    # DFS is general graph search with a Stack as the data structure
    return generalGraphSearch(problem, stack)
    # ----------------------------------------------------------------------- #

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
    # ----------------------------------------------------------------------- #
    # Initialize an empty Queue
    queue = util.Queue()

    # BFS is general graph search with a Queue as the data structure
    return generalGraphSearch(problem, queue)
    # ----------------------------------------------------------------------- #

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
    # ----------------------------------------------------------------------- #
    # The cost for UCS only the backward cost
    # get the actions in the path which are the second element for each tuple in the path, ignoring the first "Stop"
    # calculate the cost of the actions specific to the Problem using problem.getCostOfActions
    cost = lambda path: problem.getCostOfActions([x[1] for x in path][1:])

    # Construct an empty priority queue that sorts using this backwards cost
    priorityQueueWithoutHeuristic = util.PriorityQueueWithFunction(cost)

    # UCS is general graph search with the PriorityQueue sorting by the cost as the data structure
    return generalGraphSearch(problem, priorityQueueWithoutHeuristic)
    # ----------------------------------------------------------------------- #

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
    # ----------------------------------------------------------------------- #
    # The cost for a* seach is f(x) = g(x) + h(x)
    # The backward cost defined in UCS (problem.getCostOfActions([x[1] for x in path][1:])) is g(x)
    # The heuristic is h(x), heuristic(state, problem),
    # where state = path[-1][0], which is the first element in the last tuple of the path
    cost = lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)

    # Construct an empty priority queue that sorts using f(x)
    priorityQueueWithHeuristic = util.PriorityQueueWithFunction(cost)

    # A* is general graph search with the PriorityQueue sorting by the f(x) as the data structure
    return generalGraphSearch(problem, priorityQueueWithHeuristic)
    # ----------------------------------------------------------------------- #


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
