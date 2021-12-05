import ghostAgents
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import itertools
from ActualGeneralEvaluation import evalOffense

def getLegalPositions(gameState):
    walls = set(gameState.getWalls().asList())
    legalPosList = []
    for i in range(gameState.getWalls().width):
        for j in range(gameState.getWalls().height):
            if (i, j) not in walls:
                legalPosList.append((i, j))
    return legalPosList

class ParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """



    def __init__(self, myteam, myOpponent, gameState, numParticles=100):
        self.setNumParticles(numParticles)
        self.particles = []
        # self.Agent1 = Agent1
        # self.Agent2 = Agent2
        self.agent1index = myteam[0]
        self.agent2index = myteam[1]
        self.gameState = gameState
        self.myOpponent = myOpponent
        self.height = gameState.getWalls().height
        self.width = gameState.getWalls().width
        self.startingPos1 = gameState.getAgentPosition(self.agent1index)# TODO: get position from agent state?
        self.startingPos2 = gameState.getAgentPosition(self.agent2index)
        self.legalPositions = getLegalPositions(gameState)
        self.opponentStartingPos = self.getOpponentStartingPosition(gameState, myOpponent)
        self.setBothToStart()
        self.previousFood = None
        # self.initializeParticles()
        # self.initialize(gameState)
        # self.counter = gameState.


    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    # #def initialize(self, gameState, captureAgent):#,legalPositions):
    #     "Stores information about the game, then initializes particles."
    #
    #     self.captureAgent = captureAgent
    #     self.opponents = captureAgent.getOpponents(gameState)
    #     self.team = captureAgent.getTeam(gameState)
    #     self.numGhosts = len(self.opponents)
    #     # self.legalPositions = legalPositions
    #     self.height = gameState.getWalls().height
    #     self.width = gameState.getWalls().width
    #     self.startingPos = gameState.getAgentPosition(self.team[0])
    #     self.opponentStartingPos1 = self.getOpponentStartingPosition(gameState, self.opponents[0])
    #     self.opponentStartingPos2 = self.getOpponentStartingPosition(gameState, self.opponents[1])
    #     self.particles = []
    #
    #     self.setBothToStart()
        # initializeParticles

    def setBothToStart(self):
        for _ in range(self.numParticles):
            self.particles.append(self.opponentStartingPos)  # initialize with tuple

    def initializeParticles(self):
        """
        initialize uniformly
        """
        self.particles = []
        location_index = 0
        for i in range(self.numParticles): # check this is right
            self.particles.append(self.legalPositions[location_index])
            location_index = (location_index + 1) % len(self.legalPositions)



    def observeState(self, Agent, gameState, agentIndex, justMoved):

        myPos = gameState.getAgentPosition(agentIndex)
        noisyDistances = gameState.getAgentDistances()
        # emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]
        # if len(noisyDistances) < self.opponents:
        #     return
        new_belief_distribution = util.Counter()
        currentFood = set(Agent.getFoodYouAreDefending(gameState).asList())
        belief_distribution = self.getBeliefDistribution()
        enemyPos = None
        if justMoved and self.previousFood is not None: # if we're looking at the agent that just moved, check for eaten food
            eatenFood = [f for f in self.previousFood if f not in currentFood]
            if len(eatenFood) > 0:
                enemyPos = eatenFood[0]
        else:
            enemyPos = gameState.getAgentPosition(self.myOpponent)

        if enemyPos is not None:

            new_belief_distribution[enemyPos] = 1
        else:
            for position, particle_weight in belief_distribution.iteritems():
                # print(particle_weight)
                true_distance = util.manhattanDistance(myPos, position)
                if true_distance < 6 or position in currentFood: # if they're in a place where we'd see them guaranteed
                    new_belief_distribution[position] = 0
                else:

                    if gameState.getDistanceProb(true_distance, noisyDistances[self.myOpponent]) > 0:
                        if new_belief_distribution[position] > 0:
                            new_belief_distribution[position] = new_belief_distribution[position] * gameState.getDistanceProb(true_distance, noisyDistances[self.myOpponent]) * 1000000
                        else:
                            new_belief_distribution[position] = belief_distribution[position] * gameState.getDistanceProb(true_distance, noisyDistances[self.myOpponent]) * 1000000  # the * 1000000 is necessary for success. Rounding errors?

        if sum(new_belief_distribution.values()) == 0:  # handles case where there is 0 weight in particles
            self.initializeParticles()
            # for i in range(self.opponents): # ghost eaten
                # if noisyDistances[i] is None:
                    # for j in range(len(self.particles)):
                        # self.particles[j] = self.opponentJustEaten(self.particles[j], i)
        else:
            new_belief_distribution.normalize()
            self.particles = util.nSample(new_belief_distribution.values(), new_belief_distribution.keys(),
                                          self.numParticles)

        self.previousFood = set(Agent.getFoodYouAreDefending(gameState).asList())

        #
        # opponentPos = [None for _ in range(4)]
        # for opponent in self.opponents:
        #     opponentPos[opponent] = gameState.getAgentPosition(opponent)  # Might be None or a tuple
        # noisyDistances1 = gameState.getAgentDistances()
        # if len(noisyDistances1) < self.numGhosts:
        #     return
        # noisyDistances2 = gameState.getAgentDistances()
        #
        # beliefDist = self.getBeliefDistribution()
        #
        #
        # for i, pos in enumerate(opponentPos):
        #     if pos is not None:
        #         "Set all particles to be at this position"
        #         for particleIndex, particle in enumerate(self.particles[i]):
        #             self.particles[particleIndex] = self.getParticleWithGhostAtTrueLocation(particle, i, pos)
        #
        # for i in self.opponents:
        #     if self.opponentJustEaten(i):
        #         for j in range(len(self.particles)):
        #             pass
        #
        # for p in beliefDist:
        #     trueD = 1
        #     for i in range(self.numGhosts):
        #         if noisyDistances[i] is not None:
        #             trueDistances = [util.manhattanDistance(p[i], gameState.getAgentPosition(self.team[0])),
        #                              util.manhattanDistance(p[i], gameState.getAgentPosition(self.team[1]))]
        #             trueD *= gameState.getDistanceProb(min(trueDistances), noisyDistances[self.opponents[i]])
        #     beliefDist[p] = trueD * beliefDist[p]
        #
        #
        # if beliefDist.totalCount() == 0:
        #
        #     self.initializeParticles()
        #     for i in self.opponents:
        #         if self.opponentJustEaten(i):
        #             for j in range(len(self.particles)):
        #                 pass
        #     for i, pos in enumerate(opponentPos):
        #         if pos is not None:
        #             for particleIndex, particle in enumerate(self.particles[i]):
        #                 self.particles[particleIndex] = self.getParticleWithGhostAtTrueLocation(particle, i, pos)
        #
        # else:
        #     for i in range(len(self.particles)):
        #         self.particles[i] = util.sample(beliefDist)


    def elapseTime(self, observer, gameState):
        newParticles = []

        for oldParticle in self.particles:
            newPosDist = self.getOppPosDistr(gameState,observer, oldParticle, self.myOpponent)
            newParticle = util.sample(newPosDist)
            # now loop through and update each entry in newParticle...

            "*** YOUR CODE HERE ***"

            "*** END YOUR CODE HERE ***"
            newParticles.append(newParticle)

        self.particles = newParticles
        # print "elapsing time"
        # newParticles = []
        # for oldParticle in self.particles:
        #     newParticle = list(oldParticle)  # A list of ghost positions
        #     # now loop through and update each entry in newParticle...
        #
        #     currState = self.setGhostPositions(gameState, newParticle)
        #     for i in range(len(newParticle)):
        #         newPosDist = self.getPositionDistributionForGhost(currState, i, self.opponents[i])
        #         newParticle[i] = util.sample(newPosDist)
        #
        #     newParticles.append(tuple(newParticle))
        # self.particles = newParticles

    def getBeliefDistribution(self):
        "*** YOUR CODE HERE ***"

        beliefDist = util.Counter()
        for particle in self.particles:
            beliefDist[particle] += 1
        beliefDist.normalize()
        return beliefDist

    def getParticleWithGhostAtTrueLocation(self, particle, ghostIndex, position):
        particle = list(particle)
        particle[ghostIndex] = position
        return tuple(particle)

    def opponentJustEaten(self, i):
        # TODO: fill out
        return False

    def getOpponentStartingPosition(self, gameState, opponentIndex):
        return gameState.getInitialAgentPosition(opponentIndex)

    def getPositionDistributionForGhost(self, gameState, ghostIndex):
        ghostPosition = gameState.getAgentPosition(ghostIndex)

        actions = gameState.getLegalActions(agentIndex=ghostIndex)

        actionDist = util.Counter()
        for a in actions:
            actionDist[a] += 1
        actionDist.normalize()
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = gameState.generateSuccessor(ghostIndex, action).getAgentState(ghostIndex).getPosition()
            dist[successorPosition] = prob


        return dist

    def getOppPosDistr(self, gameState, observer, enemyPos, enemyIndex):
        # enemyPos = gameState.getAgentPosition(enemyIndex)
        walls = gameState.getWalls()
        legalMoves = ["Stop"]
        positions = {"Stop":enemyPos}
        if not walls[enemyPos[0]-1][enemyPos[1]]:
            legalMoves.append('West')
            positions['West'] = (enemyPos[0]-1,enemyPos[1])
        if not walls[enemyPos[0]+1][enemyPos[1]]:
            legalMoves.append('East')
            positions['East'] = (enemyPos[0]+1,enemyPos[1])

        if not walls[enemyPos[0]][enemyPos[1]-1]:
            legalMoves.append('South')
            positions['South'] =(enemyPos[0],enemyPos[1]-1)

        if not walls[enemyPos[0]][enemyPos[1]+1]:
            legalMoves.append('North')
            positions['North'] = (enemyPos[0],enemyPos[1]+1)

        dist = util.Counter()

        if enemyPos is None:
            dist[self.opponentStartingPos] = 1 #TODO: this is placeholder
            return dist
        # actions = gameState.getLegalActions(agentIndex=enemyIndex)
        actionDist = util.Counter() #TODO: particles when we eat them
        for a in legalMoves:
            # if a is "SOUTH":
            #     actionDist[a] +=100
            # score = evalOffense(observer, self.myOpponent, self.setEnemyPositions(gameState.deepCopy(), positions[a]))
            # print a
            # print(score)
            # if score < 0:
            #     actionDist[a] = 0.1
            # else:
            #     actionDist[a] = score
            actionDist[a] = 1

        actionDist.normalize()

        for action, prob in actionDist.items():
            # successorPosition = gameState.generateSuccessor(enemyIndex, action).getAgentState(enemyIndex).getPosition()
            successorPosition = positions[action]
            dist[successorPosition] = prob
        return dist

    def setEnemyPositions(self, gameState, enemyPosition):
        conf = game.Configuration(enemyPosition, game.Directions.STOP)
        gameState.data.agentStates[self.myOpponent] = game.AgentState(conf, False)
        return gameState



#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='ReflexCaptureAgent', second='ReflexCaptureAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    Inference1 = None
    Inference2 = None
    particleFilters = {}
    return [eval(first)(firstIndex,particleFilters), eval(second)(secondIndex,particleFilters)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, index, particleFilters):
        CaptureAgent.__init__(self, index)
        self.particleFilters = particleFilters

    def registerInitialState(self, gameState):
        self.deadEnds = findDeadEnds(gameState)

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.legalPositions = self.getLegalPositions(gameState)
        if len(self.particleFilters) ==0:
            self.particleFilters[self.getOpponents(gameState)[0]] = ParticleFilter(self.getTeam(gameState), self.getOpponents(gameState)[0],gameState)
            self.particleFilters[self.getOpponents(gameState)[1]] = ParticleFilter(self.getTeam(gameState), self.getOpponents(gameState)[1] ,gameState)

            # self.jointInference.initialize(gameState, self.legalPositions, self)

    def getAndUpdateBeliefs(self, gameState):

        enemyThatMoved = self.index - 1 # TODO: if you have the first turn there should be a special case here
        if enemyThatMoved == -1:
            enemyThatMoved = 3
        self.particleFilters[enemyThatMoved].elapseTime(self, gameState)

        otherEnemy = enemyThatMoved + 2
        if otherEnemy > 3:
            otherEnemy = otherEnemy - 4


        self.particleFilters[enemyThatMoved].observeState(self, gameState, self.index, True)
        self.particleFilters[otherEnemy].observeState(self, gameState, self.index, False)

        beliefs1 = self.particleFilters[self.getOpponents(gameState)[0]].getBeliefDistribution()
        beliefs2 = self.particleFilters[self.getOpponents(gameState)[1]].getBeliefDistribution()
        self.debugDraw(getLegalPositions(gameState), [0, 0, 0], clear=True)

        for pos in beliefs1:
            self.debugDraw([pos], [0, 0, min(beliefs1[pos] * 5, 1)], clear=False)
        for pos in beliefs2:
            self.debugDraw([pos], [min(beliefs2[pos] * 5, 1), 0, 0], clear=False)

        return beliefs1, beliefs2

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # self.getMazeDistance((1, 1), (1, 1))
        beliefs1, beliefs2 = self.getAndUpdateBeliefs(gameState)

        actions = gameState.getLegalActions(self.index)


        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


    def getLegalPositions(self, gameState):
        walls = set(gameState.getWalls().asList())
        legalPosList = []
        for i in range(gameState.getWalls().width):
            for j in range(gameState.getWalls().height):
                if (i,j) not in walls:
                    legalPosList.append((i,j))
        return legalPosList

def findDeadEnds(gameState):
    walls = gameState.deepCopy().getWalls()
    deadEnds = {}  # dict storing cells that are dead ends, and the direction to go to escape them
    for i in range(walls.width):
        for j in range(walls.height):
            if not walls[i][j]:  # check for out of bounds? prob not necessary, walls at border likely
                # wallcount = 0
                # path = [(i+1, j),(i,j+1), (i-1, j), (i,j-1)]
                # pathdirs = ['E','N', 'W', 'S']
                #
                # for adj in list(path):
                #     if walls[adj[0]][adj[1]]:
                #         index = path.index(adj)
                #         path.remove(adj)
                #         del pathdirs[index]
                # if len(path) is 1: # is a dead end
                #     deadEnds[(i,j)] = (path[0], pathdirs[0])
                deadEnds.update(checkIfDeadEnd(i,j,walls,None))



    for deadEnd in deadEnds:
        walls[deadEnd[0]][deadEnd[1]] = deadEnds[deadEnd][1]

    for i in range(walls.width):
        for j in range(walls.height):
            if walls[i][j] is False:
                walls[i][j] = ' '
    print(walls)
    return deadEnds

def checkIfDeadEnd(i, j, wallgrid, confirmed_dead):
    wallcount = 0
    deadEndDict = {}
    pathdirs = ['E', 'N', 'W', 'S']
    path = [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]
    if confirmed_dead is not None:
        index = path.index(confirmed_dead)
        path.remove(confirmed_dead)
        del pathdirs[index]

    for adj in list(path):
        if wallgrid[adj[0]][adj[1]]:
            index = path.index(adj)
            path.remove(adj)
            del pathdirs[index]
    if len(path) is 1:  # is a dead end
        deadEndDict[(i,j)] = (path[0], pathdirs[0])

        deadEndDict.update(checkIfDeadEnd(path[0][0],path[0][1],wallgrid, (i,j)))
    return deadEndDict