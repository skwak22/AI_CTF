import ghostAgents
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import itertools

class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions, captureAgent):
        "Stores information about the game, then initializes particles."

        self.captureAgent = captureAgent
        self.opponents = captureAgent.getOpponents(gameState)
        self.team = captureAgent.getTeam(gameState)
        self.numGhosts = len(self.opponents)
        self.legalPositions = legalPositions
        self.height = gameState.getWalls().height
        self.width = gameState.getWalls().width
        self.startingPos = gameState.getAgentPosition(self.team[0])
        self.opponentStartingPos1 = self.getOpponentStartingPosition(gameState, self.opponents[0])
        self.opponentStartingPos2 = self.getOpponentStartingPosition(gameState, self.opponents[1])
        self.particles = []

        self.setBothToStart()
        # initializeParticles

    def setBothToStart(self):
        for _ in range(self.numParticles):
            self.particles.append((self.opponentStartingPos1, self.opponentStartingPos2))  # initialize with tuple

    def initializeParticles(self):
        """
        Initialize particles to be at the starting position of the opponent's side
        (mirror of our pacman's start) if we are calling this for the first time
        else, initialize uniformly
        """
        cartProduct = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        random.shuffle(cartProduct)

        positionsLength = self.numParticles
        whole = positionsLength // len(cartProduct)
        remain = positionsLength % len(cartProduct)
        for i in range(whole):
            for c in cartProduct:
                self.particles.append(c)
        index = 0
        for i in range(remain):
            self.particles.append(cartProduct[index])
            index += 1


    def observeState(self, gameState, currAgent):

        opponentPos = [None for _ in range(4)]
        for opponent in self.opponents:
            opponentPos[opponent] = gameState.getAgentPosition(opponent)  # Might be None or a tuple
        noisyDistances = gameState.getAgentDistances()
        if len(noisyDistances) < self.numGhosts:
            return

        beliefDist = self.getBeliefDistribution()


        for i, pos in enumerate(opponentPos):
            if pos is not None:
                "Set all particles to be at this position"
                for particleIndex, particle in enumerate(self.particles[i]):
                    self.particles[particleIndex] = self.getParticleWithGhostAtTrueLocation(particle, i, pos)

        for i in self.opponents:
            if self.opponentJustEaten(i):
                for j in range(len(self.particles)):
                    pass

        for p in beliefDist:
            trueD = 1
            for i in range(self.numGhosts):
                if noisyDistances[i] is not None:
                    trueDistances = [util.manhattanDistance(p[i], gameState.getAgentPosition(self.team[0])),
                                     util.manhattanDistance(p[i], gameState.getAgentPosition(self.team[1]))]
                    trueD *= gameState.getDistanceProb(min(trueDistances), noisyDistances[self.opponents[i]])
            beliefDist[p] = trueD * beliefDist[p]


        if beliefDist.totalCount() == 0:

            self.initializeParticles()
            for i in self.opponents:
                if self.opponentJustEaten(i):
                    for j in range(len(self.particles)):
                        pass
            for i, pos in enumerate(opponentPos):
                if pos is not None:
                    for particleIndex, particle in enumerate(self.particles[i]):
                        self.particles[particleIndex] = self.getParticleWithGhostAtTrueLocation(particle, i, pos)

        else:
            for i in range(len(self.particles)):
                self.particles[i] = util.sample(beliefDist)


    def elapseTime(self, gameState):
        print "elapsing time"
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions
            # now loop through and update each entry in newParticle...

            currState = self.setGhostPositions(gameState, newParticle)
            for i in range(len(newParticle)):
                newPosDist = self.getPositionDistributionForGhost(currState, i, self.opponents[i])
                newParticle[i] = util.sample(newPosDist)

            newParticles.append(tuple(newParticle))
        self.particles = newParticles

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

    def getPositionDistributionForGhost(self, gameState, ghostIndex, agent):
        ghostPosition = gameState.getAgentPosition(ghostIndex)

        actions = gameState.getLegalActions(agentIndex=ghostIndex)

        actionDist = util.Counter()
        print actions
        for a in actions:
            actionDist[a] += 1
        actionDist.normalize()
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = gameState.generateSuccessor(ghostIndex, action).getAgentState(ghostIndex).getPosition()
            dist[successorPosition] = prob
        return dist

    def setGhostPositions(self, gameState, ghostPositions):
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index+1] = game.AgentState(conf, False)
        return gameState

jointInference = JointParticleFilter()


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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    jointInference = JointParticleFilter()

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.legalPositions = self.getLegalPositions(gameState)
        ReflexCaptureAgent.jointInference.initialize(gameState, self.legalPositions, self)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        self.jointInference.observeState(gameState, self.index)

        self.jointInference.elapseTime(gameState)

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
