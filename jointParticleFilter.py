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
        self.opponentStartingPos = self.getOpponentStartingPosition(gameState, self.opponents[0])  # TODO: Build out this function
        self.particles = []

        # self.setBothToStart()
        # initializeParticles

    def setBothToStart(self):
        for _ in range(self.numParticles):
            self.particles.append((self.opponentStartingPos, self.opponentStartingPos))  # initialize with tuple


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

        currentObservation = gameState.getCurrentObservation()
        currAgentPos = currentObservation.getAgentPosition(currAgent)
        opponentPos = [None for _ in range(4)]
        for opponent in self.opponents:
            opponentPos[opponent] = currentObservation.getAgentPosition(opponent)  # Might be None or a tuple
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
                    trueDistances = [util.manhattanDistance(p[i], currentObservation.getAgentPosition(self.team[0])),
                                     util.manhattanDistance(p[i], currentObservation.getAgentPosition(self.team[1]))]
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
        ghostPosition = gameState.getGhostPosition(ghostIndex + 1)
        actionDist = agent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPositions(self, gameState, ghostPositions):
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index+1] = game.AgentState(conf, False)
        return gameState