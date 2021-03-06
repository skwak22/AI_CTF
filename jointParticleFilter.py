import ghostAgents
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import itertools


def getLegalPositions(gameState):
    walls = set(gameState.getWalls().asList())
    legalPosList = []
    for i in range(gameState.getWalls().width):
        for j in range(gameState.getWalls().height):
            if (i, j) not in walls:
                legalPosList.append((i, j))
    return legalPosList

class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """



    def __init__(self, myteam, opponents, gameState, numParticles=200):
        self.setNumParticles(numParticles)
        self.particles = []
        # self.Agent1 = Agent1
        # self.Agent2 = Agent2
        self.agent1index = myteam[0]
        self.agent2index = myteam[1]
        self.gameState = gameState
        self.opponents = opponents
        self.height = gameState.getWalls().height
        self.width = gameState.getWalls().width
        self.startingPos1 = gameState.getAgentPosition(self.agent1index)# TODO: get position from agent state?
        self.startingPos2 = gameState.getAgentPosition(self.agent2index)
        self.legalPositions = getLegalPositions(gameState)
        self.opponentStartingPos1 = self.getOpponentStartingPosition(gameState, self.opponents[0])
        self.opponentStartingPos2 = self.getOpponentStartingPosition(gameState, self.opponents[1])
        self.setBothToStart()
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
            self.particles.append((self.opponentStartingPos1, self.opponentStartingPos2))  # initialize with tuple

    def initializeParticles(self):
        """
        Initialize particles to be at the starting position of the opponent's side
        (mirror of our pacman's start) if we are calling this for the first time
        else, initialize uniformly
        """
        cartProduct = list(itertools.product(self.legalPositions, repeat=2))
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


    def observeState(self, gameState, agentIndex):
        myPos = gameState.getAgentPosition(agentIndex)
        noisyDistances = gameState.getAgentDistances()
        # emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]
        # if len(noisyDistances) < self.opponents:
        #     return
        new_belief_distribution = util.Counter()

        belief_distribution = self.getBeliefDistribution()
        # for i in belief_distribution:
        #     new_belief_distribution[i] = belief_distribution[i] # initialize with old values
        enemyPos1 = gameState.getAgentPosition(self.getOpponents(gameState)[0])  # TODO: make team-general
        enemyPos2 = gameState.getAgentPosition(self.getOpponents(gameState)[1])
        for position, particle_weight in belief_distribution.iteritems():
            # print(particle_weight)
            for i in range(len(self.opponents)):
                true_distance = util.manhattanDistance(myPos, position[i])
                if gameState.getDistanceProb(true_distance, noisyDistances[self.opponents[i]]) > 0:
                    if new_belief_distribution[position] > 0:
                        new_belief_distribution[position] = new_belief_distribution[position] * gameState.getDistanceProb(true_distance, noisyDistances[self.opponents[i]]) * 1000000
                    else:
                        new_belief_distribution[position] = belief_distribution[position] * gameState.getDistanceProb(true_distance, noisyDistances[self.opponents[i]]) * 1000000  # the * 1000000 is necessary for success. Rounding errors?

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


    def elapseTime(self, gameState, agentIndex):
        if agentIndex ==0: # oppIndex is the index of the opp that is actually being updated
            oppIndex = 3
        else:
            oppIndex = agentIndex-1
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle) # A list of ghost positions

            newPosDist = self.getOppPosDistr(
                self.setEnemyPositions(gameState, oldParticle), oppIndex)
            newParticle[oppIndex/2] = util.sample(newPosDist)
            # now loop through and update each entry in newParticle...

            "*** YOUR CODE HERE ***"

            "*** END YOUR CODE HERE ***"
            newParticles.append(tuple(newParticle))
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
        print actions
        for a in actions:
            actionDist[a] += 1
        actionDist.normalize()
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = gameState.generateSuccessor(ghostIndex, action).getAgentState(ghostIndex).getPosition()
            dist[successorPosition] = prob
        return dist

    def getOppPosDistr(self, gameState, enemyIndex):
        enemyPos = gameState.getAgentPosition(enemyIndex)
        dist = util.Counter()

        if enemyPos is None:
            dist[self.opponentStartingPos1] = 1 #TODO: this is placeholder
            return dist
        actions = gameState.getLegalActions(agentIndex=enemyIndex)
        actionDist = util.Counter()
        for a in actions:
            # if a is "SOUTH":
            #     actionDist[a] +=100
            actionDist[a] += 1
        actionDist.normalize()

        for action, prob in actionDist.items():
            successorPosition = gameState.generateSuccessor(enemyIndex, action).getAgentState(enemyIndex).getPosition()
            dist[successorPosition] = prob
        return dist

    def setEnemyPositions(self, gameState, enemyPositions):
        for index, pos in enumerate(enemyPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index+1] = game.AgentState(conf, False)
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
    jointInference = None

    return [eval(first)(firstIndex,jointInference), eval(second)(secondIndex,jointInference)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, index, jointInference):
        CaptureAgent.__init__(self,index)
        self.jointInference = jointInference

    def registerInitialState(self, gameState):

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.legalPositions = self.getLegalPositions(gameState)
        if self.jointInference is None:
            self.jointInference = JointParticleFilter(self.getTeam(gameState), self.getOpponents(gameState),gameState)
            # self.jointInference.initialize(gameState, self.legalPositions, self)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        self.jointInference.observeState(gameState, self.index)

        self.jointInference.elapseTime(gameState, self.index)
        beliefs = self.jointInference.getBeliefDistribution()
        # print(beliefs)
        self.debugDraw(getLegalPositions(gameState),[0,0,0], clear=True)
        if self.red:
            for pos in beliefs:
                self.debugDraw([i for i in pos],[0,0,min(beliefs[pos]*5,1)],clear=False)
        else:
            for pos in beliefs:
                self.debugDraw([i for i in pos],[min(beliefs[pos]*5,1),0,0],clear=False)
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
