# sTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import itertools
import initializationfunctions


###########
# GLOBALS #
###########

# on initialization - mark dead end spots, mark the way towards the exit at them

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DefensiveReflexAgent', second='OffensiveReflexAgent'):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DefaultAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        # self.myTeamAgents = tuple(self.getTeam(gameState))
        # self.opponentTeamAgents = tuple(self.getOpponents(gameState))
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.deadEnds = initializationfunctions.findDeadEnds(gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # determine minimax depth
        DEPTH = 3

        # trying expectimax

        # find the known positions of all agents
        knownPositions = []
        enemiesSeen = []
        for i in range(4):
            position = gameState.getAgentPosition(i)
            knownPositions.append(position)
            if gameState.getAgentPosition(i) is not None and i in self.getOpponents(gameState):
                enemiesSeen.append(i)


        # If we can't see any enemies, just return the best action


        if len(enemiesSeen) == 0:
            actions = gameState.getLegalActions(self.index)

            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

            # You can profile your evaluation time by uncommenting these lines
            # start = time.time()
            successorStates = [gameState.generateSuccessor(self.index, action) for action in actions]
            values = []
            for i, successorState in enumerate(successorStates):
                isStop = False
                isReverse = False
                if actions[i] == "Stop":
                    isStop = True
                if actions[i] == reverse:
                    isReverse = True
                evalValue = self.evaluate(successorState, isReverse, isStop)
                values.append(evalValue)
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

        # if more than one enemy is seen, only run minimax on the one closest to our current agent

        if len(enemiesSeen) > 1:
            closest = 10000
            for enemy in enemiesSeen:
                mazeDist = self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(enemy))
                if mazeDist < closest:
                    closest = mazeDist
                    enemiesSeen = [enemy]

        def alpha_beta_search(gameState):

            # implement forward pruning, where we sort available actions and only consider the top n actions

            res_score = -10000000
            res_action = "Stop"
            init_beta = 10000000
            init_alpha = -100000000
            actions = gameState.getLegalActions(self.index)
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            actions = self.forwardPrune(gameState, actions, self.index, reverse, 2)
            for action in actions:
                successorState = gameState.generateSuccessor(self.index, action)

                score = min_value(successorState, enemiesSeen[0], 0, init_alpha, init_beta)
                if score > res_score:
                    res_score = score
                    res_action = action
                init_alpha = max(init_alpha, score)
            return res_action

        def max_value(gameState, player, depth, alpha, beta):
            actions = gameState.getLegalActions(player)
            if len(actions) == 0:
                return 0
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if depth == DEPTH or len(actions) == 0:
                return self.evaluate(gameState, False, False)
            v = -10000
            actions = self.forwardPrune(gameState, actions, self.index, reverse, 2)
            for action in actions:
                successorState = gameState.generateSuccessor(player, action)
                v = max(v, min_value(successorState,
                                     enemiesSeen[0], depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(gameState, player, depth, alpha, beta):
            actions = gameState.getLegalActions(player)
            if depth == DEPTH or actions == 0:
                return self.evaluate(gameState, False, False)
            v = 10000
            for action in actions:
                if len(actions) > 1 and action == "Stop":
                    continue
                successorState = gameState.generateSuccessor(player, action)
                if player == enemiesSeen[0]:
                    v = min(v, max_value(successorState,
                                         self.index, depth + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        return alpha_beta_search(gameState)


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

    def evaluate(self, gameState, isReverse, isStop):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, isReverse, isStop)
        weights = self.getWeights(gameState)
        return features * weights

    def getFeatures(self, gameState, isReverse, isStop):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = gameState
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def forwardPrune(self, gameState, actions, player, reverse, num):
        #our team
        #sort actions and return the top n actions
        try:
            score = []
            for action in actions:
                successor = gameState.generateSuccessor(player, action)
                evaluation = self.evaluate(successor, action == reverse, action == "Stop")
                score.append((evaluation, action))

            score.sort(reverse=True)
            if len(score) >= 3:
                toReturn = [i[1] for i in score[:num]]
                return toReturn
            else:
                return [i[1] for i in score[:1]]
        except Exception as e:
            print "Forward Prune exception"
            print e
            return ["Stop"]

class OffensiveReflexAgent(DefaultAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, isReverse, isStop):
        features = util.Counter()
        successor = gameState
        foodList = self.getFood(successor).asList()
        # features['successorScore'] = -len(foodList)
        features['successorScore'] = self.getScore(successor)
        successorState = successor.getAgentState(self.index)
        successorPos = successorState.getPosition()

        enemyPos1 = gameState.getAgentPosition(self.getOpponents(gameState)[0])  # TODO: make team-general
        enemyPos2 = gameState.getAgentPosition(self.getOpponents(gameState)[1])

        if enemyPos1 is not None:
            enemydist1 = self.getMazeDistance(successorPos, enemyPos1)
            if enemydist1 < 3:
                features['terror'] += 2 - enemydist1
            if enemydist1 is 0:
                features['distanceFromEnemy1'] = 1000.0
            else:
                features['distanceFromEnemy1'] = 1.0 / float(enemydist1)
            # print(features['distanceFromEnemy1'])
            if successorPos in self.deadEnds:
                features['distanceFromEnemy1'] *= 50
        if enemyPos2 is not None:
            enemydist2 = self.getMazeDistance(successorPos, enemyPos2)
            if enemyPos2 is not None and enemydist2 < 3:
                features['terror'] += 2 - enemydist2
            if enemydist2 is 0:
                features['distanceFromEnemy2'] = 1000.0
            else:
                features['distanceFromEnemy2'] = 1.0 / float(enemydist2)
            if successorPos in self.deadEnds:
                features['distanceFromEnemy2'] *= 50

        # beliefDistribution1 = util.Counter()
        # beliefDistribution2 = util.Counter()

        # features['terror'] = 0
        # for possiblePos in beliefDistribution1:
        #     features['distanceFromEnemy1'] += self.getMazeDistance(myPos, possiblePos) * beliefDistribution1[possiblePos]
        #     if self.getMazeDistance(myPos, possiblePos) < 2:
        #         features['terror'] +=beliefDistribution1[possiblePos]
        # for possiblePos in beliefDistribution2:
        #     features['distanceFromEnemy2'] += self.getMazeDistance(myPos, possiblePos) * beliefDistribution2[
        #         possiblePos]
        #     if self.getMazeDistance(myPos, possiblePos) < 2:
        #         features['terror'] += beliefDistribution2[possiblePos]

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(successorPos, food) for food in foodList])
            features['distanceToFood'] = 1 / float(minDistance)
        if successorState.isPacman:
            if gameState.isOnRedTeam:
                friendlyBorder = gameState.getWalls().width / 2 - 1  # TODO: ASSUMES RED IS ON LEFT
                minDistToHome = 99999999999
                for i in range(gameState.getWalls().height):
                    location = (friendlyBorder, i)
                    if not gameState.hasWall(location[0], location[1]) and self.getMazeDistance(successorPos,
                                                                                                location) < minDistToHome:
                        minDistToHome = self.getMazeDistance(successorPos, (friendlyBorder, i))
                features['distanceToHome'] = minDistToHome
            else:
                friendlyBorder = gameState.getWalls().width / 2
                for i in range(gameState.getWalls().height):
                    location = (friendlyBorder, i)
                    if not gameState.hasWall(location[0], location[1]) and self.getMazeDistance(successorPos,
                                                                                                location) < minDistToHome:
                        minDistToHome = self.getMazeDistance(successorPos, (friendlyBorder, i))
                features['distanceToHome'] = minDistToHome

            features['foodCarried'] = successorState.numCarrying

            features['BringingHomeBacon'] = float(features['foodCarried'] ** 2.0) / float(
                features['distanceToHome'] + 1)

        # print(features)

        # if(myState.isPacman):
        #     enemy_positions = gameState.getAgentDistances()
        #     if self.index is 2:
        #         features['distanceFromEnemy1'] = enemy_positions[2]
        #         features['distanceFromEnemy2'] = enemy_positions[3]
        #         if enemy_positions[2] < 3 or enemy_positions[3] < 3:
        #             features['terror'] = 1
        #     if self.index is 0:
        #         features['distanceFromEnemy1'] = enemy_positions[0]
        #         features['distanceFromEnemy2'] = enemy_positions[1]
        #         if enemy_positions[0] < 3 or enemy_positions[1] < 3:
        #             features['terror'] = 1

        # print(enemy_positions)
        #
        # pass

        return features

    def getWeights(self, gameState):
        return {'successorScore': 100, 'distanceToFood': 2, 'distanceToHome': 0, 'BringingHomeBacon': 5,
                'foodCarried': 5,
                'distanceFromEnemy1': -3, 'distanceFromEnemy2': -3, 'enemyCutOff': -10, 'terror': -100000000}


class DefensiveReflexAgent(DefaultAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, isReverse, isStop):
        features = util.Counter()
        successor = gameState

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        dists = None
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if isStop: features['stop'] = 1

        if isReverse: features['reverse'] = 1

        # Amount of food left

        ourFood = self.getFoodYouAreDefending(gameState).asList()

        features['numFood'] = len(ourFood)

        # Capsule Defense

        capsule = self.getCapsulesYouAreDefending(gameState)
        distanceToCapsule = 0

        if len(capsule) > 0:
            features['capsuleInPlay'] = 1
            distanceToCapsule = self.getMazeDistance(capsule[0], myPos)

        if distanceToCapsule == 0:
            features['capsuleProximity'] = 0
        else:
            features['capsuleProximity'] = 1 / distanceToCapsule

        # Scared tactics (run away reflexively when scared)

        scaredTimer = myState.scaredTimer
        if scaredTimer > 0:
            if dists is not None:
                if min(dists) < 1:
                    features['avoidWhenScared'] = 1

        if len(invaders) == 0:
            noisyDistances = gameState.getAgentDistances()
            opps = self.getOpponents(gameState)
            enemyNoisyDistances = [noisyDistances[opps[0]], noisyDistances[opps[1]]]
            allys = self.getTeam(gameState)
            allyNoisyDistances = [noisyDistances[allys[0]], allys[1]]
            # print allyNoisyDistances
            features['noisyClosestEnemy'] = min(enemyNoisyDistances)

        return features
    def getWeights(self, gameState):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,
                'numFood': 5, 'capsuleProximity': 10, 'avoidWhenScared': -10000, 'capsuleInPlay': 10,
                'noisyClosestEnemy': -100}

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
    Your initialization code goes here, if you need any.
    '''

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        actions = gameState.getLegalActions(self.index)

        '''
    You should change this in your own agent.
    '''

        return random.choice(actions)


# Features to consider - food, distance to food, distance to our own side, distance to enemies with consideration to whether they're between us and our side


class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions, opponents=None):
        "Stores information about the game, then initializes particles."
        if opponents is None:
            opponents = [1, 3]
        self.numGhosts = 2
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeParticles()
        self.opponents = opponents

    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.
        Each particle is a tuple of ghost positions. Use self.numParticles for
        the number of particles. You may find the `itertools` package helpful.
        Specifically, you will need to think about permutations of legal ghost
        positions, with the additional understanding that ghosts may occupy the
        same space. Look at the `itertools.product` function to get an
        implementation of the Cartesian product.
        Note: If you use itertools, keep in mind that permutations are not
        returned in a random order; you must shuffle the list of permutations in
        order to ensure even placement of particles across the board. Use
        self.legalPositions to obtain a list of positions a ghost may occupy.
        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """

        self.particles = []

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

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observeState(self, gameState):

        currentObservation = gameState.getCurrentObservation()
        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = gameState.getAgentDistances()
        if len(noisyDistances) < self.numGhosts:
            return

        # use getDistanceProb() instead of emissionModel

        beliefDist = self.getBeliefDistribution()
        for p in beliefDist:
            trueD = 1
            for i in range(self.numGhosts):
                if noisyDistances[i] is not None:
                    trueDistance = util.manhattanDistance(p[i], pacmanPosition)
                    trueD *= gameState.getDistanceProb(trueDistance, noisyDistances[i])
            beliefDist[p] = trueD * beliefDist[p]

        for ghostIndex, opponent in enumerate(self.opponents):
            if currentObservation.getAgentPosition(opponent) is not None:
                for i in range(len(self.particles)):
                    self.particles[i] = self.getParticleWithGhostAtTrueLocation(
                        currentObservation.getAgentPosition(opponent), ghostIndex)

        if beliefDist.totalCount() == 0:
            self.initializeParticles()
            for i in range(self.numGhosts):
                if noisyDistances[i] is None:
                    for j in range(len(self.particles)):
                        self.particles[j] = self.getParticleWithGhostInJail(self.particles[j], i)
        else:
            for i in range(len(self.particles)):
                self.particles[i] = util.sample(beliefDist)

    def getParticleWithGhostInJail(self, particle, ghostIndex):
        """
        Takes a particle (as a tuple of ghost positions) and returns a particle
        with the ghostIndex'th ghost in jail.
        """
        particle = list(particle)
        particle[ghostIndex] = self.getJailPosition(ghostIndex)
        return tuple(particle)

    def elapseTime(self, gameState):

        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions
            # now loop through and update each entry in newParticle...

            "*** YOUR CODE HERE ***"

            "*** END YOUR CODE HERE ***"
            newParticles.append(tuple(newParticle))
        self.particles = newParticles

    def getBeliefDistribution(self):
        "*** YOUR CODE HERE ***"

        beliefDist = util.Counter()
        for particle in self.particles:
            beliefDist[particle] += 1
        beliefDist.normalize()
        return beliefDist

    def getParticleWithGhostAtTrueLocation(self, particle, ghostIndex):
        particle = list(particle)

        # finish

        return tuple(particle)
