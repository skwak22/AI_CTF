# myTeam.py
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
import random, util
from game import Directions
import game
from util import nearestPoint
import itertools
import time
# on initialization - mark dead end spots, mark the way towards the exit at them
teamIsRed = None
TIME_ALLOWANCE = 0.4
DRAW = True
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

    global teamIsRed
    teamIsRed = isRed
    # The following line is an example only; feel free to change it.
    Inference1 = None
    Inference2 = None
    particleFilters = {}
    return [eval(first)(firstIndex,particleFilters), eval(second)(secondIndex,particleFilters)]


##########
# Agents #
##########

def getBeliefDistributionDummy():
    return {(1,1): .6, (1,2): .4}


class DefaultAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, index, particleFilters):
        CaptureAgent.__init__(self, index)
        self.particleFilters = particleFilters

    def registerInitialState(self, gameState):
        # self.myTeamAgents = tuple(self.getTeam(gameState))
        # self.opponentTeamAgents = tuple(self.getOpponents(gameState))
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.deadEnds = findDeadEnds(gameState)
        self.enemySide = getOpposideSidePositions(gameState, self.index)
        self.ourSide = getOurSidePositions(gameState, self.index)
        self.enemySideAsASet = set(self.enemySide)
        self.teammate = self.getTeam(gameState)
        self.teammate.remove(self.index)
        self.ourBorder = getOurBorder(gameState, self.index)
        self.borderForPatroling = getBorderForPatroling(self.ourBorder)
        self.borderPatrolCycle = 0
        # getBackHomePolicy maps to the closest home tile and the mazeDistance to that tile
        self.getBackHomePolicy = findBestPathsHome(self.ourBorder, self.enemySide, self)

        # find optimal defense tile by calculating the maze distances to every single tile on our side
        self.optimalDefenseTile = findOptimalDefenseTile(self.ourSide, self)
        self.optimalBorderTile = findOptimalBorderTile(self.ourSide, self.ourBorder, self)
        self.debugDraw(self.optimalBorderTile, [1,1,1])
        # print self.optimalDefenseTile
        # print self.optimalBorderTile
        self.lastEatenFood = self.optimalDefenseTile

        self.mostLeftOrRightFood = self.getMostExtremeFood(gameState)

        self.lastScore = self.getScore(gameState)
        self.lastTimeScoreChanged = 300

        self.foodEatenAtTime = 300

        self.isOffensiveAgent = self.getIsOffensiveAgent()

        # print self.isOffensiveAgent

        self.movesLeft = 300
        # self.debugDraw(self.optimalDefenseTile, [0, 0, 1])
        if len(self.particleFilters) ==0:
            self.particleFilters[self.getOpponents(gameState)[0]] = ParticleFilter(self.getTeam(gameState), self.getOpponents(gameState)[0],gameState)
            self.particleFilters[self.getOpponents(gameState)[1]] = ParticleFilter(self.getTeam(gameState), self.getOpponents(gameState)[1] ,gameState)


        # self.debugDraw(self.enemySide, [0,0,1])


        # self.debugDraw(self.borderForPatroling, [0,1,0])


        # self.debugDraw(self.ourSide, [0, 1, 0])

    def getAndUpdateBeliefs(self, gameState):

        enemyThatMoved = self.index - 1  # TODO: if you have the first turn there should be a special case here
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

        if(DRAW):
            for pos in beliefs1:
                self.debugDraw([pos], [0, 0, min(beliefs1[pos] * 5, 1)], clear=False)
            for pos in beliefs2:
                self.debugDraw([pos], [min(beliefs2[pos] * 5, 1), 0, 0], clear=False)

        return beliefs1, beliefs2

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        start = time.time()
        print(start)
        self.movesLeft -= 1
        beliefs1, beliefs2 = self.getAndUpdateBeliefs(gameState)


        if self.getScore(gameState) != self.lastScore:
            self.lastScore = self.getScore(gameState)
            self.lastTimeScoreChanged = self.movesLeft

        if self.lastTimeScoreChanged - self.movesLeft > 30:
            # we might be in a loop here, try to break it by introducing a random move (if we're losing)
            global teamIsRed
            if teamIsRed and self.getScore(gameState) < 0:
                # we are losing on red
                actions = gameState.getLegalActions(self.index)
                print('final time')
                print(time.time() - start)
                return random.choice(actions)
            elif not teamIsRed and self.getScore(gameState) > 0:
                # we are losing on blue
                actions = gameState.getLegalActions(self.index)
                print('final time')
                print(time.time() - start)
                return random.choice(actions)
            self.lastTimeScoreChanged = self.movesLeft

        # determine minimax depth
        DEPTH = 4

        # find the known positions of all agents
        knownPositions = []
        enemiesSeen = []
        for i in range(4):
            position = gameState.getAgentPosition(i)
            knownPositions.append(position)
            if gameState.getAgentPosition(i) is not None and i in self.getOpponents(gameState):
                enemiesSeen.append(i)

        enemyNoisyDistances = gameState.getAgentDistances()
        # print gameState
        # print "agent "+ str(self.index) + str(enemyNoisyDistances)

        # If we can't see any enemies, just return the best action

        # {(1,1): .01}

        # TODO: Modify N Value

        N = 1
        beliefDist = getBeliefDistributionDummy()
        distributions = []
        for position, value in beliefDist.items():
            distributions.append((value, position))
        # print distributions
        distributions.sort(reverse=True)

        top_n = [i[1] for i in distributions[:N]]

        if len(enemiesSeen) >= 1:
            closest = 10000
            for enemy in enemiesSeen:
                mazeDist = self.getMazeDistance(gameState.getAgentPosition(self.index),
                                                gameState.getAgentPosition(enemy))
                if mazeDist < closest:
                    closest = mazeDist
                    enemiesSeen = [enemy]

            # discard enemy if irrelevant (more than 8 distance away from agent)
            if self.getMazeDistance(gameState.getAgentPosition(enemiesSeen[0]),
                                    gameState.getAgentPosition(self.index)) > 8:
                enemiesSeen = []

        if len(enemiesSeen) == 0:
            actions = gameState.getLegalActions(self.index)

            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

            # You can profile your evaluation time by uncommenting these lines
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
                print('final time')
                print(time.time() - start)
                return bestAction
            print('final time')
            print(time.time() - start)
            return random.choice(bestActions)

        # if more than one enemy is seen, only run minimax on the one closest to our current agent

        def alpha_beta_search(gameState, startTime):
            print('hi')
            # implement forward pruning, where we sort available actions and only consider the top n actions

            actions = gameState.getLegalActions(self.index)

            res_score = -10000000
            res_action = actions[-1]
            init_beta = 10000000
            init_alpha = -100000000
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            actions = self.forwardPrune(gameState, actions, self.index, reverse, 2) #TODO: this is sorted right?
            for action in actions:
                if len(actions) > 1 and action == "Stop": #TODO: is this saying never stop?
                    continue

                successorState = gameState.generateSuccessor(self.index, action)

                score = min_value(successorState, enemiesSeen[0], 0, init_alpha, init_beta,startTime)
                if score > res_score:
                    res_score = score
                    res_action = action
                init_alpha = max(init_alpha, score)
                if (time.time() - startTime) > 0.9 * TIME_ALLOWANCE: #if less than 0.1 seconds left
                    print(time.time() - startTime)

                    break
            print('final time')
            print(time.time() - start)
            return res_action

        def max_value(gameState, player, depth, alpha, beta, startTime):
            actions = gameState.getLegalActions(player)
            if len(actions) == 0:
                return 0
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if depth == DEPTH or len(actions) == 0:
                return self.evaluate(gameState, False, False)
            v = -10000
            actions = self.forwardPrune(gameState, actions, self.index, reverse, 2)
            for action in actions:
                if len(actions) > 1 and action == "Stop":
                    continue
                successorState = gameState.generateSuccessor(player, action)
                v = max(v, min_value(successorState,
                                     enemiesSeen[0], depth, alpha, beta, startTime))

                if v > beta:
                    return v
                if (time.time() - startTime) > 0.9 * TIME_ALLOWANCE: #if less than 0.1 seconds left
                    print(time.time() - startTime)

                    break
                alpha = max(alpha, v)
            return v

        def min_value(gameState, player, depth, alpha, beta,startTime):
            actions = gameState.getLegalActions(player)
            if depth == DEPTH or actions == 0:
                return self.evaluate(gameState, False, False)
            v = 10000
            actions = self.forwardPruneTheEnemy(gameState, actions, player, 2)
            for action in actions:
                if len(actions) > 1 and action == "Stop":
                    continue
                successorState = gameState.generateSuccessor(player, action)
                if player == enemiesSeen[0]:
                    v = min(v, max_value(successorState,
                                         self.index, depth + 1, alpha, beta, startTime))
                if v < alpha:
                    return v
                if (time.time() - startTime) > 0.9 * TIME_ALLOWANCE: #if less than 0.1 seconds left
                    print(time.time() - startTime)
                    break
                beta = min(beta, v)

            return v

        answer = alpha_beta_search(gameState, start)
        print('final time')
        print(time.time() - start)
        return answer

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
        # Adopt a desperation play strategy
        # "Pull the goalie" if we're losing and turtle if we're winning

        if self.movesLeft < 80:
            global teamIsRed
            if (teamIsRed and self.getScore(gameState) > 0) or (not teamIsRed and self.getScore(gameState) < 0):
                features = self.getFeaturesDefense(gameState, isReverse, isStop)
                weights = self.getWeightsDefensive(gameState)
            elif self.getScore(gameState) == 0:
                features = self.getFeatures(gameState, isReverse, isStop)
                weights = self.getWeights(gameState)
            else:
                features = self.getFeaturesOffense(gameState, isReverse, isStop)
                weights = self.getWeightsOffensive(gameState)

        else:
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
        # our team
        # sort actions and return the top n actions
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

    def forwardPruneTheEnemy(self, gameState, actions, player, num):
        # enemy team
        # sort actions and return the bottom n actions
        score = []
        for action in actions:
            successor = gameState.generateSuccessor(player, action)
            evaluation = self.evaluate(successor, False, False)
            score.append((evaluation, action))

        score.sort()
        if len(score) >= 3:
            toReturn = [i[1] for i in score[:num]]
            return toReturn
        else:
            return [i[1] for i in score[:1]]

    def getIsOffensiveAgent(self):
        return True

    def getMostExtremeFood(self, gameState):

        # Get the most extreme food's x coordinate (furthest away from border)
        global teamIsRed
        if teamIsRed:
            mostExtreme = 100000
            for food in self.getFoodYouAreDefending(gameState).asList():
                mostExtreme = min(mostExtreme, food[0])
        else:
            mostExtreme = 0
            for food in self.getFoodYouAreDefending(gameState).asList():
                mostExtreme = max(mostExtreme, food[0])
        return mostExtreme

    # DEFENSE
    def getFeaturesDefense(self, gameState, isReverse, isStop):
        features = util.Counter()
        successor = gameState

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        teammatePos = successor.getAgentState(self.teammate[0]).getPosition()

        if self.movesLeft < 250:
            if self.getMazeDistance(teammatePos, myPos) <= 3:
                features['closeToTeammate'] = 1

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
        foodLen = len(ourFood)
        features['numFood'] = foodLen


        # Charge to last eaten food

        previousState = self.getPreviousObservation()
        if previousState is not None:
            prevFoodList = set(self.getFoodYouAreDefending(previousState).asList())
            eatenFood = prevFoodList - set(ourFood)
            if len(eatenFood) > 0:
                for i in eatenFood:
                    self.lastEatenFood = i
                    self.foodEatenAtTime = self.movesLeft

        if self.foodEatenAtTime - self.movesLeft < 6:
        # self.debugDraw(self.lastEatenFood, [1,1,0], clear=True)
            features['distanceToLastEatenFood'] = 1/float(self.getMazeDistance(self.lastEatenFood, myPos)+1)


        # Distance to pieces of food
        if foodLen > 0:
            distancesToRandomFood = []
            # self.debugDraw([(0,0)], [1, 1, 1], clear=True)
            for _ in range(foodLen):
                randomInt = random.randint(0,foodLen-1)
                distancesToRandomFood.append(self.getMazeDistance(ourFood[randomInt], myPos))
                # self.debugDraw(ourFood[randomInt], [1,1,1])
            avgDistance = sum(distancesToRandomFood)/float(foodLen)

            features['distanceToFood'] = 1/(avgDistance+4.0)

            # print features['distanceToFood']

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

        # Distance to optimal defense tile

        distanceToOptimalDefenseTile = self.getMazeDistance(self.optimalBorderTile, myPos)
        if distanceToOptimalDefenseTile == 0:
            features['optimalDefenseTile'] = 1
        else:
            features['optimalDefenseTile'] = 1/(float(distanceToOptimalDefenseTile)+29)


        # Stay on the correct side of the most extreme food
        global teamIsRed
        if teamIsRed:
            # print self.mostLeftOrRightFood
            if myPos[0] < self.mostLeftOrRightFood:
                features['wrongSideOfFood'] = 1
        else:
            if myPos[0] > self.mostLeftOrRightFood:
                features['wrongSideOfFood'] = 1

        # Patrol border (ONLY THE ORIGINAL DEFENSIVE AGENT WILL PATROL)
        if not self.isOffensiveAgent:
            borderLen = len(self.borderForPatroling)
            if self.borderForPatroling[self.borderPatrolCycle] == myPos:
                self.borderPatrolCycle = (self.borderPatrolCycle + 1) % borderLen
            features['borderPatrol'] = 1/float(self.getMazeDistance(self.borderForPatroling[self.borderPatrolCycle], myPos)+1)


        return features

    def getWeightsDefensive(self, gameState):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,
                'numFood': 5, 'capsuleProximity': 10, 'avoidWhenScared': -10000, 'capsuleInPlay': 10,
                'noisyClosestEnemy': -1, 'optimalDefenseTile': 2, 'distanceToFood': 1, 'distanceToLastEatenFood': 10000,
                'closeToTeammate': -1, 'wrongSideOfFood': -3, 'borderPatrol': 3}


    # OFFENSE
    def getFeaturesOffense(self, gameState, isReverse, isStop):
        features = util.Counter()
        successor = gameState
        foodList = self.getFood(successor).asList()
        # features['successorScore'] = -len(foodList)
        features['successorScore'] = self.getScore(successor)
        successorState = successor.getAgentState(self.index)
        successorPos = successorState.getPosition()

        enemyPos1 = gameState.getAgentPosition(self.getOpponents(gameState)[0])  # TODO: make team-general
        enemyPos2 = gameState.getAgentPosition(self.getOpponents(gameState)[1])

        scaredTimer = successorState.scaredTimer
        teammatePos = successor.getAgentState(self.teammate[0]).getPosition()

        if self.movesLeft < 10 and successorState.isPacman:
            # Astar search back home
            features['bookItHome'] = 1/float(self.getBackHomePolicy[successorPos][1] + 1)

        global teamIsRed
        if teamIsRed:
            if successorPos[0] >= self.ourBorder[0][0]:
                features['atOrBeyondBorder'] = 1
        else:
            if successorPos[0] <= self.ourBorder[0][0]:
                features['atOrBeyondBorder'] = 1


        if self.movesLeft < 250:
            if self.getMazeDistance(teammatePos, successorPos) <= 3:
                features['closeToTeammate'] = 1

        # print scaredTimer



        if enemyPos1 is not None:
            if (teamIsRed and enemyPos1[0] > self.ourBorder[0][0]) or (not teamIsRed and enemyPos1[0] < self.ourBorder[0][0]):


                enemydist1 = self.getMazeDistance(successorPos, enemyPos1)
                if enemydist1 < 3 and scaredTimer < 4:
                    features['terror'] += 2 - enemydist1
                if enemydist1 is 0:
                    features['distanceFromEnemy1'] = 1000.0
                else:
                    features['distanceFromEnemy1'] = 1.0 / float(enemydist1)
                # print(features['distanceFromEnemy1'])
                if successorPos in self.deadEnds:
                    features['distanceFromEnemy1'] *= 50
            if enemyPos2 is not None:
                if (teamIsRed and enemyPos2[0] > self.ourBorder[0][0]) or (
                        not teamIsRed and enemyPos2[0] < self.ourBorder[0][0]):
                    enemydist2 = self.getMazeDistance(successorPos, enemyPos2)
                    if enemyPos2 is not None and enemydist2 < 3 and scaredTimer < 4:
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

    def getWeightsOffensive(self, gameState):
        return {'successorScore': 100, 'distanceToFood': 2, 'distanceToHome': 0, 'BringingHomeBacon': 5,
                'foodCarried': 5,
                'distanceFromEnemy1': -3, 'distanceFromEnemy2': -3, 'enemyCutOff': -10, 'terror': -100000000,
                'closeToTeammate': -1, 'bookItHome': 100, 'atOrBeyondBorder': 3}



class OffensiveReflexAgent(DefaultAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, isReverse, isStop):
        return self.getFeaturesOffense(gameState, isReverse, isStop)

    def getWeights(self, gameState):
        return self.getWeightsOffensive(gameState)

    def getIsOffensiveAgent(self):
        return True

class DefensiveReflexAgent(DefaultAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, isReverse, isStop):
        return self.getFeaturesDefense(gameState, isReverse, isStop)

    def getWeights(self, gameState):
        return self.getWeightsDefensive(gameState)

    def getIsOffensiveAgent(self):
        return False


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


class ParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """



    def __init__(self, myteam, myOpponent, gameState, numParticles=50):
        self.setNumParticles(numParticles)
        self.particles = []
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

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def setBothToStart(self):
        for _ in range(self.numParticles):
            self.particles.append(self.opponentStartingPos)  # initialize with tuple

    def initializeParticles(self):
        """
        initialize uniformly
        """
        self.particles = []
        legalPositionsLen = len(self.legalPositions)
        for i in range(self.numParticles): # check this is right
            self.particles.append(self.legalPositions[random.randint(0,legalPositionsLen-1)])



    def observeState(self, Agent, gameState, agentIndex, justMoved):

        myPos = gameState.getAgentPosition(agentIndex)
        noisyDistances = gameState.getAgentDistances()
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
        else:
            new_belief_distribution.normalize()
            self.particles = util.nSample(new_belief_distribution.values(), new_belief_distribution.keys(),
                                          self.numParticles)

        self.previousFood = set(Agent.getFoodYouAreDefending(gameState).asList())



    def elapseTime(self, observer, gameState):
        newParticles = []

        for oldParticle in self.particles:
            newPosDist = self.getOppPosDistr(gameState,observer, oldParticle, self.myOpponent)
            newParticle = util.sample(newPosDist)
            # now loop through and update each entry in newParticle...
            newParticles.append(newParticle)

        self.particles = newParticles

    def getBeliefDistribution(self):

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
            score = evalOpponent(observer, self.myOpponent, self.setEnemyPositions(gameState.deepCopy(), positions[a]), enemyIndex)
            # print a
            # print(score)
            if score < 0:
                actionDist[a] = 0.1
            else:
                actionDist[a] = score
            # actionDist[a] = 1

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

def evalOpponent(observer, opponent, gameState, enemyIndex):
    opponentPos = gameState.getAgentState(enemyIndex).getPosition()
    ourTeamPos = [gameState.getAgentState(observer.getTeam(gameState)[0]).getPosition(),gameState.getAgentState(observer.getTeam(gameState)[1]).getPosition()]
    if gameState.isOnRedTeam(enemyIndex):
        if opponentPos[0] > gameState.getWalls().width//2:
            foodDistance = []
            for food in observer.getFoodYouAreDefending(gameState).asList():
                foodDistance.append(observer.getMazeDistance(food,opponentPos))
            eval = 100/(sum(foodDistance)+1.0)
            eval += 1000/(len(foodDistance)+1.0)

            if (min(observer.getMazeDistance(opponentPos, ourTeamPos[0]),
                                    observer.getMazeDistance(opponentPos, ourTeamPos[1]))) < 2:
                eval -= 100000
            return eval
        else:
            return 100/float(min(observer.getMazeDistance(opponentPos, ourTeamPos[0]), observer.getMazeDistance(opponentPos, ourTeamPos[1]))+1)
    else:
        if opponentPos[0] < gameState.getWalls().width//2:
            foodDistance = []
            for food in observer.getFoodYouAreDefending(gameState).asList():
                foodDistance.append(observer.getMazeDistance(food,opponentPos))
            eval = 100/(sum(foodDistance)+1.0)
            eval += 1000/(len(foodDistance)+1.0)

            if (min(observer.getMazeDistance(opponentPos, ourTeamPos[0]),
                                    observer.getMazeDistance(opponentPos, ourTeamPos[1]))) < 2:
                eval -= 100000
            return eval
        else:
            eval = 1000/float(min(observer.getMazeDistance(opponentPos, ourTeamPos[0]), observer.getMazeDistance(opponentPos, ourTeamPos[1]))+1)
            # print eval
            return eval
    return 0

############################
# Initialization Functions #
############################

def findDeadEnds(gameState):
    walls = gameState.getWalls()
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
                deadEnds.update(checkIfDeadEnd(i, j, walls, None))

    for deadEnd in deadEnds:
        walls[deadEnd[0]][deadEnd[1]] = deadEnds[deadEnd][1]

    for i in range(walls.width):
        for j in range(walls.height):
            if walls[i][j] is False:
                walls[i][j] = ' '
    # print(walls)
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
        deadEndDict[(i, j)] = (path[0], pathdirs[0])

        deadEndDict.update(checkIfDeadEnd(path[0][0], path[0][1], wallgrid, (i, j)))
    return deadEndDict


def getOpposideSidePositions(gameState, playerIndex):
    halfwayWidth = gameState.getWalls().width // 2
    rightSide = getLegalPositionsOneSide(gameState, halfwayWidth, gameState.getWalls().width)
    leftSide = getLegalPositionsOneSide(gameState, 0, halfwayWidth)

    if gameState.getInitialAgentPosition(playerIndex)[0] > halfwayWidth:
        return leftSide
    else:
        return rightSide


def getOurSidePositions(gameState, playerIndex):
    halfwayWidth = gameState.getWalls().width // 2
    rightSide = getLegalPositionsOneSide(gameState, halfwayWidth, gameState.getWalls().width)
    leftSide = getLegalPositionsOneSide(gameState, 0, halfwayWidth)

    if gameState.getInitialAgentPosition(playerIndex)[0] < halfwayWidth:
        return leftSide
    else:
        return rightSide


def getOurBorder(gameState, playerIndex):
    halfwayWidth = gameState.getWalls().width // 2

    if gameState.getInitialAgentPosition(playerIndex)[0] > halfwayWidth:
        return getLegalPositionsOneSide(gameState, halfwayWidth, halfwayWidth + 1)

    else:
        return getLegalPositionsOneSide(gameState, halfwayWidth - 1, halfwayWidth)

def getBorderForPatroling(border):
    newBorder = border[:]
    newBorder.sort()

    return newBorder[::3]

def findBestPathsHome(border, oppositeGrid, agent):
    closestBorderTiles = util.Counter()
    for legalPosition in oppositeGrid:
        closestDistance = 10000
        bestTile = None
        for borderTile in border:
            distance = agent.getMazeDistance(borderTile, legalPosition)
            if distance < closestDistance:
                closestDistance = distance
                bestTile = borderTile

        closestBorderTiles[legalPosition] = [bestTile, closestDistance]

    return closestBorderTiles


def findOptimalDefenseTile(ourSideGrid, agent):
    lowestSoFar = 100000
    bestDefenseTile = None
    n = len(ourSideGrid) - 1

    for legalPosition in ourSideGrid:
        sumOfDistances = 0
        for legalPosition2 in ourSideGrid:
            if legalPosition2 == legalPosition:
                continue
            sumOfDistances += agent.getMazeDistance(legalPosition, legalPosition2)
        if sumOfDistances / n < lowestSoFar:
            lowestSoFar = sumOfDistances / n
            bestDefenseTile = legalPosition
    # print bestDefenseTile

    return bestDefenseTile

def findOptimalBorderTile(ourSideGrid, border, agent):
    lowestSoFar = 100000
    bestDefenseTile = None
    n = len(ourSideGrid) - 1

    for legalPosition in ourSideGrid:
        sumOfDistances = 0
        for legalPosition2 in border:
            if legalPosition2 == legalPosition:
                continue
            sumOfDistances += agent.getMazeDistance(legalPosition, legalPosition2)
        if sumOfDistances / n < lowestSoFar:
            lowestSoFar = sumOfDistances / n
            bestDefenseTile = legalPosition
    # print bestDefenseTile

    return bestDefenseTile


#########
# UTILS #
#########
def getLegalPositions(gameState, width=None, height=None):
    if width is None:
        width = gameState.getWalls().width
    if height is None:
        height = gameState.getWalls().height
    walls = set(gameState.getWalls().asList())
    legalPosList = []
    for i in range(width):
        for j in range(height):
            if (i, j) not in walls:
                legalPosList.append((i, j))
    return legalPosList


def getLegalPositionsOneSide(gameState, widthStart, widthEnd):
    walls = set(gameState.getWalls().asList())
    legalPosList = []
    for i in range(widthStart, widthEnd):
        for j in range(gameState.getWalls().height):
            if (i, j) not in walls:
                legalPosList.append((i, j))
    return legalPosList
