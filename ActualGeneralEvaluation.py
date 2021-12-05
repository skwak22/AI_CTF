import util

#NOT GENERAL, for opponents!
def getFeaturesOffense(observer, opponent, gameState):
    features = util.Counter()
    foodList = observer.getFoodYouAreDefending(gameState).asList()
    # features['successorScore'] = -len(foodList)
    features['successorScore'] = -observer.getScore(gameState)
    oppState = gameState.getAgentState(opponent)
    oppPos = oppState.getPosition()

    enemyPos1 = gameState.getAgentPosition(observer.getTeam(gameState)[0])  # TODO: make team-general
    enemyPos2 = gameState.getAgentPosition(observer.getTeam(gameState)[1])

    if enemyPos1 is not None:
        enemydist1 = observer.getMazeDistance(oppPos, enemyPos1)
        if enemydist1 < 3:
            features['terror'] += 2 - enemydist1
        if enemydist1 is 0:
            features['distanceFromEnemy1'] = 1000.0
        else:
            features['distanceFromEnemy1'] = 1.0 / float(enemydist1)
        if oppPos in observer.deadEnds:
            features['distanceFromEnemy1'] *= 50
    if enemyPos2 is not None:
        enemydist2 = observer.getMazeDistance(oppPos, enemyPos2)
        if enemyPos2 is not None and enemydist2 < 3:
            features['terror'] += 2 - enemydist2
        if enemydist2 is 0:
            features['distanceFromEnemy2'] = 1000.0
        else:
            features['distanceFromEnemy2'] = 1.0 / float(enemydist2)
        if oppPos in observer.deadEnds:
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
        minDistance = min([observer.getMazeDistance(oppPos, food) for food in foodList])
        features['distanceToFood'] = 1 / float(minDistance)
    if oppState.isPacman:
        if gameState.isOnRedTeam:
            friendlyBorder = gameState.getWalls().width / 2 - 1  # TODO: ASSUMES RED IS ON LEFT
            minDistToHome = 99999999999
            for i in range(gameState.getWalls().height):
                location = (friendlyBorder, i)
                if not gameState.hasWall(location[0], location[1]) and observer.getMazeDistance(oppPos,
                                                                                                location) < minDistToHome:
                    minDistToHome = observer.getMazeDistance(oppPos, (friendlyBorder, i))
            features['distanceToHome'] = minDistToHome
        else:
            friendlyBorder = gameState.getWalls().width / 2
            for i in range(gameState.getWalls().height):
                location = (friendlyBorder, i)
                if not gameState.hasWall(location[0], location[1]) and observer.getMazeDistance(oppPos,
                                                                                                location) < minDistToHome:
                    minDistToHome = observer.getMazeDistance(oppPos, (friendlyBorder, i))
            features['distanceToHome'] = minDistToHome

        features['foodCarried'] = oppState.numCarrying

        features['BringingHomeBacon'] = float(features['foodCarried'] ** 2.0) / float(features['distanceToHome'] + 1)

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


def getWeightsOffense():
    return {'successorScore': 100, 'distanceToFood': 2, 'distanceToHome': 0, 'BringingHomeBacon': 5, 'foodCarried': 5,
            'distanceFromEnemy1': -3, 'distanceFromEnemy2': -3, 'enemyCutOff': -10, 'terror': -100000000}

def evalOffense(observer,opponent, gameState):
    features = getFeaturesOffense(observer,opponent,gameState)
    weights = getWeightsOffense()
    return features * weights