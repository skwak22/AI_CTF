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