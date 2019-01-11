import networkx as nx
import numpy as np
import mcts


def tst_complete():
    state=np.array([
        [1,2,2,3,0,],
        [1,0,0,3,0,],
        [0,0,0,3,0,]
    ])

    row, col = state.shape
    print(state)

    # complete horizontal
    for x in range(0, col - 1):
        start = None
        for y in range(row - 1, -1, -1):
            if state[y][x] != 0 and start == None:
                start = y
                value = state[y][x]

            if start != None:
                if value != state[y][x]:
                    end = y
                    if (state[end + 1:start + 1, x + 1] == 0).all():
                        state[end + 1:start + 1, x + 1] = value

                    break
            if y == 0:
                assert start != None, 'find empty column which is not the last column'
                assert value == state[y][x], 'should break'
                if (state[0:start + 1, x + 1] == 0).all():
                    state[0:start + 1, x + 1] = value

    # complete vertically
    for x in range(0, col):
        for y in range(0, row):
            if state[y][x] == 0:
                value = state[y - 1][x]
                assert value != 0, 'vertical fill value=0'
                state[y:row, x] = value
                break

    assert (state[:] != 0).all(), 'still have empty space'
    print(state)
    return


def tst_children():
    Cons = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    )
    roomids = [1, 2, 3, 4, 5]
    design = mcts.MCTS(Cons, 2000)
    design.play()
    states = []
    rootnode = design.real_path[0]
    queue = [rootnode]
    while len(queue)>0:
        node = queue.pop(0)
        if node.expanded:
            if node.terminal is False:
                for child in node.children:
                    queue.append(child)
                    if child.type=='R':
                        states.append(child.state)


    states_path = states[0:63]
    vis = mcts.Visualisation(roomids, states_path, Cons, 'unknown')
    vis.vis_static()




if __name__=='__main__':
    #tst_complete()
    tst_children()

