import networkx as nx
import numpy as np


def tst_complete():
    state=np.array([
        [1,2,2,3,4,4],
        [1,6,0,3,5,0],
        [0,6,0,3,0,0]
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


if __name__=='__main__':
    tst_complete()

