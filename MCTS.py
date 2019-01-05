
import numpy as np
from copy import deepcopy



class Node(object):
    def __init__(self, state, remain_rooms, init_Q, info, next_positions):
        """

        :param state: the state matrix of this node
        :param remain_rooms: the remaining_rooms to put into this state
        :param init_Q: initial Q value
        :param info: (type, value)
                    if type is 'R', value is None
                    if type is 'X', value is selected end_x
                    if type is 'Y', value is (y,x)
        :param next_positions: ( (pos_y, pos_x), (original_y_intervals, original_x_intervals) )
        """
        self.state = state
        self.row, self.col = self.state.shape
        self.remain_rooms = remain_rooms
        self.ini_Q = init_Q
        self.type = info[0]
        self.value = info[1]
        (self.pos_y, self.pos_x), (self.ori_y_intervals, self.ori_x_intervals) = next_positions

        self.N = 0
        self.Q = init_Q
        self.W = 0

    def expand(self):
        if self.type in ['X','Y']:
            self.terminal = False
            self.fetch_children()
        else:
            assert self.type=='R', 'unknown node type'
            if len(self.remain_rooms) == 0:
                self.terminal = True
                self.complete()
            else:
                self.terminal = False
                self.fetch_children()

        return

    def find_next_position(self):
        for y in range(0, self.row):
            for x in range(0, self.col):
                if self.state[y][x] == 0:
                    return y,x

        raise TypeError('can not find next empty space')


    def find_x_intervals(self):
        intervals = [self.pos_x]

        # get two end intervals
        for x in range(self.pos_x, self.col):
            if self.state[self.pos_y][x] != 0:
                intervals.append(x)
                break
        if len(intervals) == 1:
            intervals.append(self.col)
        else:
            assert len(intervals) == 2, 'x first interval check should be 2'

        # get upper medium intervals
        if self.pos_y != 0:  # not upper border
            sign = self.state[self.pos_y - 1][self.pos_x]
            for x in range(self.pos_x, intervals[1]):
                if sign != self.state[self.pos_y - 1][x]:
                    sign = self.state[self.pos_y - 1][x]
                    intervals.insert(-1, x)

        return intervals

    def find_y_intervals(self):
        intervals = set()
        intervals.add(self.pos_y)
        intervals.add(self.row)
        edges= []
        for x in range(0, self.col):
            for y in range(0, self.row):
                if self.state[y][x]==0:
                    edges.append(y)
                    break
                if y == self.row - 1:
                    edges.append(self.row)

        edges=np.array(edges)
        assert len(edges) == self.col, 'edge size not match self.col'
        assert self.pos_x == edges.argmin(), 'minimum position is not pos_x?'
        assert self.pos_y == np.min(edges), 'minimum position is not pos_y?'
        max_left = self.pos_y
        for i in range(self.pos_x,-1,-1):
            if edges[i] > max_left:
                max_left = edges[i]
                intervals.add(max_left)
        max_right = self.pos_y
        for i in range(self.pos_x, self.col):
            if edges[i] > max_right:
                max_right = edges[i]
                intervals.add(max_right)

        intervals = list(intervals)
        intervals.sort()
        return intervals


    def fetch_children(self):
        self.children = []

        if self.type == 'R':
            self.pos_y, self.pos_x = self.find_next_position()
            self.ori_x_intervals = self.find_x_intervals()
            self.ori_y_intervals = self.find_y_intervals()
            xs, x_states = self.choose_x()  # get all possible x states
            for x, x_state in zip(xs, x_states):
                child = Node(x_state,self.remain_rooms,self.ini_Q, ('X',x), ((self.pos_y, self.pos_x),(self.ori_y_intervals,self.ori_x_intervals)))
                self.children.append(child)

        elif self.type == 'X':
            ys, y_states = self.choose_y()
            for y, y_state in zip(ys,y_states):
                child = Node( y_state, self.remain_rooms, self.ini_Q, ('Y',(y,self.value)), ((self.pos_y, self.pos_x), (self.ori_y_intervals,self.ori_x_intervals)) )
                self.children.append(child)

        elif self.type == 'Y':
            for id in self.remain_rooms:
                remain_rooms = [n for n in self.remain_rooms if n!=id]
                new_state = self.create_room(id)
                child = Node(new_state, remain_rooms, self.ini_Q, ('R', None), ((None,None),(None,None)) )
                self.children.append(child)

        else:
            raise TypeError('unknown node type')

        return


    def choose_x(self): # checked ok
        xs = []
        states = []

        # create states
        for i in range(0, len(self.ori_x_intervals)-1):
            start = self.ori_x_intervals[i]
            end = self.ori_x_intervals[i+1]
            # all
            xs.append( end )
            states.append( deepcopy(self.state) )

            # half
            if start == end-1:
                new_state = np.insert( self.state, start+1 , self.state[:,start] , axis=1) # No problem, only changes interval of this state
            else:
                assert end-1 > start, 'end-1: {} should > start: {}'.format(end-1 , start)
                new_state = deepcopy(self.state)
            xs.append(start+1)
            states.append(new_state)

        return xs, states

    def choose_y(self):
        ys = []
        states = []

        # create states
        for i in range(0, len(self.ori_y_intervals) -1):
            start = self.ori_y_intervals[i]
            end = self.ori_y_intervals[i+1]
            # all
            if self.value==self.col and end==self.row:
                pass
            else:
                ys.append(end)
                states.append( deepcopy(self.state) )

            # half
            if start==end-1:
                new_state = np.insert(self.state, start+1, self.state[start,:], axis=0)
            else:
                assert end-1 > start, 'y end-1 should > y start'
                new_state=deepcopy(self.state)
            ys.append( start+1 )
            states.append(new_state)

        return ys,states

    def create_room(self,id):

        end_y, end_x = self.value
        new_state = deepcopy(self.state)
        for y in range(self.pos_y, end_y):
            for x in range(self.pos_x, end_x):
                assert new_state[y][x] == 0, 'create new room, but not empty'
                new_state[y][x]=id

        return new_state

    def complete(self):
        # complete horizontal
        for x in range(0, self.col-1):
            start = None
            for y in range(self.row-1, -1, -1):
                if self.state[y][x] != 0 and start==None:
                    start = y
                    value = self.state[y][x]

                if start != None:
                    if value != self.state[y][x]:
                        end = y
                        if (self.state[end+1:start+1,x+1]==0).all():
                            self.state[end + 1:start + 1,x+1] = value

                        break

        # complete vertically
        for x in range(0, self.col):
            for y in range(0, self.row):
                if self.state[y][x] == 0:
                    value = self.state[y-1][x]
                    assert value!=0, 'vertical fill value=0'
                    self.state[y:self.row, x] = value
                    break

        assert (self.state[:]!=0).all(), 'still have empty space'

        return











class MCTS(object):
    def __init__(self, Cons):
        """

        :param Cons: numpy constraints array where -1 means not to be neighbor, +1 means to be neighbor and 0 means no constraints.
        [[0, 1,-1],
         [0, 0, 1],
         [0, 0, 0]]

         remain_rooms = [1,2,3,...]

         state =
         [
         [4,4,3,3]
         [4,4,1,1]
         [2,2,2,0]
         [0,0,0,0]
         ]
        """
        assert Cons.shape[0]== Cons.shape[1], 'constraints matrix should have the same dimension on col and row'
        self.num_rooms = Cons.shape[0]
        self.Stat = np.zeros(Cons.shape,dtype=np.int32)

    def play(self):

