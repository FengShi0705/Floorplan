import math

Q = [0.5,0.1]
N = [0,0]
T = [0,0]


for n in range(100):
    for i in [0, 1]:
        T[i] = Q[i] + ( math.sqrt(sum(N)) / (1 + N[i]))
    print(Q, N, T)

    ind = T.index(max(T))
    print('take', ind)
    N[ind] = N[ind] + 1



"""
assert T[0]>T[1]

while True:
    N[0] = N[0] + 1
    for i in [0, 1]:
        T[i] = Q[i] + math.sqrt(sum(N)) / (1 + N[i])
    if T[0]<T[1]:
        break

print(Q,N,T)
"""
