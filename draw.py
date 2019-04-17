import matplotlib.pyplot as plt
import json

fig, ax = plt.subplots()

"""
with open('Case1-final/250search_Falsesimrand_Trueplayrand__Falseavg_1caseid.json','r') as f:
    records=json.load(f)

x = [record[0] for record in records]
y = [record[2] for record in records]

ax.plot(x,y,'.-', label='our algorithm without nonadj', color='r')

with open('Case1-final/250search_Falsesimrand_Trueplayrand__Trueavg_1caseid.json','r') as f:
    records=json.load(f)

x = [record[0] for record in records]
y = [record[2] for record in records]

ax.plot(x,y,'<:', label='on-policy without nonadj', color='r')

with open('Case1-final/1000search_Falsesimrand_Trueplayrand__Falseavg_1noncaseid.json','r') as f:
    records=json.load(f)

x = [record[0] for record in records]
y = [record[2] for record in records]

ax.plot(x,y,'.-', label='our algorithm with nonadj', color='b')

with open('Case1-final/1000search_Falsesimrand_Trueplayrand__Trueavg_1noncaseid.json','r') as f:
    records=json.load(f)

x = [record[0] for record in records]
y = [record[2] for record in records]

ax.plot(x,y,'<:', label='on-policy with nonadj', color='b')


ax.legend()
ax.set_xlabel('time (s)')
ax.set_ylabel('Reward at each replay')
"""

with open('Case2-final/3000search_Falsesimrand_Falseplayrand_Falseavg_2caseid_2.0exp.json','r') as f:
    records=json.load(f)

x = [record[0] for record in records]
y = [record[2] for record in records]

ax.plot(x,y,'.-', label='our algorithm without nonadj', color='r')

with open('Case2-final/3000search_Falsesimrand_Falseplayrand_Trueavg_2caseid_2.0exp.json','r') as f:
    records=json.load(f)

x = [record[0] for record in records]
y = [record[2] for record in records]

ax.plot(x,y,'<:', label='on-policy without nonadj', color='r')

with open('Case2-final/3000search_Falsesimrand_Falseplayrand_Falseavg_2noncaseid_2.0exp.json','r') as f:
    records=json.load(f)

x = [record[0] for record in records]
y = [record[2] for record in records]

ax.plot(x,y,'.-', label='our algorithm with nonadj', color='b')

with open('Case2-final/3000search_Falsesimrand_Falseplayrand_Trueavg_2noncaseid_2.0exp.json','r') as f:
    records=json.load(f)

x = [record[0] for record in records]
y = [record[2] for record in records]

ax.plot(x,y,'<:', label='on-policy with nonadj', color='b')


ax.legend()
ax.set_xlabel('time (s)')
ax.set_ylabel('Reward at each replay')


plt.show()
