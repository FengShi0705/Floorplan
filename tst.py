import networkx as nx


def node_match(a,b):
    if a['label']==b['label']:
        return True
    else:
        return False

a=nx.Graph()
for n in range(1,4):
    a.add_node(n,label=n)


b=nx.Graph()
for n in range(4,7):
    b.add_node(n,label=n)

print('None specification')
print(nx.is_isomorphic(a,b))

print('with node match')
print(nx.is_isomorphic(a,b,node_match=node_match))

