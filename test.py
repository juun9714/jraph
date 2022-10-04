import jraph
import jax

import jax.numpy as jnp

# Define a three node graph, each node has an integer as its feature.
# 3 nodes -> each node's features as number
node_features = jnp.array([[0.], [1.], [2.]])

# We will construct a graph for which there is a directed edge between each node
# and its successor. We define this with `senders` (source nodes) and `receivers`
# (destination nodes).

senders = jnp.array([0, 1, 2]) # source nodes
receivers = jnp.array([1, 2, 0]) # destination nodes

# 0 -> 1 -> 2 -> 0

# You can optionally add edge attributes.
edges = jnp.array([[5.], [6.], [7.]])

# We then save the number of nodes and the number of edges.
# This information is used to make running GNNs over multiple graphs
# in a GraphsTuple possible.
n_node = jnp.array([3]) # number of nodes
n_edge = jnp.array([3]) # number of edges

# Optionally you can add `global` information, such as a graph label.

global_context = jnp.array([[1]]) # global information
graph = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)
two_graph_graphstuple = jraph.batch([graph, graph])

print(two_graph_graphstuple)
