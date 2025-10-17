
import numpy as np
from random import seed

# ------------------------------------------------------------
# 1.  helpers (weighted sum, sigmoid, network init)
# ------------------------------------------------------------
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-weighted_sum))

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name, num_nodes = "output", num_nodes_output
        else:
            layer_name, num_nodes = f"layer_{layer+1}", num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = f"node_{node+1}"
            network[layer_name][node_name] = {
                "weights": np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                "bias":    np.around(np.random.uniform(size=1), decimals=2),
            }
        num_nodes_previous = num_nodes
    return network

# ------------------------------------------------------------
# 2.  forward propagation through arbitrary network
# ------------------------------------------------------------
def forward_propagate(network, inputs):
    layer_inputs = list(inputs)
    for layer in network:
        layer_data = network[layer]
        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            node_output = node_activation(
                compute_weighted_sum(layer_inputs, node_data["weights"], node_data["bias"])
            )
            layer_outputs.append(np.around(node_output[0], decimals=4))
        if layer != "output":
            print(f"Outputs of nodes in {layer}: {layer_outputs}")
        layer_inputs = layer_outputs
    return layer_outputs

# ------------------------------------------------------------
# 3.  tiny 2-input, 1-hidden-layer (2 nodes), 1-output demo
# ------------------------------------------------------------
if __name__ == "__main__":
    # ---- manual step-by-step (original notebook section) ----
    weights = np.around(np.random.uniform(size=6), decimals=2)
    biases  = np.around(np.random.uniform(size=3), decimals=2)
    print("Manual weights:", weights)
    print("Manual biases :", biases)

    x1, x2 = 0.5, 0.85
    print(f"Inputs x1={x1}, x2={x2}")

    # hidden node 1
    z11 = x1 * weights[0] + x2 * weights[1] + biases[0]
    a11 = node_activation(z11)
    print("z11 =", np.around(z11, 4), " -> a11 =", np.around(a11, 4))

    # hidden node 2
    z12 = x1 * weights[2] + x2 * weights[3] + biases[1]
    a12 = node_activation(z12)
    print("z12 =", np.around(z12, 4), " -> a12 =", np.around(a12, 4))

    # output node
    z2 = a11 * weights[4] + a12 * weights[5] + biases[2]
    a2 = node_activation(z2)
    print("Network output (a2) =", np.around(a2, 4))

    print("\n" + "-" * 50)

    # ---- generic network demo ----
    small_net = initialize_network(2, 1, [2], 1)
    inputs = np.around(np.random.uniform(size=2), decimals=2)
    print("Generic network inputs:", inputs)
    pred = forward_propagate(small_net, inputs)
    print("Generic network output:", pred)

    # ---- build any network on-the-fly ----
    my_net = initialize_network(5, 3, [2, 3, 2], 3)
    inputs = np.around(np.random.uniform(size=5), decimals=2)
    print("\nMy network inputs:", inputs)
    predictions = forward_propagate(my_net, inputs)
    print("My network predictions:", predictions)