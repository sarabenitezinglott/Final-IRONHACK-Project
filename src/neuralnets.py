import sys
import pycore

# DenseNet architecture definition
densenet_architecture = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # C1
    to_Conv("conv1", 32, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    
    # C2
    to_Conv("conv2", 64, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2),
    to_connection("pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    
    # C3
    to_Conv("conv3", 128, 64, offset="(1,0,0)", to="(pool2-east)", height=14, depth=14, width=2),
    to_connection("pool2", "conv3"),
    
    # Flatten
    to_FullyConnected("flatten", 512, offset="(1,0,0)", to="(conv3-east)", width=1),
    
    # Dense
    to_FullyConnected("dense1", 128, offset="(1,0,0)", to="(flatten-east)", width=1),
    
    # Output 
    to_Output("output", 1, offset="(1,0,0)", to="(dense1-east)", width=1, caption="sigmoid"),
    
    to_end()
]

def plot_neural_network(architecture):
    # Create a TikZ picture
    tikz = tikzpicture()
    for element in architecture:
        tikz.add(element)

    tikz_code = tikz.dumps()

    with open("neural_network.tex", "w") as tex_file:
        tex_file.write(tikz_code)
    print(tikz_code)

# Plot the DenseNet architecture
plot_neural_network(densenet_architecture)



