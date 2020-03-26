import numpy as np
from os import path

from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class, create_custom_weight_update_class,
                               GeNNModel, init_var)
from pygenn.genn_wrapper import NO_DELAY
from mlxtend.data import loadlocal_mnist
import csv
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
IF_PARAMS = {"Vthr": 5.0}
# STDP_PARAMS = {"gmax": 1.0,
#                "taupre": 2.0,
#                "taupost": 2.0,
#                "incpre": 0.1,
#                "incpost": -0.105}
STDP_PARAMS = {"gmax": 1.0,
               "taupre": 4.0,
               "taupost": 4.0,
               "gmin": 0.0}
TIMESTEP = 1.0
PRESENT_TIMESTEPS = 100
# INPUT_CURRENT_SCALE = 1.0 / 200.0
# OUTPUT_CURRENT_SCALE = 10000.0
INPUT_CURRENT_SCALE = 1.0 / 100.0
OUTPUT_CURRENT_SCALE = 10.0

# ----------------------------------------------------------------------------
# Custom GeNN models
# ----------------------------------------------------------------------------
# Very simple integrate-and-fire neuron model
if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vthr"],
    var_name_types=[("V", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code="$(V) += $(Isyn) * DT;",
    reset_code="""
    $(V) = 0.0;
    $(SpikeCount)++;
    """,
    threshold_condition_code="$(V) >= $(Vthr)")

# stdp_model = create_custom_weight_update_class(
#     "stdp_model",
#     param_names=["gmax", "taupre", "taupost", "incpre", "incpost"],
#     var_name_types=[("g", "scalar"), ("apre", "scalar"), ("apost", "scalar")],
#     sim_code=
#         """
#         $(addToInSyn, $(g));
#         $(apre) += -$(apre) / $(taupre) * DT;
#         $(apost) += -$(apost) / $(taupost) * DT
#         """,
#     learn_post_code=
#         """$(apost) += $(incpost);
#         const scalar newg = $(g) + $(apre);
#         $(g) = $(gmax) <= newg ? $(gmax) : newg;
#         """,
#     pre_spike_code=
#         """$(apre) += $(incpre);
#         const scalar newg = g + $(apost);
#         $(g) = $(gmax) <= newg ? $(gmax) : newg;
#         """,
#     is_pre_spike_time_required=True,
#     is_post_spike_time_required=True
# )

stdp_model = create_custom_weight_update_class(
    "stdp_model",
    param_names=["gmax", "taupre", "taupost", "gmin"],
    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("apre", "scalar")],
    post_var_name_types=[("apost", "scalar")],
    sim_code=
        """
        const scalar deltat = $(t) - $(sT_post);
        const scalar tracepost = $(apost) * exp( - $(taupost) * deltat);
        const scalar newg = $(g) + tracepost;
        $(g) = $(gmax) <= newg ? $(gmax) : newg;
        $(g) = $(gmin) >= newg ? $(gmin) : newg;
        $(addToInSyn, $(g));
        """,
    learn_post_code=
        """
        const scalar deltat = $(t) - $(sT_pre);
        const scalar tracepre = $(apre) * exp( - $(taupre) * deltat);
        const scalar newg = $(g) + tracepre;
        $(g) = $(gmax) <= newg ? $(gmax) : newg;
        $(g) = $(gmin) >= newg ? $(gmin) : newg;
        """,
    pre_spike_code=
        """ $(apre) += 0.1;""",
    post_spike_code=
        """ $(apost) -= 0.105;""",
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True
)

# Current source model which injects current with a magnitude specified by a state variable
cs_model = create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

output_cs_model = create_custom_current_source_class(
    "output_cs_model",
    var_name_types=[("co_magnitude", "scalar")],
    injection_code="$(injectCurrent, $(co_magnitude));"
)

# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", "tutorial_1")
model.dT = TIMESTEP

# Initial values for initialisation
if_init = {"V": 0.0, "SpikeCount":0}
stdp_init = {"g": init_var("Uniform", {"min": 0.0, "max": STDP_PARAMS["gmax"]})}
stdp_pre_init = {"apre": 0.0}
stdp_post_init = {"apost": 0.0}

neurons_count = [784, 128, 10]

# Create first neuron layer
neuron_layers = [model.add_neuron_population("neuron0", neurons_count[0],
                                             if_model, IF_PARAMS, if_init)]

# Create subsequent neuron layer
for i in range(1, len(neurons_count)):
    neuron_layers.append(model.add_neuron_population("neuron%u" % (i),
                                                     neurons_count[i], if_model,
                                                     IF_PARAMS, if_init))

synapses = []
# Create synaptic connections between layers
for i, (pre, post) in enumerate(zip(neuron_layers[:-1], neuron_layers[1:])):
    synapses.append(model.add_synapse_population(
        "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
        pre, post,
        stdp_model, STDP_PARAMS, stdp_init, stdp_pre_init, stdp_post_init,
        "DeltaCurr", {}, {}))

# Create current source to deliver input to first layers of neurons
current_input = model.add_current_source("current_input", cs_model,
                                         "neuron0" , {}, {"magnitude": 0.0})

# Create current source to deliver target output to last layer of neurons
current_output = model.add_current_source("current_output", output_cs_model,
                                         "neuron2" , {}, {"co_magnitude": 0.0})

# Build and load our model
model.build()
model.load()

# ----------------------------------------------------------------------------
# Import training data
# ----------------------------------------------------------------------------
data_dir = "/home/manvi/Documents/pygenn_ml_tutorial/mnist"
X, y = loadlocal_mnist(
        images_path=path.join(data_dir, 'train-images-idx3-ubyte'),
        labels_path=path.join(data_dir, 'train-labels-idx1-ubyte'))

# # reduce the dataset so that we can plot it periodically and see what's happening in the network
# X = X[:10, :]
# y = y[:10]

print("Loading training images of size: " + str(X.shape))
print("Loading training labels of size: " + str(y.shape))

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
# Get views to efficiently access state variables
current_input_magnitude = current_input.vars["magnitude"].view
current_output_co_magnitude = current_output.vars["co_magnitude"].view
layer_voltages = [l.vars["V"].view for l in neuron_layers]

# Simulate
while model.timestep < (PRESENT_TIMESTEPS * X.shape[0]):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    example = int(model.timestep // PRESENT_TIMESTEPS)

    # If this is the first timestep of presenting the example
    if timestep_in_example == 0:

        # init a data structure for plotting the raster plots for this example
        layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]
        # synapse_weights = [np.array([]) for _ in enumerate(synapses)]
        # layer_currents = [np.array([]) for _ in enumerate(neuron_layers)]

        if example % 100 == 0:
            print("Example: " + str(example))

        current_input_magnitude[:] = X[example, :].flatten() * INPUT_CURRENT_SCALE
        one_hot = np.zeros((10))
        one_hot[y[example]] = 1
        current_output_co_magnitude[:] = one_hot.flatten() * OUTPUT_CURRENT_SCALE
        model.push_var_to_device("current_input", "magnitude")
        model.push_var_to_device("current_output", "co_magnitude")

        # Loop through all layers and their corresponding voltage views
        for l, v in zip(neuron_layers, layer_voltages):
            # Manually 'reset' voltage
            v[:] = 0.0

            # Upload
            model.push_var_to_device(l.name, "V")

        # Zero spike count
        # output_spike_count[:] = 0
        # model.push_var_to_device(neuron_layers[-1].name, "SpikeCount")

    # Advance simulation
    model.step_time()

    # if timestep_in_example % 1 == 0:
    #     # Record the synapse weights
    #     for i, l in enumerate(synapses):
    #
    #         model.pull_var_from_device(l.name, "g")
    #
    #         # Add to data structure
    #         weight_values = l.get_var_values("g")
    #         weight_values = weight_values.reshape((1, len(weight_values)))
    #         if synapse_weights[i].size == 0:
    #             synapse_weights[i] = weight_values
    #         else:
    #             synapse_weights[i] = np.concatenate((synapse_weights[i], weight_values), axis=0)
    #
    #     for i, l in enumerate(neuron_layers):
    #
    #         model.pull_var_from_device(l.name, "V")
    #
    #         # Add to data structure
    #         current_values = l.vars["V"].view
    #         current_values = current_values.reshape((1, len(current_values)))
    #         if layer_currents[i].size == 0:
    #             layer_currents[i] = current_values
    #         else:
    #             layer_currents[i] = np.concatenate((layer_currents[i], current_values), axis=0)

    # populate the raster plot data structure with the spikes of this example and this timestep
    for i, l in enumerate(neuron_layers):
        # print("Neuron layer: " + str(i))
        # Download spikes
        model.pull_current_spikes_from_device(l.name)

        # Add to data structure
        spike_times = np.ones_like(l.current_spikes) * model.t
        # print(spike_times)
        # print(len(spike_times))
        layer_spikes[i] = (np.hstack((layer_spikes[i][0], l.current_spikes)),
                           np.hstack((layer_spikes[i][1], spike_times)))
        # print(layer_spikes[i])

    # If this is the LAST timestep of presenting the example
    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        # Download spike count from last layer
        # model.pull_var_from_device(neuron_layers[-1].name, "SpikeCount")

        # Make a plot every 10000th example
        if example % 10000 == 0:

            # for s, w in zip(synapses, synapse_weights):
            #     print("\n")
            #     print("Synapse population: " + str(s.name))
            #     print("\n")
            #     print("Minimum synaptic weight: " + str(np.amin(w)))
            #     print("Maximum synaptic weight: " + str(np.amax(w)))
            #
            # for s, w in zip(neuron_layers, layer_currents):
            #     print("\n")
            #     print("Synapse population: " + str(s.name))
            #     print("5 was hit? " + str(5 in w))
            #     print("\n")
            #     print("Minimum input current: " + str(np.amin(w)))
            #     print("Maximum input current: " + str(np.amax(w)))


            print("Creating raster plot")

            # Create a plot with axes for each
            fig, axes = plt.subplots(len(neuron_layers), sharex=True)


            # Loop through axes and their corresponding neuron populations
            for a, s, l in zip(axes, layer_spikes, neuron_layers):
                # Plot spikes
                a.scatter(s[1], s[0], s=1)

                # Set title, axis labels
                a.set_title(l.name)
                a.set_ylabel("Spike number")
                a.set_xlim((example * PRESENT_TIMESTEPS, (example + 1) * PRESENT_TIMESTEPS))
                a.set_ylim((-1, l.size + 1))


            # Add an x-axis label
            axes[-1].set_xlabel("Time [ms]")
            # axes[-1].hlines(testing_labels[0], xmin=0, xmax=PRESENT_TIMESTEPS,
            #                 linestyle="--", color="gray", alpha=0.2)

            # Show plot
            plt.savefig('example' + str(example) + '.png')

            # print("Creating V figure")
            #
            # fig, axes = plt.subplots(len(neuron_layers), sharex=True)
            #
            # # Loop through axes and their corresponding neuron populations
            # for a, s, l in zip(axes, layer_currents, neuron_layers):
            #
            #     plot_x = list(range(s.shape[0]))
            #     # Plot evolution of weights over time as a line
            #     for w in range(s.shape[1]):
            #         if w%100 ==0:
            #             print("At neuron " + str(w))
            #         plot_y = s[:, w]
            #         a.plot(plot_x, plot_y)
            #
            #     # Set title, axis labels
            #     a.set_title(l.name)
            #     a.set_ylabel("V")
            #     # a.set_xlim(s.shape[0])
            #     a.set_ylim((-1, 6))
            #     a.axhline(y = 5, xmin=0, xmax=s.shape[0])
            #
            #     # Add an x-axis label
            #     axes[-1].set_xlabel("Timesteps")
            #
            #     # Show plot
            #     plt.savefig('example' + str(example) + 'V.png')


            # print("Creating plot for weights")
            # # Create a plot with axes for each
            # fig, axes = plt.subplots(len(synapses), sharex=True)
            #
            # # Loop through axes and their corresponding neuron populations
            # for a, s, l in zip(axes, synapse_weights, synapses):
            #
            #     plot_x = list(range(s.shape[0]))
            #     # Plot evolution of weights over time as a line
            #     for w in range(s.shape[1]):
            #         if w%1000 ==0:
            #             print("At weight " + str(w))
            #         plot_y = s[:, w]
            #         a.plot(plot_x, plot_y)
            #
            #     # Set title, axis labels
            #     a.set_title(l.name)
            #     a.set_ylabel("Synaptic weight")
            #     a.set_xlim(s.shape[0])
            #     a.set_ylim((np.amin(s), np.amax(s)))
            #
            #     # Add an x-axis label
            #     axes[-1].set_xlabel("Timesteps")
            #
            #     # Show plot
            #     plt.savefig('example' + str(example) + 'syn.png')


print("Completed training.")

for i, l in enumerate(synapses):

    model.pull_var_from_device(l.name, "g")

    weight_values = l.get_var_values("g")
    print(type(weight_values))
    print(weight_values.shape)
    np.save("w_"+str(i)+"_"+str(i+1)+".npy", weight_values)

print("Dumped data.")