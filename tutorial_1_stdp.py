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
# STDP_PARAMS = {"gmax": 1.0,
#                "taupre": 4.0,
#                "taupost": 4.0,
#                "gmin": -1.0,
#                "aplus": 0.1,
#                "aminus": 0.105}
STDP_PARAMS = {"gmax": 1.0,
               "tau": 20.0,
               "gmin": -1.0,
               "rho": 0.5,
               "eta": 0.01}
TIMESTEP = 1.0
PRESENT_TIMESTEPS = 100
# INPUT_CURRENT_SCALE = 1.0 / 200.0
# OUTPUT_CURRENT_SCALE = 10000.0
INPUT_CURRENT_SCALE = 1.0 / 100.0
OUTPUT_CURRENT_SCALE = 10.0
NUM_CLASSES = 10

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
#     param_names=["gmax", "taupre", "taupost", "gmin", "aplus", "aminus"],
#     var_name_types=[("g", "scalar")],
#     sim_code=
#         """
#         $(addToInSyn, $(g));
#         scalar deltat = $(t) - $(sT_post);
#         if (deltat > 0) {
#             scalar newg = $(g) - ($(aminus) * exp( - deltat / $(taupost)));
#             $(g) = fmin($(gmax), fmax($(gmin), newg));
#         }
#         """,
#     learn_post_code=
#         """
#         const scalar deltat = $(t) - $(sT_pre);
#         if (deltat > 0) {
#             scalar newg = $(g) + ($(aplus) * exp( - deltat / $(taupre)));
#             $(g) = fmin($(gmax), fmax($(gmin), newg));
#         }
#         """,
#     is_pre_spike_time_required=True,
#     is_post_spike_time_required=True
# )

stdp_model = create_custom_weight_update_class(
    "stdp_model",
    param_names=["tau", "rho", "eta", "gmin", "gmax"],
    var_name_types=[("g", "scalar")],
    sim_code=
        """
        $(addToInSyn, $(g));
        scalar deltat = $(t) - $(sT_post);
        if (deltat > 0) {
            scalar timing = exp(-deltat / $(tau)) - $(rho);
            scalar newg = $(g) + ($(eta) * timing);
            $(g) = fmin($(gmax), fmax($(gmin), newg));
        }
        """,
    learn_post_code=
        """
        const scalar deltat = $(t) - $(sT_pre);
        if (deltat > 0) {
            scalar timing = exp(-deltat / $(tau));
            scalar newg = $(g) + ($(eta) * timing);
            $(g) = fmin($(gmax), fmax($(gmin), newg));
        }
        """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True
)

# Current source model which injects current with a magnitude specified by a state variable
cs_model = create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", "tutorial_1")
model.dT = TIMESTEP

# Initial values for initialisation
if_init = {"V": 0.0, "SpikeCount":0}
stdp_init = {"g": init_var("Uniform", {"min": STDP_PARAMS["gmin"], "max": STDP_PARAMS["gmax"]})}

neurons_count = [784, 128, NUM_CLASSES]
neuron_layers = []

# Create neuron layers
for i in range(len(neurons_count)):
    neuron_layers.append(model.add_neuron_population("neuron%u" % (i),
                                                     neurons_count[i], if_model,
                                                     IF_PARAMS, if_init))

weights_0_1 = np.load("weights_0_1.npy")

synapses = []
# Create synaptic connections between layers
# for i, (pre, post) in enumerate(zip(neuron_layers[:-1], neuron_layers[1:])):
#     if i == 0:
#         synapses.append(model.add_synapse_population(
#             "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
#             pre, post,
#             "StaticPulse", {}, {"g": weights_0_1.flatten()}, {}, {},
#             "DeltaCurr", {}, {}))
#     else:
#         synapses.append(model.add_synapse_population(
#             "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
#             pre, post,
#             stdp_model, STDP_PARAMS, stdp_init, {}, {},
#             "DeltaCurr", {}, {}))

for i, (pre, post) in enumerate(zip(neuron_layers[:-1], neuron_layers[1:])):
    if i == 0:
        synapses.append(model.add_synapse_population(
            "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
            pre, post,
            "StaticPulse", {}, stdp_init, {}, {},
            "DeltaCurr", {}, {}))
    else:
        synapses.append(model.add_synapse_population(
            "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
            pre, post,
            stdp_model, STDP_PARAMS, stdp_init, {}, {},
            "DeltaCurr", {}, {}))



# Create current source to deliver input to first layers of neurons
current_input = model.add_current_source("current_input", cs_model,
                                         "neuron0", {}, {"magnitude": 0.0})

# Create current source to deliver target output to last layer of neurons
current_output = model.add_current_source("current_output", cs_model,
                                          "neuron2", {}, {"magnitude": 0.0})

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

# binary classification
# idx = [i for i in range(len(y)) if y[i] == 0 or y[i] == 1]
# X = X[idx]
# y = y[idx]

# one vs all type classification
# idx = [i for i in range(len(y)) if y[i] != 0]
# y[idx] = 1

print("Loading training images of size: " + str(X.shape))
print("Loading training labels of size: " + str(y.shape))

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
# Get views to efficiently access state variables
current_input_magnitude = current_input.vars["magnitude"].view
current_output_magnitude = current_output.vars["magnitude"].view
layer_voltages = [l.vars["V"].view for l in neuron_layers]

exp = "vogels"
prefix = "vogels"
save_png_dir = "/home/manvi/Documents/pygenn_ml_tutorial/imgs/" + exp

print("Experiment: " + prefix)

model.pull_var_from_device(synapses[1].name, "g")
weight_values = synapses[1].get_var_values("g")
print("max: ")
print(np.amax(weight_values))
print("min: ")
print(np.amin(weight_values))

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

        if example % 1000 == 0:
            print("Example: " + str(example))

        current_input_magnitude[:] = X[example, :].flatten() * INPUT_CURRENT_SCALE
        one_hot = np.zeros((NUM_CLASSES))
        one_hot[y[example]] = 1
        current_output_magnitude[:] = one_hot.flatten() * OUTPUT_CURRENT_SCALE
        model.push_var_to_device("current_input", "magnitude")
        model.push_var_to_device("current_output", "magnitude")

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

            model.pull_var_from_device(synapses[1].name, "g")
            weight_values = synapses[1].get_var_values("g")
            print("max: ")
            print(np.amax(weight_values))
            print("min: ")
            print(np.amin(weight_values))

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
            save_filename = path.join(save_png_dir, prefix + '_example' + str(example) + '.png')
            plt.savefig(save_filename)

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

model.pull_var_from_device(synapses[1].name, "g")
weight_values = synapses[1].get_var_values("g")
print("max: ")
print(np.amax(weight_values))
print("min: ")
print(np.amin(weight_values))

for i, l in enumerate(synapses):

    model.pull_var_from_device(l.name, "g")

    weight_values = l.get_var_values("g")
    print(type(weight_values))
    print(weight_values.shape)
    np.save(prefix + "w1_"+str(i)+"_"+str(i+1)+".npy", weight_values)

print("Dumped data.")
