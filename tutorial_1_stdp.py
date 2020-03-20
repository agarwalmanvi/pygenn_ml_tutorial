import numpy as np
from os import path

from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class, create_custom_weight_update_class,
                               GeNNModel, init_var)
from pygenn.genn_wrapper import NO_DELAY
from mlxtend.data import loadlocal_mnist
import pickle
import csv

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
               "taupre": 2.0,
               "taupost": 2.0}
TIMESTEP = 1.0
PRESENT_TIMESTEPS = 100
INPUT_CURRENT_SCALE = 1.0 / 100.0

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
    param_names=["gmax", "taupre", "taupost"],
    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("apre", "scalar")],
    post_var_name_types=[("apost", "scalar")],
    sim_code=
        """
        $(addToInSyn, $(g));
        const scalar deltat = $(t) - $(sT_post);
        const scalar tracepost = $(apost) * exp( - $(taupost) * deltat);
        const scalar newg = $(g) + tracepost;
        $(g) = $(gmax) <= newg ? $(gmax) : newg;
        """,
    learn_post_code=
        """
        const scalar deltat = $(t) - $(sT_pre);
        const scalar tracepre = $(apre) * exp( - $(taupre) * deltat);
        const scalar newg = $(g) + tracepre;
        $(g) = $(gmax) <= newg ? $(gmax) : newg;
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

# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", "tutorial_1")
model.dT = TIMESTEP

# Initial values to initialise all neurons to
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

# Create synaptic connections between layers
for i, (pre, post) in enumerate(zip(neuron_layers[:-1], neuron_layers[1:])):
    model.add_synapse_population(
        "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
        pre, post,
        stdp_model, STDP_PARAMS, stdp_init, stdp_pre_init, stdp_post_init,
        "DeltaCurr", {}, {})

# Create current source to deliver input to first layers of neurons
current_input = model.add_current_source("current_input", cs_model,
                                         "neuron0" , {}, {"magnitude": 0.0})

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

print("Loading training images of size: " + str(X.shape))
print("Loading training labels of size: " + str(y.shape))

# ----------------------------------------------------------------------------
# Simulate
# ----------------------------------------------------------------------------
# Get views to efficiently access state variables
current_input_magnitude = current_input.vars["magnitude"].view
output_spike_count = neuron_layers[-1].vars["SpikeCount"].view
layer_voltages = [l.vars["V"].view for l in neuron_layers]

csv_filename = 'training.csv'

with open(csv_filename, 'w') as f:
    csv_write = csv.writer(f)
    csv_write.writerow(["example", "label", "reactive_neuron"])

# Simulate
while model.timestep < (PRESENT_TIMESTEPS * X.shape[0]):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    example = int(model.timestep // PRESENT_TIMESTEPS)

    # If this is the first timestep of presenting the example
    if timestep_in_example == 0:
        print("Example: " + str(example))
        current_input_magnitude[:] = X[example, :].flatten() * INPUT_CURRENT_SCALE
        model.push_var_to_device("current_input", "magnitude")

        # Loop through all layers and their corresponding voltage views
        for l, v in zip(neuron_layers, layer_voltages):
            # Manually 'reset' voltage
            v[:] = 0.0

            # Upload
            model.push_var_to_device(l.name, "V")

        # Zero spike count
        output_spike_count[:] = 0
        model.push_var_to_device(neuron_layers[-1].name, "SpikeCount")

    # Advance simulation
    model.step_time()

    # If this is the LAST timestep of presenting the example
    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        # Download spike count from last layer
        model.pull_var_from_device(neuron_layers[-1].name, "SpikeCount")

        true_label = y[example]
        most_reactive_neuron = np.argmax(output_spike_count)

        data_to_save = [example, true_label, most_reactive_neuron]

        with open(csv_filename, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data_to_save)

print("Completed training.")

# # ----------------------------------------------------------------------------
# # Plotting
# # ----------------------------------------------------------------------------
# import matplotlib.pyplot as plt
#
# # Create a plot with axes for each
# fig, axes = plt.subplots(len(neuron_layers), sharex=True)
#
#
# # Loop through axes and their corresponding neuron populations
# for a, s, l in zip(axes, layer_spikes, neuron_layers):
#     # Plot spikes
#     a.scatter(s[1], s[0], s=1)
#
#     # Set title, axis labels
#     a.set_title(l.name)
#     a.set_ylabel("Spike number")
#     a.set_xlim((0, PRESENT_TIMESTEPS * TIMESTEP))
#     a.set_ylim((0, l.size))
#
#
# # Add an x-axis label and translucent line showing the correct label
# axes[-1].set_xlabel("Time [ms]")
# axes[-1].hlines(testing_labels[0], xmin=0, xmax=PRESENT_TIMESTEPS,
#                 linestyle="--", color="gray", alpha=0.2)
#
# # Show plot
# plt.show()
