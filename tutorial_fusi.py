import numpy as np
from os import path

from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class, create_custom_weight_update_class,
                               GeNNModel, init_var)
from pygenn.genn_wrapper import NO_DELAY
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
IF_PARAMS = {"Vtheta": 1.0,
             "lambda": 0.1,
             "Vrest": 0.0,
             "Vreset": 0.0}
FUSI_PARAMS = {"tauC": 60.0, "a": 0.1, "b": 0.1, "thetaV": 0.8, "thetaLUp": 3.0,
               "thetaLDown": 3.0, "thetaHUp": 13.0, "thetaHDown": 4.0, "thetaX": 0.5,
               "alpha": 3.5, "beta": 3.5, "Xmax": 1.0, "Xmin": 0.0, "JC": 1.0}
TIMESTEP = 1.0
PRESENT_TIMESTEPS = 100
INPUT_CURRENT_SCALE = 1.0 / 100.0
OUTPUT_CURRENT_SCALE = 10.0
NUM_CLASSES = 10

# ----------------------------------------------------------------------------
# Custom GeNN models
# ----------------------------------------------------------------------------
# Very simple integrate-and-fire neuron model
if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vtheta", "lambda", "Vrest", "Vreset"],
    var_name_types=[("V", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code="""
    $(V) += (-$(lambda) + $(Isyn)) * DT;
    $(V) = fmin($(V), $(Vrest));
    """,
    reset_code="""
    $(V) = $(Vreset);
    $(SpikeCount)++;
    """,
    threshold_condition_code="$(V) >= $(Vtheta)"
)

fusi_model = create_custom_weight_update_class(
    "fusi_model",
    param_names=["tauC", "a", "b", "thetaV", "thetaLUp", "thetaLDown", "thetaHUp", "thetaHDown",
                 "thetaX", "alpha", "beta", "Xmax", "Xmin", "JC"],
    var_name_types=[("X", "scalar"), ("tpre", "scalar")],
    post_var_name_types=[("C", "scalar")],
    sim_code="""
    $(addToInSyn, $(X));
    if ($(V_post) > $(thetaV) and $(thetaLUp) < $(C) and $(C) < $(thetaHUp)) {
        $(X) += $(a);
    }
    else if ($(V_post) <= $(thetaV) and $(thetaLDown) < $(C) and $(C) < $(thetaHDown)) {
        $(X) -= $(b);
    }
    else {
        if ($(X) > $(thetaX)) {
            $(X) += $(alpha) * $(tpre);
        }
        else {
            $(X) -= $(beta) * $(tpre);
        }
    }
    $(X) = fmin($(Xmax), fmax($(Xmin), $(X)));
    $(tpre) = $(t);
    """,
    post_spike_code="""
    const scalar dt = $(t) - $(sT_post);
    $(C) = $(C) * exp(-dt / $(tauC)) + $(JC);
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
if_init = {"V": 0.0, "SpikeCount": 0}
# fusi_init = {"C": 0.0,
#              "X": 0.0}
fusi_init = {"X": 0.0, "tpre": 0.0}
fusi_post_init = {"C": 0.0}

neurons_count = [784, 128, NUM_CLASSES]
neuron_layers = []

# Create neuron layers
for i in range(len(neurons_count)):
    neuron_layers.append(model.add_neuron_population("neuron%u" % (i),
                                                     neurons_count[i], if_model,
                                                     IF_PARAMS, if_init))

weights_0_1 = np.load("weights_0_1.npy")

synapses = []

for i, (pre, post) in enumerate(zip(neuron_layers[:-1], neuron_layers[1:])):
    if i == 0:
        synapses.append(model.add_synapse_population(
            "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
            pre, post,
            "StaticPulse", {}, {"g": weights_0_1.flatten()}, {}, {},
            "DeltaCurr", {}, {}))
    else:
        synapses.append(model.add_synapse_population(
            "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
            pre, post,
            fusi_model, FUSI_PARAMS, fusi_init, {}, fusi_post_init,
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

print("Loading training images of size: " + str(X.shape))
print("Loading training labels of size: " + str(y.shape))

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
# Get views to efficiently access state variables
current_input_magnitude = current_input.vars["magnitude"].view
current_output_magnitude = current_output.vars["magnitude"].view
layer_voltages = [l.vars["V"].view for l in neuron_layers]

prefix = "fusi"
save_png_dir = "/home/manvi/Documents/pygenn_ml_tutorial/imgs/" + prefix

# Simulate
while model.timestep < (PRESENT_TIMESTEPS * X.shape[0]):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    example = int(model.timestep // PRESENT_TIMESTEPS)

    # If this is the first timestep of presenting the example
    if timestep_in_example == 0:

        # init a data structure for plotting the raster plots for this example
        layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]

        if example % 100 == 0:
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

    # Advance simulation
    model.step_time()

    # populate the raster plot data structure with the spikes of this example and this timestep
    for i, l in enumerate(neuron_layers):
        # Download spikes
        model.pull_current_spikes_from_device(l.name)

        # Add to data structure
        spike_times = np.ones_like(l.current_spikes) * model.t
        layer_spikes[i] = (np.hstack((layer_spikes[i][0], l.current_spikes)),
                           np.hstack((layer_spikes[i][1], spike_times)))

    # If this is the LAST timestep of presenting the example
    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        # Make a plot every 10000th example
        if example % 10000 == 0:

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

            # Show plot
            save_filename = path.join(save_png_dir, 'example' + str(example) + '.png')
            plt.savefig(save_filename)

print("Completed training.")

for i, l in enumerate(synapses):
    model.pull_var_from_device(l.name, "X")

    weight_values = l.get_var_values("X")
    print(type(weight_values))
    print(weight_values.shape)
    np.save(prefix + "w1_" + str(i) + "_" + str(i + 1) + ".npy", weight_values)

print("Dumped data.")
