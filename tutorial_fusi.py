import numpy as np
import csv
import os
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class, create_custom_weight_update_class,
                               GeNNModel, init_var, init_connectivity)
from pygenn.genn_wrapper import NO_DELAY
from mlxtend.data import loadlocal_mnist
import math
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
IF_PARAMS = {"Vtheta": 1.0,
             "lambda": 0.01,
             "Vrest": 0.0,
             "Vreset": 0.0}
FUSI_PARAMS = {"tauC": 60.0, "a": 0.1, "b": 0.1, "thetaV": 0.8, "thetaLUp": 3.0,
               "thetaLDown": 3.0, "thetaHUp": 13.0, "thetaHDown": 4.0, "thetaX": 0.5,
               "alpha": 0.0035, "beta": 0.0035, "Xmax": 1.0, "Xmin": 0.0, "JC": 1.0,
               "Jplus": 1.0, "Jminus": 0.0}
TIMESTEP = 1.0
PRESENT_TIMESTEPS = 100
INPUT_CURRENT_SCALE = 1.0 / 100.0
OUTPUT_CURRENT_SCALE = 10.0
NUM_CLASSES = 10

# ----------------------------------------------------------------------------
# Custom GeNN models
# ----------------------------------------------------------------------------
if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vtheta", "lambda", "Vrest", "Vreset"],
    var_name_types=[("V", "scalar")],
    sim_code="""
    if ($(V) >= $(Vtheta)) {
        $(V) = $(Vreset);
    }
    $(V) += (-$(lambda) + $(Isyn)) * DT;
    $(V) = fmax($(V), $(Vrest));
    """,
    reset_code="""
    """,
    threshold_condition_code="$(V) >= $(Vtheta)"
)

fusi_model = create_custom_weight_update_class(
    "fusi_model",
    param_names=["tauC", "a", "b", "thetaV", "thetaLUp", "thetaLDown", "thetaHUp", "thetaHDown",
                 "thetaX", "alpha", "beta", "Xmax", "Xmin", "JC", "Jplus", "Jminus"],
    var_name_types=[("X", "scalar"), ("last_tpre", "scalar")],
    post_var_name_types=[("C", "scalar")],
    sim_code="""
    $(addToInSyn, ($(X) > $(thetaX)) ? $(Jplus) : $(Jminus));
    const scalar dt = $(t) - $(sT_post);
    const scalar decayC = $(C) * exp(-dt / $(tauC));
    if ($(V_post) > $(thetaV) && $(thetaLUp) < decayC && decayC < $(thetaHUp)) {
        $(X) += $(a);
    }
    else if ($(V_post) <= $(thetaV) && $(thetaLDown) < decayC && decayC < $(thetaHDown)) {
        $(X) -= $(b);
    }
    else {
        const scalar X_dt = $(t) - $(last_tpre);
        if ($(X) > $(thetaX)) {
            $(X) += $(alpha) * X_dt;
        }
        else {
            $(X) -= $(beta) * X_dt;
        }
    }
    $(X) = fmin($(Xmax), fmax($(Xmin), $(X)));
    $(last_tpre) = $(t);
    """,
    post_spike_code="""
    const scalar dt = $(t) - $(sT_post);
    $(C) = ($(C) * exp(-dt / $(tauC))) + $(JC);
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
# Import training data
# ----------------------------------------------------------------------------
data_dir = "/home/manvi/Documents/pygenn_ml_tutorial/mnist"
X, y = loadlocal_mnist(
    images_path=os.path.join(data_dir, 'train-images-idx3-ubyte'),
    labels_path=os.path.join(data_dir, 'train-labels-idx1-ubyte'))

X = np.divide(X, 255)

X = X[:100, :]
y = y[:100]

print("Loading training images of size: " + str(X.shape))
print("Loading training labels of size: " + str(y.shape))

# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", "tutorial_1")
model.dT = TIMESTEP

# Initial values for initialisation
if_init = {"V": 0.0}
fusi_init = {"X": 0.0,
             "last_tpre": 0.0}
fusi_post_init = {"C": 2.0}

NUM_INPUT = X.shape[1]

neurons_count = {"inp": NUM_INPUT,
                 "inh": 2000,
                 "out": NUM_CLASSES,
                 "teacher": NUM_CLASSES}

neuron_layers = {}

poisson_params = {"rate": 1.0}
poisson_init = {"timeStepToSpike": 0.0}

for k in neurons_count.keys():
    if k == "out":
        neuron_layers[k] = model.add_neuron_population(k, neurons_count[k],
                                                       if_model, IF_PARAMS, if_init)
    else:
        neuron_layers[k] = model.add_neuron_population(k, neurons_count[k],
                                                       "PoissonNew",
                                                       poisson_params, poisson_init)

inp2out = model.add_synapse_population(
    "inp2out", "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers['inp'], neuron_layers['out'],
    fusi_model, FUSI_PARAMS, fusi_init, {}, fusi_post_init,
    "DeltaCurr", {}, {})

inh2out = model.add_synapse_population(
    "inh2out", "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers['inh'], neuron_layers['out'],
    "StaticPulse", {}, {"g": -0.035}, {}, {},
    "DeltaCurr", {}, {})

teacher2out = model.add_synapse_population(
    "teacher2out", "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers['teacher'], neuron_layers['out'],
    "StaticPulse", {}, {"g": 0.5}, {}, {},
    "DeltaCurr", {}, {})

# Build and load our model
model.build()
model.load()

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
# Get views to efficiently access state variables
out_voltage = neuron_layers['out'].vars['V'].view
input_rate = neuron_layers['inp'].params['rate'].view
# layer_voltages = [l.vars["V"].view for l in list(neuron_layers.values())]

while model.timestep < (PRESENT_TIMESTEPS * X.shape[0]):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    example = int(model.timestep // PRESENT_TIMESTEPS)

    # If this is the first timestep of presenting the example
    if timestep_in_example == 0:

        # init a data structure for plotting the raster plots for this example
        layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]

        if example % 10 == 0:
            print("Example: " + str(example))

        digit = X[example, :].flatten()
        active_pixels = np.count_nonzero(digit)
        poisson_rates = 50 * active_pixels / NUM_INPUT
        input_rate = (digit * 48) + 2
        model.push_param_to_device("inp", "rate")
#
#
#
#         # Loop through all layers and their corresponding voltage views
#         for l, v in zip(list(neuron_layers.values()), layer_voltages):
#             # Manually 'reset' voltage
#             v[:] = 0.0
#
#             # Upload
#             model.push_var_to_device(l.name, "V")
#
#     # Advance simulation
#     model.step_time()
#
#     for i, l in enumerate(list(neuron_layers.values())):
#         # Download spikes
#         model.pull_current_spikes_from_device(l.name)
#
#         # Add to data structure
#         spike_times = np.ones_like(l.current_spikes) * model.t
#         layer_spikes[i] = (np.hstack((layer_spikes[i][0], l.current_spikes)),
#                            np.hstack((layer_spikes[i][1], spike_times)))
#
#     # If this is the LAST timestep of presenting the example
#     if timestep_in_example == (PRESENT_TIMESTEPS - 1):
#
#         # Make a plot every 10th example
#         if example % 10 == 0:
#
#             print("Creating raster plot")
#
#             # Create a plot with axes for each
#             fig, axes = plt.subplots(len(neuron_layers), sharex=True)
#
#             # Loop through axes and their corresponding neuron populations
#             for a, s, l in zip(axes, layer_spikes, list(neuron_layers.values())):
#                 # Plot spikes
#                 a.scatter(s[1], s[0], s=1)
#
#                 # Set title, axis labels
#                 a.set_title(l.name)
#                 a.set_ylabel("Spike number")
#                 a.set_xlim((example * PRESENT_TIMESTEPS, (example + 1) * PRESENT_TIMESTEPS))
#                 a.set_ylim((-1, l.size + 1))
#
#             # Add an x-axis label
#             axes[-1].set_xlabel("Time [ms]")
#             # axes[-1].hlines(testing_labels[0], xmin=0, xmax=PRESENT_TIMESTEPS,
#             #                 linestyle="--", color="gray", alpha=0.2)
#
#             # Show plot
#             save_filename = os.path.join('example' + str(example) + '.png')
#             plt.savefig(save_filename)
#
# print("Completed training.")
