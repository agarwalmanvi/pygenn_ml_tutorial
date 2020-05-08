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
TIMESTEP = 1.0
PRESENT_TIMESTEPS = 300
NUM_CLASSES = 10

# ----------------------------------------------------------------------------
# Custom GeNN models
# ----------------------------------------------------------------------------
if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vtheta", "lambda", "Vrest", "Vreset"],
    var_name_types=[("V", "scalar"), ("SpikeCount", "scalar")],
    sim_code="""
    if ($(V) >= $(Vtheta)) {
        $(V) = $(Vreset);
        $(SpikeCount) += 1;
    }
    $(V) += (-$(lambda) + $(Isyn)) * DT;
    $(V) = fmax($(V), $(Vrest));
    """,
    reset_code="""
    """,
    threshold_condition_code="$(V) >= $(Vtheta)"
)

poisson_model = create_custom_neuron_class(
    "poisson_model",
    var_name_types=[("rate", "scalar")],
    sim_code="""
    """,
    reset_code="""
    """,
    threshold_condition_code="$(gennrand_uniform) >= exp(-$(rate) * 0.001 * DT)"
)

# ----------------------------------------------------------------------------
# Import training data
# ----------------------------------------------------------------------------
data_dir = "/home/manvi/Documents/pygenn_ml_tutorial/mnist"
X, y = loadlocal_mnist(
    images_path=os.path.join(data_dir, 't10k-images-idx3-ubyte'),
    labels_path=os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

print("Loading testing images of size: " + str(X.shape))
print("Loading testing labels of size: " + str(y.shape))

# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", "tutorial_1")
model.dT = TIMESTEP

# Initial values for initialisation
if_init = {"V": 0.0,
           "SpikeCount": 0.0}
poisson_init = {"rate": 1.0}

NUM_INPUT = X.shape[1]

neurons_count = {"inp": NUM_INPUT,
                 "inh": 2000,
                 "out": NUM_CLASSES}

neuron_layers = {}

for k in neurons_count.keys():
    if k == "out":
        neuron_layers[k] = model.add_neuron_population(k, neurons_count[k],
                                                       if_model, IF_PARAMS, if_init)
    else:
        neuron_layers[k] = model.add_neuron_population(k, neurons_count[k],
                                                       poisson_model, {}, poisson_init)

inp2out_w = np.load("fusi.npy")

inp2out = model.add_synapse_population(
    "inp2out", "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers['inp'], neuron_layers['out'],
    "StaticPulse", {}, {"g": inp2out_w.flatten()}, {}, {},
    "DeltaCurr", {}, {})

inh2out = model.add_synapse_population(
    "inh2out", "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers['inh'], neuron_layers['out'],
    "StaticPulse", {}, {"g": -0.035}, {}, {},
    "DeltaCurr", {}, {})

# Build and load our model
model.build()
model.load()

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
# Get views to efficiently access state variables
out_voltage = neuron_layers['out'].vars['V'].view
input_rate = neuron_layers['inp'].vars['rate'].view
inh_rate = neuron_layers['inh'].vars['rate'].view
output_spike_count = neuron_layers["out"].vars["SpikeCount"].view

all_spike_rates = []

INH_V = 50
INPUT_UNSTIM = 2
INPUT_STIM = 50
TEACHER_V = 150

num_correct = 0

while model.timestep < (PRESENT_TIMESTEPS * X.shape[0]):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    example = int(model.timestep // PRESENT_TIMESTEPS)

    # If this is the first timestep of presenting the example
    if timestep_in_example == 0:

        if example % 100 == 0:
            print("Example: " + str(example))

        # calculate the correct spiking rates for all populations
        digit = X[example, :].flatten()
        digit = np.divide(digit, np.amax(digit))
        active_pixels = np.count_nonzero(digit)

        inh_rate[:] = INH_V * active_pixels / NUM_INPUT
        model.push_var_to_device("inh", "rate")

        input_rate[:] = (digit * (INPUT_STIM - INPUT_UNSTIM)) + INPUT_UNSTIM
        model.push_var_to_device("inp", "rate")

        out_voltage[:] = 0.0
        model.push_var_to_device('out', "V")

    # Advance simulation
    model.step_time()

    # If this is the LAST timestep of presenting the example
    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        model.pull_var_from_device("out", "SpikeCount")
        predicted_label = np.argmax(output_spike_count)
        true_label = y[example]

        print("\tExample=%u, true label=%u, predicted label=%u" % (example,
                                                                   true_label,
                                                                   predicted_label))

        if predicted_label == true_label:
            num_correct += 1

print("Accuracy %f%%" % ((num_correct / float(X.shape[0])) * 100.0))