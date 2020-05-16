import numpy as np
import os
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class, create_custom_weight_update_class,
                               GeNNModel, init_var, init_connectivity)
from pygenn.genn_wrapper import NO_DELAY
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import pickle as pkl
from collections import Counter

# TODO Parameters to vary:
# Teacher strength
# presentation time
# no. of teacher neurons

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
PRESENT_TIMESTEPS = 300
INPUT_CURRENT_SCALE = 1.0 / 100.0
OUTPUT_CURRENT_SCALE = 10.0

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

poisson_model = create_custom_neuron_class(
    "poisson_model",
    var_name_types=[("rate", "scalar")],
    sim_code="""
    """,
    reset_code="""
    """,
    threshold_condition_code="$(gennrand_uniform) >= exp(-$(rate) * 0.001 * DT)"
)

fusi_model = create_custom_weight_update_class(
    "fusi_model",
    param_names=["tauC", "a", "b", "thetaV", "thetaLUp", "thetaLDown", "thetaHUp", "thetaHDown",
                 "thetaX", "alpha", "beta", "Xmax", "Xmin", "JC", "Jplus", "Jminus"],
    var_name_types=[("X", "scalar"), ("last_tpre", "scalar"), ("g", "scalar")],
    post_var_name_types=[("C", "scalar")],
    sim_code="""
    $(addToInSyn, $(g));
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
    $(g) = ($(X) > $(thetaX)) ? $(Jplus) : $(Jminus);
    $(last_tpre) = $(t);
    """,
    post_spike_code="""
    const scalar dt = $(t) - $(sT_post);
    $(C) = ($(C) * exp(-dt / $(tauC))) + $(JC);
    """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True
)

# ----------------------------------------------------------------------------
# Import training data
# ----------------------------------------------------------------------------
data_dir = "/home/manvi/Documents/pygenn_ml_tutorial/mnist"
X, y = loadlocal_mnist(
    images_path=os.path.join(data_dir, 'train-images-idx3-ubyte'),
    labels_path=os.path.join(data_dir, 'train-labels-idx1-ubyte'))

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

NUM_INPUT = X.shape[1]
NUM_CLASSES = len(np.unique(y))
TEACHER_NUM = 50
OUTPUT_NEURON_NUM = 15

neurons_count = {"inp": NUM_INPUT,
                 "inh": 1000,
                 "out": NUM_CLASSES * OUTPUT_NEURON_NUM,
                 "teacher": NUM_CLASSES * TEACHER_NUM}

# Values for initialisation of parameters in different models
if_init = {"V": 0.0}
poisson_init = {"rate": 1.0}

fusi_init = {"X": 0.0,
             "last_tpre": 0.0,
             "g": np.random.choice(2, (neurons_count["inp"], neurons_count["out"])).flatten()}
fusi_post_init = {"C": 2.0}

neuron_layers = {}

for k in neurons_count.keys():
    if k == "out":
        neuron_layers[k] = model.add_neuron_population(k, neurons_count[k],
                                                       if_model, IF_PARAMS, if_init)
    else:
        neuron_layers[k] = model.add_neuron_population(k, neurons_count[k],
                                                       poisson_model, {}, poisson_init)

# fully connected input to output
inp2out = model.add_synapse_population(
    "inp2out", "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers['inp'], neuron_layers['out'],
    fusi_model, FUSI_PARAMS, fusi_init, {}, fusi_post_init,
    "DeltaCurr", {}, {})

# fully connected inhibitory to output
inh2out = model.add_synapse_population(
    "inh2out", "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers['inh'], neuron_layers['out'],
    "StaticPulse", {}, {"g": -0.035}, {}, {},
    "DeltaCurr", {}, {})

TEACHER_STRENGTH = 0.1

teacher2out_mat = np.zeros((neurons_count["out"], neurons_count["teacher"]))
fill_idx = [(i*TEACHER_NUM, (i+1)*TEACHER_NUM) for i in range(NUM_CLASSES)]
counter = 0
for i in range(teacher2out_mat.shape[0]):
    teacher2out_mat[i, fill_idx[counter][0]:fill_idx[counter][1]] = TEACHER_STRENGTH
    if (i+1) % OUTPUT_NEURON_NUM == 0:
        counter += 1
teacher2out_mat = teacher2out_mat.transpose()

teacher2out = model.add_synapse_population(
    "teacher2out", "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers['teacher'], neuron_layers['out'],
    "StaticPulse", {}, {"g": teacher2out_mat.flatten()}, {}, {},
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
teacher_rate = neuron_layers['teacher'].vars['rate'].view

all_spike_rates = []

INH_V = 50
INPUT_UNSTIM = 2
INPUT_STIM = 50
TEACHER_V = 150

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

        # calculate the correct spiking rates for all populations
        digit = X[example, :].flatten()
        digit = np.divide(digit, np.amax(digit))
        active_pixels = np.count_nonzero(digit)

        inh_rate[:] = INH_V * active_pixels / NUM_INPUT
        model.push_var_to_device("inh", "rate")

        input_rate[:] = (digit * (INPUT_STIM - INPUT_UNSTIM)) + INPUT_UNSTIM
        model.push_var_to_device("inp", "rate")

        one_hot = np.zeros(neurons_count["teacher"])
        chosen_class = y[example]
        one_hot[chosen_class*TEACHER_NUM:(chosen_class+1)*TEACHER_NUM] = TEACHER_V
        teacher_rate[:] = one_hot
        model.push_var_to_device("teacher", "rate")

        out_voltage[:] = 0.0
        model.push_var_to_device('out', "V")

    # Advance simulation
    model.step_time()

    for idx, layer in enumerate(neuron_layers):
        # Download spikes
        model.pull_current_spikes_from_device(layer)

        # Add to data structure
        spike_times = np.ones_like(neuron_layers[layer].current_spikes) * model.t
        layer_spikes[idx] = (np.hstack((layer_spikes[idx][0], neuron_layers[layer].current_spikes)),
                           np.hstack((layer_spikes[idx][1], spike_times)))

    # If this is the LAST timestep of presenting the example
    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        # Calculate spiking rate
        for i, layer in enumerate(neuron_layers):

            if layer == "out":
                chosen_idx = list(range(chosen_class*OUTPUT_NEURON_NUM,(chosen_class+1)*OUTPUT_NEURON_NUM))
                class_Counter = Counter(layer_spikes[i][0])
                total_spikes = 0
                for idx in chosen_idx:
                    total_spikes += class_Counter[idx]
                s = (total_spikes / ((PRESENT_TIMESTEPS * OUTPUT_NEURON_NUM) / 1000))
                all_spike_rates.append(s)

                print("Avg spiking rate for neurons of chosen output class (" + str(chosen_class) + ") : " + str(s) + " Hz")

        # Make a plot every 10th example
        if example % 10 == 0:

            print("Creating raster plot")

            # Create a plot with axes for each
            fig, axes = plt.subplots(len(neuron_layers), sharex=True)
            fig.tight_layout(pad=2.0)

            # Loop through axes and their corresponding neuron populations
            for a, s, l in zip(axes, layer_spikes, list(neuron_layers.values())):
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
            save_filename = os.path.join('example' + str(example) + '.png')
            plt.savefig(save_filename)

print("Avg spiking rate: " + str(sum(all_spike_rates) / len(all_spike_rates)))

# with open('spike_rates.pkl', 'wb') as f:
#     pkl.dump(all_spike_rates, f)

print("Completed training.")

# model.pull_var_from_device(inp2out.name, "g")
# weight_values = inp2out.get_var_values("g")
# print(type(weight_values))
# print(weight_values.shape)
# np.save("fusi.npy", weight_values)
