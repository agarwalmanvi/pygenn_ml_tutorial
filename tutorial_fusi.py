import numpy as np
from os import path

from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class, create_custom_weight_update_class,
                               GeNNModel, init_var, init_connectivity)
from pygenn.genn_wrapper import NO_DELAY
from mlxtend.data import loadlocal_mnist
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
# PRESENT_TIMESTEPS = 100
# INPUT_CURRENT_SCALE = 1.0 / 100.0
# OUTPUT_CURRENT_SCALE = 10.0
# NUM_CLASSES = 10
PRESENT_TIMESTEPS = 300

# ----------------------------------------------------------------------------
# Custom GeNN models
# ----------------------------------------------------------------------------
if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vtheta", "lambda", "Vrest", "Vreset"],
    var_name_types=[("V", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code="""
    if ($(V) >= $(Vtheta)) {
        $(V) = $(Vreset);
    }
    $(V) += (-$(lambda) + $(Isyn)) * DT;
    $(V) = fmax($(V), $(Vrest));
    """,
    reset_code="""
    $(SpikeCount)++;
    """,
    threshold_condition_code="$(V) >= $(Vtheta)"
)

fusi_model = create_custom_weight_update_class(
    "fusi_model",
    param_names=["tauC", "a", "b", "thetaV", "thetaLUp", "thetaLDown", "thetaHUp", "thetaHDown",
                 "thetaX", "alpha", "beta", "Xmax", "Xmin", "JC", "Jplus", "Jminus"],
    var_name_types=[("X", "scalar"), ("last_tpre", "scalar"), ("decayC", "scalar")],
    post_var_name_types=[("C", "scalar")],
    sim_code="""
    $(addToInSyn, ($(X) > $(thetaX)) ? $(Jplus) : $(Jminus));
    const scalar dt = $(t) - $(sT_post);
    $(decayC) = $(C) * exp(-dt / $(tauC));
    if ($(V_post) > $(thetaV) && $(thetaLUp) < $(decayC) && $(decayC) < $(thetaHUp)) {
        $(X) += $(a);
    }
    else if ($(V_post) <= $(thetaV) && $(thetaLDown) < $(decayC) && $(decayC) < $(thetaHDown)) {
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

########## Reproduce Fig 1 from paper ################

# Notes: done -> optimizations, sparse init with one-to-one connectivity
#       next -> set sparse connectivity, parallel exp within models,
#               setup data structure for recording everything.

repeats = 50

for run in range(repeats):

    print("run " + str(run))

    model = GeNNModel("float", "fig1")
    model.dT = TIMESTEP

    presyn_params = {"rate" : 50.0}
    extra_poisson_params = {"rate" : 100.0}
    poisson_init = {"timeStepToSpike" : 0.0}
    if_init = {"V": 0.0, "SpikeCount": 0}
    fusi_init = {"X": 0.0,
                 "last_tpre": 0.0}
    fusi_post_init = {"C": 2.0,
                 "last_spike": 0.0}

    presyn = model.add_neuron_population("presyn", 100, "PoissonNew", presyn_params, poisson_init)
    postsyn = model.add_neuron_population("postsyn", 100, if_model, IF_PARAMS, if_init)
    extra_poisson = model.add_neuron_population("extra_poisson", 100*10, "PoissonNew",
                                                extra_poisson_params, poisson_init)

    pre2post = model.add_synapse_population(
                "pre2post", "SPARSE_INDIVIDUALG", NO_DELAY,
                presyn, postsyn,
                fusi_model, FUSI_PARAMS, fusi_init, {}, fusi_post_init,
                "DeltaCurr", {}, init_connectivity("OneToOne",{}))

    extra_poisson2post = model.add_synapse_population(
                "extra_poisson2post", "SPARSE_INDIVIDUALG", NO_DELAY,
                extra_poisson, postsyn,
                "StaticPulse", {}, {"g": 0.05}, {}, {},
                "DeltaCurr", {}, init_connectivity("OneToOne",{}))

    model.build()
    model.load()

    print("Simulating")

    neuron_layers = [presyn, postsyn, extra_poisson]

    # initialize arrays for storing all things we want to plot
    layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]
    X = np.array([0.0]) # TODO how to get initial value of X from init_var
    postsyn_V = np.array([if_init["V"]])
    C = np.array([fusi_post_init["C"]])

    while model.timestep < PRESENT_TIMESTEPS:
        model.step_time()

        # Record spikes
        for i, l in enumerate(neuron_layers):
            # Download spikes
            model.pull_current_spikes_from_device(l.name)

            # Add to data structure
            spike_times = np.ones_like(l.current_spikes) * model.t
            layer_spikes[i] = (np.hstack((layer_spikes[i][0], l.current_spikes)),
                               np.hstack((layer_spikes[i][1], spike_times)))

        # Record value of X
        model.pull_var_from_device("pre2post", "X")
        X_val = pre2post.get_var_values("X")
        X = np.concatenate((X, X_val), axis=0)

        # Record value of postsyn_V
        model.pull_var_from_device("postsyn", "V")
        V_val = postsyn.vars["V"].view
        postsyn_V = np.concatenate((postsyn_V, V_val), axis=0)

        # Record value of C
        model.pull_var_from_device("pre2post", "C")
        C_val = pre2post.post_vars["C"].view
        C = np.concatenate((C, C_val), axis=0)

    postsyn_spike_rate = len(layer_spikes[1][1]) / (PRESENT_TIMESTEPS / 1000)

    # # Create plot
    # fig, axes = plt.subplots(4, sharex=True)
    # fig.tight_layout(pad=2.0)
    #
    # # plot presyn spikes
    # presyn_spike_times = layer_spikes[0][1]
    # for s in presyn_spike_times:
    #     axes[0].set_xlim((0,PRESENT_TIMESTEPS))
    #     axes[0].axvline(s)
    # axes[0].title.set_text("Presynaptic spikes")
    #
    # # plot X
    # axes[1].title.set_text("Synaptic internal variable X(t)")
    # axes[1].plot(X)
    # axes[1].set_ylim((0,1))
    # axes[1].axhline(0.5, linestyle="--", color="black", linewidth=0.5)
    # axes[1].set_yticklabels(["0", "$\\theta_X$", "1"])
    #
    # # plot postsyn V
    # axes[2].title.set_text('Postsynaptic voltage V(t) (Spike rate: ' + str(postsyn_spike_rate) + " Hz)")
    # axes[2].plot(postsyn_V)
    # axes[2].set_ylim((0,1.2))
    # axes[2].axhline(1, linestyle="--", color="black", linewidth=0.5)
    # axes[2].axhline(0.8, linestyle="--", color="black", linewidth=0.5)
    # postsyn_spike_times = layer_spikes[1][1]
    # for s in postsyn_spike_times:
    #     axes[2].axvline(s, color="red", linewidth=0.5)
    #
    # # plot C
    # axes[3].plot(C)
    # axes[3].title.set_text("Calcium variable C(t)")
    # for i in [3, 4, 13]:
    #     axes[3].axhline(i, linestyle="--", color="black", linewidth=0.5)
    #
    # save_filename = "fig1" + str(run) + ".png"
    # plt.savefig(save_filename)
    # plt.close()

# print("Internal synaptic variable: ")
# print(X)
# print("Postsynaptic depolarization: ")
# print(postsyn_V)
# print("Calcium variable: ")
# print(C)


# # Current source model which injects current with a magnitude specified by a state variable
# cs_model = create_custom_current_source_class(
#     "cs_model",
#     var_name_types=[("magnitude", "scalar")],
#     injection_code="$(injectCurrent, $(magnitude));")
#
# # ----------------------------------------------------------------------------
# # Import training data
# # ----------------------------------------------------------------------------
# data_dir = "/home/manvi/Documents/pygenn_ml_tutorial/mnist"
# X, y = loadlocal_mnist(
#     images_path=path.join(data_dir, 'train-images-idx3-ubyte'),
#     labels_path=path.join(data_dir, 'train-labels-idx1-ubyte'))
#
# print("Loading training images of size: " + str(X.shape))
# print("Loading training labels of size: " + str(y.shape))
#
# # ----------------------------------------------------------------------------
# # Build model
# # ----------------------------------------------------------------------------
# # Create GeNN model
# model = GeNNModel("float", "tutorial_1")
# model.dT = TIMESTEP
#
# # Initial values for initialisation
# if_init = {"V": 0.0, "SpikeCount": 0}
# fusi_init = {"X": init_var("Uniform", {"min": FUSI_PARAMS["Xmin"], "max": FUSI_PARAMS["Xmax"]}),
#              "tpre": 0.0}
# fusi_post_init = {"C": 0.0}
#
# inp_neuron_num = X.shape[0]
# out_neuron_num = NUM_CLASSES
#
# neuron_layer_names = {"input_layer": inp_neuron_num,
#                       "inh_layer": inp_neuron_num,
#                       "output_layer": out_neuron_num}
#
# neuron_layers = []
#
# # Create neuron layers
# for i in neuron_layer_names:
#     neuron_layers.append(model.add_neuron_population(i, neuron_layer_names[i], if_model,
#                                                      IF_PARAMS, if_init))
#
# synapses = []
#
# # TODO one-to-one connections?
# # Create non-plastic synapses for input->inhibitory and inhibitory->output connections
# synapses.append(model.add_synapse_population(
#     "inp2inh", "DENSE_INDIVIDUALG", NO_DELAY,
#     neuron_layers[0], neuron_layers[1],
#     "StaticPulse", {}, {"g": 1.0}, {}, {},
#     "DeltaCurr", {}, {}))
# synapses.append(model.add_synapse_population(
#     "inh2out", "DENSE_INDIVIDUALG", NO_DELAY,
#     neuron_layers[1], neuron_layers[2],
#     "StaticPulse", {}, {"g": -1.0}, {}, {},
#     "DeltaCurr", {}, {}))
#
# # Create plastic synapses between input and output layer
# synapses.append(model.add_synapse_population(
#     "inp2out", "DENSE_INDIVIDUALG", NO_DELAY,
#     neuron_layers[0], neuron_layers[-1],
#     fusi_model, FUSI_PARAMS, fusi_init, {}, fusi_post_init,
#     "DeltaCurr", {}, {}))
#
# # Create current source to deliver input to first layers of neurons
# current_input = model.add_current_source("current_input", cs_model,
#                                          "input_layer", {}, {"magnitude": 0.0})
#
# # Create current source to deliver target output to last layer of neurons
# current_output = model.add_current_source("current_output", cs_model,
#                                           "output_layer", {}, {"magnitude": 0.0})
#
# # Build and load our model
# model.build()
# model.load()
#
# # ----------------------------------------------------------------------------
# # Training
# # ----------------------------------------------------------------------------
#
# # Get views to efficiently access state variables
# current_input_magnitude = current_input.vars["magnitude"].view
# current_output_magnitude = current_output.vars["magnitude"].view
# layer_voltages = [l.vars["V"].view for l in neuron_layers]
#
# prefix = "fusi"
# save_png_dir = "/home/manvi/Documents/pygenn_ml_tutorial/imgs/" + prefix
#
# # Simulate
# while model.timestep < (PRESENT_TIMESTEPS * X.shape[0]):
#     # Calculate the timestep within the presentation
#     timestep_in_example = model.timestep % PRESENT_TIMESTEPS
#     example = int(model.timestep // PRESENT_TIMESTEPS)
#
#     # If this is the first timestep of presenting the example
#     if timestep_in_example == 0:
#
#         # init a data structure for plotting the raster plots for this example
#         layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]
#
#         if example % 100 == 0:
#             print("Example: " + str(example))
#
#         current_input_magnitude[:] = X[example, :].flatten() * INPUT_CURRENT_SCALE
#         one_hot = np.zeros((NUM_CLASSES))
#         one_hot[y[example]] = 1
#         current_output_magnitude[:] = one_hot.flatten() * OUTPUT_CURRENT_SCALE
#         model.push_var_to_device("current_input", "magnitude")
#         model.push_var_to_device("current_output", "magnitude")
#
#         # Loop through all layers and their corresponding voltage views
#         for l, v in zip(neuron_layers, layer_voltages):
#             # Manually 'reset' voltage
#             v[:] = 0.0
#
#             # Upload
#             model.push_var_to_device(l.name, "V")
#
#     # Advance simulation
#     model.step_time()
#
#     # populate the raster plot data structure with the spikes of this example and this timestep
#     for i, l in enumerate(neuron_layers):
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
#         # Make a plot every 10000th example
#         if example % 10000 == 0:
#
#             print("Creating raster plot")
#
#             # Create a plot with axes for each
#             fig, axes = plt.subplots(len(neuron_layers), sharex=True)
#
#             # Loop through axes and their corresponding neuron populations
#             for a, s, l in zip(axes, layer_spikes, neuron_layers):
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
#
#             # Show plot
#             save_filename = path.join(save_png_dir, 'example' + str(example) + '.png')
#             plt.savefig(save_filename)
#
# print("Completed training.")
#
# for i, l in enumerate(synapses):
#     model.pull_var_from_device(l.name, "X")
#
#     weight_values = l.get_var_values("X")
#     print(type(weight_values))
#     print(weight_values.shape)
#     np.save(prefix + "w1_" + str(i) + "_" + str(i + 1) + ".npy", weight_values)
#
# print("Dumped data.")
