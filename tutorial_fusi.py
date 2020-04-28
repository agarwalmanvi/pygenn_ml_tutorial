import numpy as np
import csv
import os
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class, create_custom_weight_update_class,
                               GeNNModel, init_var, init_connectivity)
from pygenn.genn_wrapper import NO_DELAY

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

########## Reproduce Fig 1 from paper ################

# Notes: done -> optimizations, sparse init with one-to-one connectivity
#       next -> set sparse connectivity, parallel exp within models,
#               setup data structure for recording everything.

pre_spike_rate = [10.0, 20.0, 30.0, 40.0, 50.0]
post_spike_weight = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

# pre_spike_rate = [50.0]
# post_spike_weight = [0.08]

for pre_rate in pre_spike_rate:

    print("\n")
    print("Mean spike rate for presynaptic neuron: " + str(pre_rate))

    for post_weight in post_spike_weight:

        print("Weight for presynaptic neuron: " + str(post_weight))

        iterations = 1000

        for iter_num in range(iterations):

            if iter_num % 50 == 0:
                print("Iteration number: " + str(iter_num))

            model = GeNNModel("float", "fig1")
            model.dT = TIMESTEP

            presyn_params = {"rate" : pre_rate}
            extra_poisson_params = {"rate" : 100.0}
            poisson_init = {"timeStepToSpike" : 0.0}
            if_init = {"V": 0.0}
            fusi_init = {"X": 0.0,
                         "last_tpre": 0.0}
            fusi_post_init = {"C": 2.0}

            n = 1000

            presyn = model.add_neuron_population("presyn", n, "PoissonNew", presyn_params, poisson_init)
            postsyn = model.add_neuron_population("postsyn", n, if_model, IF_PARAMS, if_init)
            extra_poisson = model.add_neuron_population("extra_poisson", n*10, "PoissonNew",
                                                        extra_poisson_params, poisson_init)

            w = np.zeros((n, n * 10))
            for i in range(n):
                w[i, i * 10:(i + 1) * 10] = post_weight

            pre2post = model.add_synapse_population(
                "pre2post", "SPARSE_INDIVIDUALG", NO_DELAY,
                presyn, postsyn,
                fusi_model, FUSI_PARAMS, fusi_init, {}, fusi_post_init,
                "DeltaCurr", {}, {}, init_connectivity("OneToOne", {}))

            extra_poisson2post = model.add_synapse_population(
                "extra_poisson2post", "DENSE_INDIVIDUALG", NO_DELAY,
                extra_poisson, postsyn,
                "StaticPulse", {}, {"g": w.flatten()}, {}, {},
                "DeltaCurr", {}, {})

            model.build()
            model.load()

            # print("Simulating")

            neuron_layers = [postsyn]

            # initialize arrays for storing all things we want to plot
            layer_spikes = [(np.empty(0)) for _ in enumerate(neuron_layers)]
            # X = np.array([fusi_init["X"]])
            # postsyn_V = np.array([if_init["V"]])
            # C = np.array([fusi_post_init["C"]])

            # data structures to put into a csv


            while model.timestep < PRESENT_TIMESTEPS:
                model.step_time()

                # Record spikes
                for layer_i, l in enumerate(neuron_layers):
                    # Download spikes
                    model.pull_current_spikes_from_device(l.name)

                    # Add to data structure
                    layer_spikes[layer_i] = (np.hstack((layer_spikes[layer_i], l.current_spikes)))

                # # Record value of X
                # model.pull_var_from_device("pre2post", "X")
                # X_val = pre2post.get_var_values("X")
                # X = np.concatenate((X, X_val), axis=0)
                #
                # # Record value of postsyn_V
                # model.pull_var_from_device("postsyn", "V")
                # V_val = postsyn.vars["V"].view
                # postsyn_V = np.concatenate((postsyn_V, V_val), axis=0)
                #
                # # Record value of C
                # model.pull_var_from_device("pre2post", "C")
                # C_val = pre2post.post_vars["C"].view
                # C = np.concatenate((C, C_val), axis=0)

            unique, counts = np.unique(layer_spikes[0], return_counts=True)
            d = dict(zip(unique, counts))
            for k in d:
                d[k] = d[k] / (PRESENT_TIMESTEPS / 1000)
            model.pull_var_from_device("pre2post", "X")
            X_val = pre2post.vars["X"].view

            filename = str(pre_rate) + '_' + str(post_weight) + '.csv'
            filepath = "/home/manvi/Documents/pygenn_ml_tutorial/fusi_data/LTP"

            if iter_num == 0:
                with open(os.path.join(filepath, filename), 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["post_spike_rate", "LTP_success"])
                    for idx in d.keys():
                        x = X_val[int(idx)]
                        s = 1 if x > 0.5 else 0
                        writer.writerow([d[idx], s])
            else:
                with open(os.path.join(filepath, filename), 'a+', newline="") as f:
                    writer = csv.writer(f)
                    for idx in d.keys():
                        x = X_val[int(idx)]
                        s = 1 if x > 0.5 else 0
                        writer.writerow([d[idx], s])

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
