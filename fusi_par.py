import numpy as np
import os
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class, create_custom_weight_update_class,
                               GeNNModel, init_var, init_connectivity)
from pygenn.genn_wrapper import NO_DELAY
from mlxtend.data import loadlocal_mnist
from collections import Counter
import csv

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

if_model_test = create_custom_neuron_class(
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
data_dir = "/home/p286814/pygenn/pygenn_scripts/mnist"
X, y = loadlocal_mnist(
    images_path=os.path.join(data_dir, 'train-images-idx3-ubyte'),
    labels_path=os.path.join(data_dir, 'train-labels-idx1-ubyte'))

print("Loading training images of size: " + str(X.shape))
print("Loading training labels of size: " + str(y.shape))

X_test, y_test = loadlocal_mnist(
    images_path=os.path.join(data_dir, 't10k-images-idx3-ubyte'),
    labels_path=os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

print("Loading testing images of size: " + str(X_test.shape))
print("Loading testing labels of size: " + str(y_test.shape))

NUM_INPUT = X.shape[1]
NUM_CLASSES = len(np.unique(y))
TEACHER_NUM = 20
OUTPUT_NEURON_NUM = 15

TARGET_NUM = OUTPUT_NEURON_NUM
NON_TARGET_NUM = OUTPUT_NEURON_NUM * (NUM_CLASSES - 1)

SPIKE_RATE_DIV = PRESENT_TIMESTEPS / 1000

# Values for initialisation of parameters in different models
if_init = {"V": 0.0}
poisson_init = {"rate": 1.0}

fusi_init = {"X": 0.0,
             "last_tpre": 0.0,
             "g": 0.0}
fusi_post_init = {"C": 2.0}

if_test_init = {"V": 0.0,
           "SpikeCount": 0.0}

WEIGHTS_PATH = "/home/p286814/pygenn/pygenn_scripts/weights"

# TEACHER_STRENGTH_R = np.arange(0.01, 0.081, 0.005)
# TEACHER_UNSTIM_STRENGTH_R = np.arange(0.005, 0.031, 0.0025)

TEACHER_STRENGTH = 1.0
TEACHER_UNSTIM_STRENGTH = 0.0

INH_V = 50
INPUT_UNSTIM = 2
INPUT_STIM = 50

TEACHER_V_R = list(range(7,21))      # vary this from 7 to 20
TEACHER_UNSTIM_V_R = list(range(6))    # vary this from 0 to 5

with open('spike_rates.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['exp', 'target', 'non_target', 'test_acc'])

for TEACHER_V in TEACHER_V_R:
    for TEACHER_UNSTIM_V in TEACHER_UNSTIM_V_R:

        EXPERIMENT = str(TEACHER_V) + "_" + str(TEACHER_UNSTIM_V)

        print("Experiment: " + EXPERIMENT)

        ######## TRAINING ##############

        # ----------------------------------------------------------------------------
        # Build model
        # ----------------------------------------------------------------------------
        # Create GeNN model
        model = GeNNModel("float", "tutorial_1")
        model.dT = TIMESTEP

        neurons_count = {"inp": NUM_INPUT,
                         "inh": 1000,
                         "out": NUM_CLASSES * OUTPUT_NEURON_NUM,
                         "teacher": NUM_CLASSES * TEACHER_NUM}

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

        teacher2out_mat = np.full((neurons_count["teacher"], neurons_count["out"]), TEACHER_UNSTIM_STRENGTH)
        fill_idx = [(i * OUTPUT_NEURON_NUM, (i + 1) * OUTPUT_NEURON_NUM) for i in range(NUM_CLASSES)]
        counter = 0
        for i in range(teacher2out_mat.shape[0]):
            teacher2out_mat[i, fill_idx[counter][0]:fill_idx[counter][1]] = TEACHER_STRENGTH
            if (i + 1) % TEACHER_NUM == 0:
                counter += 1

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

        non_target_spike_rates = []
        target_spike_rates = []

        while model.timestep < (PRESENT_TIMESTEPS * X.shape[0]):
            # Calculate the timestep within the presentation
            timestep_in_example = model.timestep % PRESENT_TIMESTEPS
            example = int(model.timestep // PRESENT_TIMESTEPS)

            # If this is the first timestep of presenting the example
            if timestep_in_example == 0:

                # init a data structure for plotting the raster plots for this example
                layer_spikes = np.empty(0)

                if example % 10000 == 0:
                    print("Example: " + str(example))

                # calculate the correct spiking rates for all populations
                digit = X[example, :].flatten()
                digit = np.divide(digit, np.amax(digit))
                active_pixels = np.count_nonzero(digit)

                inh_rate[:] = INH_V * active_pixels / NUM_INPUT
                model.push_var_to_device("inh", "rate")

                input_rate[:] = (digit * (INPUT_STIM - INPUT_UNSTIM)) + INPUT_UNSTIM
                model.push_var_to_device("inp", "rate")

                one_hot = np.full(neurons_count["teacher"], TEACHER_UNSTIM_V)
                chosen_class = y[example]
                one_hot[chosen_class * TEACHER_NUM:(chosen_class + 1) * TEACHER_NUM] = TEACHER_V
                teacher_rate[:] = one_hot
                model.push_var_to_device("teacher", "rate")

                out_voltage[:] = 0.0
                model.push_var_to_device('out', "V")

            # Advance simulation
            model.step_time()

            model.pull_current_spikes_from_device("out")
            layer_spikes = np.hstack((layer_spikes, neuron_layers["out"].current_spikes))

            # If this is the LAST timestep of presenting the example
            if timestep_in_example == (PRESENT_TIMESTEPS - 1):

                chosen_idx = list(range(chosen_class * OUTPUT_NEURON_NUM, (chosen_class + 1) * OUTPUT_NEURON_NUM))
                output_Counter = Counter(layer_spikes)
                counter = 0
                non_target_spiking = 0
                target_spiking = 0
                for n in range(neurons_count["out"]):
                    if n in chosen_idx:
                        if n in output_Counter:
                            target_spiking += output_Counter[n]
                    else:
                        if n in output_Counter:
                            non_target_spiking += output_Counter[n]

                non_target_rate = non_target_spiking / (SPIKE_RATE_DIV * NON_TARGET_NUM)
                target_rate = target_spiking / (SPIKE_RATE_DIV * TARGET_NUM)

                # if example % 500 == 0:
                #     print("Target rate: " + str(non_target_rate) + " Hz.")
                #     print("Non-target rate: " + str(target_rate) + " Hz.")

                non_target_spike_rates.append(non_target_rate)
                target_spike_rates.append(target_rate)

        print("Completed training.")

        target_final = sum(target_spike_rates) / len(target_spike_rates)
        non_target_final = sum(non_target_spike_rates) / len(non_target_spike_rates)

        # print("Target neurons spiking at: " + str(target_final) + " Hz.")
        # print("Non-target neurons spiking at: " + str(non_target_final) + " Hz.")

        CSV_DATA = [EXPERIMENT, target_final, non_target_final]

        model.pull_var_from_device(inp2out.name, "g")
        weight_values = inp2out.get_var_values("g")
        WT_FILENAME = os.path.join(WEIGHTS_PATH, EXPERIMENT + ".npy")
        np.save(WT_FILENAME, weight_values)

        ########### TESTING ##############

        # ----------------------------------------------------------------------------
        # Build model
        # ----------------------------------------------------------------------------
        # Create GeNN model
        model = GeNNModel("float", "tutorial_1")
        model.dT = TIMESTEP

        neurons_count = {"inp": NUM_INPUT,
                         "inh": 1000,
                         "out": NUM_CLASSES * OUTPUT_NEURON_NUM}

        neuron_layers = {}

        for k in neurons_count.keys():
            if k == "out":
                neuron_layers[k] = model.add_neuron_population(k, neurons_count[k],
                                                               if_model_test, IF_PARAMS, if_test_init)
            else:
                neuron_layers[k] = model.add_neuron_population(k, neurons_count[k],
                                                               poisson_model, {}, poisson_init)

        inp2out_w = np.load(WT_FILENAME)

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
        # Testing
        # ----------------------------------------------------------------------------
        # Get views to efficiently access state variables
        out_voltage = neuron_layers['out'].vars['V'].view
        input_rate = neuron_layers['inp'].vars['rate'].view
        inh_rate = neuron_layers['inh'].vars['rate'].view
        output_spike_count = neuron_layers["out"].vars["SpikeCount"].view

        num_correct = 0

        while model.timestep < (PRESENT_TIMESTEPS * X_test.shape[0]):
            # Calculate the timestep within the presentation
            timestep_in_example = model.timestep % PRESENT_TIMESTEPS
            example = int(model.timestep // PRESENT_TIMESTEPS)

            # If this is the first timestep of presenting the example
            if timestep_in_example == 0:

                if example % 1000 == 0:
                    print("Example: " + str(example))

                # calculate the correct spiking rates for all populations
                digit = X_test[example, :].flatten()
                digit = np.divide(digit, np.amax(digit))
                active_pixels = np.count_nonzero(digit)

                inh_rate[:] = INH_V * active_pixels / NUM_INPUT
                model.push_var_to_device("inh", "rate")

                input_rate[:] = (digit * (INPUT_STIM - INPUT_UNSTIM)) + INPUT_UNSTIM
                model.push_var_to_device("inp", "rate")

                out_voltage[:] = 0.0
                model.push_var_to_device('out', "V")

                output_spike_count[:] = 0
                model.push_var_to_device("out", "SpikeCount")

            # Advance simulation
            model.step_time()

            # If this is the LAST timestep of presenting the example
            if timestep_in_example == (PRESENT_TIMESTEPS - 1):

                model.pull_var_from_device("out", "SpikeCount")

                true_label = y_test[example]
                classwise_total_spikes = np.add.reduceat(output_spike_count,
                                                         np.arange(0, len(output_spike_count), OUTPUT_NEURON_NUM))
                predicted_label = np.argmax(classwise_total_spikes)

                if predicted_label == true_label:
                    num_correct += 1

        accuracy = (num_correct / float(X_test.shape[0])) * 100.0

        CSV_DATA.append(accuracy)

        with open('spike_rates.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_DATA)
