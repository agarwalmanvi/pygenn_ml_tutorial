###### Sample script to test implementation of Poisson neuron with changeable spiking frequency #######

from pygenn.genn_model import create_custom_neuron_class, GeNNModel

poisson_model = create_custom_neuron_class(
    "poisson_model",
    var_name_types=[("rate", "scalar"), ("spikeCount", "scalar")],
    sim_code="""
    """,
    reset_code="""
    $(spikeCount) += 1;
    """,
    threshold_condition_code="$(gennrand_uniform) >= exp(-$(rate) * 0.001 * DT)"
)

TIMESTEP = 1.0
PRESENT_TIMESTEPS = 1000

model = GeNNModel("float", "tutorial_1")
model.dT = TIMESTEP

poisson_init = {"rate": 30.0,
                "spikeCount": 0.0}

p = model.add_neuron_population("p", 1, poisson_model, {}, poisson_init)

model.build()
model.load()

while model.timestep < PRESENT_TIMESTEPS:
    model.step_time()

model.pull_var_from_device("p", "spikeCount")
spikeNum = p.vars["spikeCount"].view

print("Spike Rate: ")
print(spikeNum)
