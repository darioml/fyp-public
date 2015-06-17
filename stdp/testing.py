from __future__ import division
from math import exp
import numpy
import neo
from quantities import ms
import pyNN as pyNN
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import DataTable
from pyNN.parameters import Sequence
np = numpy
import scipy

# TEMPLATE FROM http://neuralensemble.org/docs/PyNN/examples/simple_STDP.html

# === Parameters ============================================================

cell_parameters = {
    "tau_m": 5.0,        # (ms)
    "tau_syn_E":15.0,    # (ms)
    # "v_thresh": -35.0,   # (mV)
    "v_reset": -60.0,    # (mV)
    "v_rest": -60.0,     # (mV)
    "cm": 1.0,           # (nF)
    "tau_refrac": 20,    # (ms) long refractory period to prevent bursting
}

n = 256                  # number of synapses / number of presynaptic neurons
t_stop = 0               # defined later if is == 0
delay = 3.0              # (ms) synaptic time delay


# === Configure the simulator ===============================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--fit-curve", "Calculate the best-fit curve to the weight-delta_t measurements", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=0.01, min_delay=delay, max_delay=delay)


# === Build the network =====================================================

dvs_data = scipy.io.loadmat('../../fyp-python-aer-lib/all_directions.mat')

ts = dvs_data['ts'][0]
ts = (ts - ts[0])/1000 #from ns to ms
x = dvs_data['X'][0]
y = dvs_data['Y'][0]
p = dvs_data['t'][0]

if t_stop == 0:
    t_stop = int(max(ts) - min(ts))


def build_spike_from_dvs(ts, x, y, p): # let's use only on events
    def get_times(i):
        if type(i) == int:
            i = np.array(i).reshape(1)
        return [Sequence(ts[(y == np.floor(j/16)) & (x == j%16)]) for j in i]

    return get_times

dvs_spikes = build_spike_from_dvs(ts,x,y,p)

# presynaptic population
p1 = sim.Population(n, sim.SpikeSourceArray(spike_times=dvs_spikes),
                    label="presynaptic")

# fourty layered postsynaptic neuron
p2 = sim.Population(20, sim.IF_cond_exp(**cell_parameters),
                    initial_values={"v": cell_parameters["v_reset"]}, label="postsynaptic")

# set up the model
stdp_model = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(tau_plus=10.0, tau_minus=10.0,
                                                    A_plus=0.001, A_minus=0.0012),
                weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.02),
                weight=pyNN.random.RandomDistribution('normal', mu=0.001, sigma=0.001),
                delay=delay,
                dendritic_delay_fraction=0)

connections = sim.Projection(p1, p2, sim.AllToAllConnector(), stdp_model)

# == Instrument the network =================================================

p1.record('spikes')
p2.record(['spikes', 'v'])

# === Run the simulation =====================================================

sim.run(t_stop)

# === Save the results, optionally plot a figure =============================

filename = normalized_filename("Results", "simple_stdp", "pkl", options.simulator)
p2.write_data(filename, annotations={'script_name': __file__})

presynaptic_data = p1.get_data().segments[0]
postsynaptic_data = p2.get_data().segments[0]
print("Post-synaptic spike times: %s" % postsynaptic_data.spiketrains[0])

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel, DataTable
    figure_filename = filename.replace("pkl", "png")
    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(presynaptic_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(0, t_stop)),
        Panel(postsynaptic_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(0, t_stop)),
        # membrane potential of the postsynaptic neuron
        Panel(postsynaptic_data.filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[p2.label], yticks=True, xlim=(0, t_stop)),
        title="STDP & DVS Data"
    ).save(figure_filename)
    print(figure_filename)


# === Clean up and quit ========================================================

sim.end()