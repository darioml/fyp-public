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
import scipy.io

# TEMPLATE FROM http://neuralensemble.org/docs/PyNN/examples/simple_STDP.html

# === Parameters ============================================================

cell_parameters = {
    "tau_m": 5.0,        # (ms)
    "tau_syn_E":15.0,    # (ms)
    "tau_syn_I":5.0,    # (ms)
    # "v_thresh": -35.0,   # (mV)
    "v_reset": -60.0,    # (mV)
    "v_rest": -60.0,     # (mV)
    "cm": 1.0,           # (nF)
    "tau_refrac": 20,    # (ms) long refractory period to prevent bursting
}

n = 256                  # number of synapses / number of presynaptic neurons
mid_n = 40
t_stop = 0               # defined later if is == 0
delay = 3.0              # (ms) synaptic time delay
episodes = 20;


# === Configure the simulator ===============================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--fit-curve", "Calculate the best-fit curve to the weight-delta_t measurements", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=0.01, min_delay=delay, max_delay=delay)


# === Build the network =====================================================

def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0])/1000 #from ns to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]

    return x,y,p,ts

(x_r, y_r, p_r, ts_r) = get_data('../data/mat/200ms_16_16_ws_norm_l2r.mat')
(x_l, y_l, p_l, ts_l) = get_data('../data/mat/200ms_16_16_ws_norm_r2l.mat')
(x_d, y_d, p_d, ts_d) = get_data('../data/mat/200ms_16_16_ws_norm_t2b.mat')
(x_u, y_u, p_u, ts_u) = get_data('../data/mat/200ms_16_16_ws_norm_b2t.mat')

# pre_loaded_weights = scipy.io.loadmat('weights.mat')

if t_stop == 0:
    t_stop = episodes * 4 * int(max(ts_u) - min(ts_u))
    print 'time will end at %i' % t_stop


def build_spike_from_dvs(ts, x, y, p): # let's use only on events
    def get_times(i):
        if type(i) == int:
            i = np.array(i).reshape(1)
        return [Sequence(ts[(y == np.floor(j/16)) & (x == j%16)]) for j in i]

    return get_times

# this could be done better
def build_spike_from_dvs_multipart(order, time, *args): # let's use only on events
    def get_times(i):
        if type(i) == int:
            i = np.array(i).reshape(1)
        final = []
        for j in i:
            rtn = []
            for idx in range(len(order)):
                loc = order[idx]
                test = np.array(time*idx + args[loc][3][(args[loc][1] == np.floor(j/16)) & (args[loc][0] == j%16)])
                if rtn == []:
                    rtn = test
                else:
                    rtn = np.concatenate( (rtn, test), 1 )
            final.append(Sequence(rtn))
        return final

    return get_times

dvs_spikes_1 = build_spike_from_dvs_multipart(np.random.randint(0,4,episodes), 200,
                                              (x_r, y_r, p_r, ts_r),
                                              (x_l, y_l, p_l, ts_l),
                                              (x_d, y_d, p_d, ts_d),
                                              (x_u, y_u, p_u, ts_u))
dvs_spikes_2 = build_spike_from_dvs(ts_r, x_r, y_r, p_r)



# presynaptic population
p1 = sim.Population(n, sim.SpikeSourceArray(spike_times=dvs_spikes_1),
                    label="presynaptic")

# fourty layered postsynaptic neuron
p2 = sim.Population(mid_n, sim.IF_cond_exp(**cell_parameters),
                    initial_values={"v": cell_parameters["v_reset"]}, label="postsynaptic")

# set up the model
stdp_model = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(tau_plus=15.0, tau_minus=25.0,
                                                    A_plus=0.000002, A_minus=0.000001),
                weight_dependence=sim.AdditiveWeightDependence(w_min=-0.02, w_max=0.02),
                weight=pyNN.random.RandomDistribution('normal', mu=0.00175, sigma=0.0003),
                # weight=pre_loaded_weights['aa'],
                delay=delay,
                dendritic_delay_fraction=0)

connections = sim.Projection(p1, p2, sim.AllToAllConnector(), stdp_model)
connections = sim.Projection(p2, p2, sim.AllToAllConnector(), sim.StaticSynapse(weight=0.1), receptor_type='inhibitory')

# == Instrument the network =================================================

p1.record('spikes')
p2.record('spikes')
# p2.record(['spikes', 'v'])

# === Run the simulation =====================================================

def report_time(t):
    print "The time is %gms" % t
    return t + 500

sim.run(t_stop, callbacks=[report_time])

scipy.io.savemat('weights.mat', {
    'la':connections.get('weight', format='list', with_address=True),
    'ln':connections.get('weight', format='list', with_address=False),
    'aa':connections.get('weight', format='array', with_address=True),
    'an':connections.get('weight', format='array', with_address=False)
})

# === Save the results, optionally plot a figure =============================

filename = normalized_filename("Results", "ball_trajectories", "pkl", options.simulator)
p2.write_data(filename, annotations={'script_name': __file__})

presynaptic_data = p1.get_data().segments[0]
postsynaptic_data = p2.get_data().segments[0]
print("Post-synaptic spike times: %s" % postsynaptic_data.spiketrains[0])


if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel, DataTable
    figure_filename =   filename.replace("pkl", "png")
    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(presynaptic_data.spiketrains,
              yticks=True, markersize=0.2, xlim=((episodes-10)*(t_stop/episodes), t_stop)), #(episodes-10)*(t_stop/episodes)
        Panel(postsynaptic_data.spiketrains,
              yticks=True, markersize=0.2, xlim=((episodes-10)*(t_stop/episodes), t_stop)),
        # membrane potential of the postsynaptic neuron
        # Panel(postsynaptic_data.filter(name='v')[0],
        #       ylabel="Membrane potential (mV)",
        #       yticks=True, xlim=(0, t_stop)), #data_labels=[p2.label],
        title="STDP & DVS Data"
    ).save(figure_filename)
    print(figure_filename)

# print postsynaptic_data.filter(name='v')
# scipy.io.savemat('test', {'t':postsynaptic_data.filter(name='v')[0]})

# === Clean up and quit ========================================================

sim.end()