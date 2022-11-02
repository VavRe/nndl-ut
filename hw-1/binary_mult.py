import numpy as np
import itertools
from mpneuron import MPNeuron



def calculate_mult_network(input_):

    ##################
    ##################
    ####### h0 #######
    ##################
    ##################
    w_h0 = np.array([0,0.5,0,0.5])
    h0 = MPNeuron(1, w_h0)
    h0 = h0.activated_output(input_)

    ##################
    ##################
    ####### h1 #######
    ##################
    ##################
    # AND 1 Neuron
    w_neuron_1 = np.array([0,0.5,0.5,0])
    neuron_1 = MPNeuron(1, w_neuron_1)
    activated_neuron_1 = neuron_1.activated_output(input_)
    # AND 2 Neuron
    w_neuron_2 = np.array([0.5,0,0,0.5])
    neuron_2 = MPNeuron(1,w_neuron_2)
    activated_neuron_2 = neuron_2.activated_output(input_)
    # AND/NOT 3 Neuron
    w_neuron_3 = np.array([1,-1])
    neuron_3 = MPNeuron(1,w_neuron_3)
    neuron_3_input = np.array([activated_neuron_1,activated_neuron_2])
    activated_neuron_3 = neuron_3.activated_output(neuron_3_input)
    # AND/NOT 4 Neuron
    w_neuron_4 = np.array([-1,1])
    neuron_4 = MPNeuron(1,w_neuron_4)
    neuron_4_input = np.array([activated_neuron_1,activated_neuron_2])
    activated_neuron_4 = neuron_4.activated_output(neuron_4_input)
    # h1
    h1_w = np.array([1,1])
    h1_neuron = MPNeuron(1,h1_w)
    h1_input = np.array([activated_neuron_3, activated_neuron_4])
    h1 = h1_neuron.activated_output(h1_input)

    ##################
    ##################
    ####### h2 #######
    ##################
    ##################
    # AND 1 Neuron
    w_neuron_1 = np.array([0,0.5,0.5,0])
    neuron_1 = MPNeuron(1, w_neuron_1)
    activated_neuron_1 = neuron_1.activated_output(input_)
    # AND 2 Neuron
    w_neuron_2 = np.array([0.5,0,0,0.5])
    neuron_2 = MPNeuron(1,w_neuron_2)
    activated_neuron_2 = neuron_2.activated_output(input_)
    # AND 3 Neuron
    w_neuron_3 = np.array([0.5,0.5])
    neuron_3 = MPNeuron(1,w_neuron_3)
    neuron_3_input = [activated_neuron_1,activated_neuron_2]
    activated_neuron_3 = neuron_3.activated_output(neuron_3_input)
    # AND 4 Neuron
    w_neuron_4 = np.array([0.5,0.5])
    neuron_4 = MPNeuron(1,w_neuron_4)
    neuron_4_input = np.array([input_[0],input_[2]])
    activated_neuron_4 = neuron_4.activated_output(neuron_4_input)
    # AND/NOT 5 Neuron
    w_neuron_5 = np.array([1,-1])
    neuron_5 = MPNeuron(1,w_neuron_5)
    neuron_5_input = np.array([activated_neuron_3,activated_neuron_4])
    activated_neuron_5 = neuron_5.activated_output(neuron_5_input)
    # AND/NOT 6 Neuron
    w_neuron_6 = np.array([-1,1])
    neuron_6 = MPNeuron(1,w_neuron_6)
    neuron_6_input = np.array([activated_neuron_3,activated_neuron_4])
    activated_neuron_6 = neuron_6.activated_output(neuron_6_input)

    # h2
    h2_w = np.array([1,1])
    h2_neuron = MPNeuron(1,h2_w)
    h2_input = np.array([activated_neuron_5, activated_neuron_6])
    h2 = h2_neuron.activated_output(h2_input)
    ##################
    ##################
    ####### h3 #######
    ##################
    ##################
    # AND 1 Neuron
    w_neuron_1 = np.array([0,0.5,0.5,0])
    neuron_1 = MPNeuron(1, w_neuron_1)
    activated_neuron_1 = neuron_1.activated_output(input_)
    # AND 2 Neuron
    w_neuron_2 = np.array([0.5,0,0,0.5])
    neuron_2 = MPNeuron(1,w_neuron_2)
    activated_neuron_2 = neuron_2.activated_output(input_)
    # AND 3 Neuron
    w_neuron_3 = np.array([0.5,0.5])
    neuron_3 = MPNeuron(1,w_neuron_3)
    neuron_3_input = [activated_neuron_1,activated_neuron_2]
    activated_neuron_3 = neuron_3.activated_output(neuron_3_input)
    # AND 4 Neuron
    w_neuron_4 = np.array([0.5,0.5])
    neuron_4 = MPNeuron(1,w_neuron_4)
    neuron_4_input = np.array([input_[0],input_[2]])
    activated_neuron_4 = neuron_4.activated_output(neuron_4_input)
    # h3 Neuron
    h3_w = np.array([0.5,0.5])
    h3_neuron = MPNeuron(1,h3_w)
    h3_input = np.array([activated_neuron_3, activated_neuron_4])
    h3 = h3_neuron.activated_output(h3_input)
    first_num = tuple(input_[0:2])
    second_num = tuple(input_[2:4])
    first_num_dec = first_num[0]*2 + first_num[1]*1
    second_num_dec = second_num[0]*2 + second_num[1]*1
    answer_dec = h3*8 + h2*4 + h1*2 + h0
    print(
        " ===========\n",
        f' {first_num_dec} * {second_num_dec} = {answer_dec} \n',
        f'{first_num} * {second_num} = {h3,h2,h1,h0}\n',
        "===========")


#################
### Test Area ###
#################

lst = list(itertools.product([0, 1], repeat=2))
for num in lst:
    for num_2 in lst:
        input_test = [num[0],num[1],num_2[0],num_2[1]]
        calculate_mult_network(input_test)

