from mpi4py import MPI
import numpy as np
from xy import *
import matplotlib.pylab as plt

comm = MPI.COMM_WORLD 
size = comm.Get_size()
rank = comm.Get_rank()

print(f"size = {size}")
print(f"rank = {rank}")

# params to change
J = 1
max_T = 2
min_T = 0.01
values_per_proccess = 10
lattice_shape = (20, 20)
steps = 100
iters_per_step = 1000
random_state = 314

# cuts T_vals into correct sizes so that MPI works correctly
T_vals = np.linspace(min_T, max_T, size * values_per_proccess)[values_per_proccess * rank:values_per_proccess * (rank+1)]
betas = 1 / T_vals

# initiate arrays to store data
correlation_lengths = []
specific_heats = []
sims = 1

# runs the XY model over all temperatures
for beta in betas:
    xy = XYModelMetropolisSimulation(lattice_shape=lattice_shape, beta=beta, J=J, random_state=random_state)
    xy.simulate(steps, iters_per_step)
    correlation_lengths.append(xy.get_correlation_length())
    specific_heats.append(xy.get_specific_heat())

    # prints the progress of the simulation
    print('Rank %d finished sim %d of %d' %(rank, sims, values_per_proccess))
    sims += 1

# converts the lists to numpy arrays for MPI gathering
correlation_lengths = np.array(correlation_lengths)
specific_heats = np.array(specific_heats)

print('Rank %d finished all sims' % (rank))

# initiates dummy buffers for non root processes
all_correlation_lengths = None
all_specific_heats = None

# initiates the actual buffers for the root process
if rank == 0:
    all_correlation_lengths = np.empty([size, values_per_proccess], dtype=np.float64)
    all_specific_heats = np.empty([size, values_per_proccess], dtype=np.float64)

# gathers the data from all processes to the root process
comm.Gather(correlation_lengths, all_correlation_lengths, root=0)
comm.Gather(specific_heats, all_specific_heats, root=0)

if rank == 0:
    T_vals = np.linspace(min_T, max_T, size * values_per_proccess)
    data = np.concatenate((T_vals.reshape(size * values_per_proccess, 1), 
                           all_correlation_lengths.flatten().reshape(size * values_per_proccess, 1),
                           all_specific_heats.flatten().reshape(size * values_per_proccess, 1)),
                          axis=1)
    
    np.save('data.npy', data)

