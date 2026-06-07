import torch
import pickle

# Set the parameters
num_rows = 5
num_cols = 50
num_repetitions = int(num_cols/num_rows)

# Create the repetition code matrix
G = torch.zeros((num_rows, num_cols))
for i in range(num_rows):
    # Set 10 ones in each row at consecutive positions
    G[i, i * num_repetitions:(i + 1) * num_repetitions] = 1

rep_matrices = {'G':G, 'D':G.T}

# Save!
path = './repetition_codes/rep_matrices_'+str(num_rows)+'_'+str(num_cols)+'.pkl'   

with open(path, 'wb') as file:
    pickle.dump(rep_matrices, file)