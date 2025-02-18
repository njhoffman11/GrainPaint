import torch 
import numpy as np 
import tile_gen6 as tile_gen
from tile_gen6 import Tile
from matplotlib import pyplot as plt   
import os

import importlib
import ddm_utils
# importlib.reload(ddm_utils)
from ddm_utils import DiffusionSampler, beta_schedule

import torch.distributed as dist
import os 
from time import time

## Schedule Parameters
T = 250 # Number of timesteps
start = 1e-4 # Starting variance 
end = 0.02 # Ending variance
# Choose a schedule (if the following are False, then a linear schedule is used)
cosine = False # Use cosine schedule
exp_biasing = False # Use exponential schedule
exp_biasing_factor = 1 # Exponential schedule factor (used if exp_biasing=True)
##

# Choose a variance schedule

betas = beta_schedule(T=T, start=start, end=end,
                       scale= 1.0, cosine=cosine, 
                       exp_biasing=exp_biasing, exp_bias_factor=exp_biasing_factor
                       )

ddm_sampler = DiffusionSampler(T, betas)

def split_data_for_devices(data, num_devices):
    batch_size = len(data) // num_devices
    batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(num_devices - 1)]
    
    # Add the remaining data to the last batch
    batches.append(data[(num_devices - 1) * batch_size:])

    return batches



def run(rank, size, model, sample_corrupt, mask, iteration):
    # Set up device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # device = 'cpu' # for cpu benchmark
    
    # Partition data
    local_sample = sample_corrupt[rank]
    local_mask = mask[rank]

    local_sample = local_sample.to(device)
    local_mask = local_mask.to(device)
    # Evaluate
    # if iteration == 0:
    #     output = ddm_sampler.generate_sample(model, local_sample, T=100, mask=local_mask, seed=None)
    # else:
    output = ddm_sampler.generate_sample_resample(model, local_sample, T=250, mask=local_mask, seed=None)
    # Gather results (optional)

    if rank == 0:
        # Determine the maximum size based on the largest tensor in sample_corrupt
        max_size = max([batch.shape[0] for batch in sample_corrupt])

        # Create a list of zeros tensors that all have the same shape (max_size)
        gathered_results = [torch.zeros(max_size, *sample_corrupt[0].shape[1:]).to(output.device) for _ in sample_corrupt]
        padded_output = torch.zeros(max_size, *output.shape[1:]).to(output.device)
        padded_output[:output.shape[0]] = output
        dist.gather(padded_output, gather_list=gathered_results, dst=0)

        # Since the tensors might have padded zeros, remove the padding after gathering
        gathered_results = [tensor[:len(batch)] for tensor, batch in zip(gathered_results, sample_corrupt)]
        return torch.cat(gathered_results, dim=0)
    else:
        # Ensure the output tensor matches the shape of the tensors in gathered_results (on rank 0)
        max_size = max([batch.shape[0] for batch in sample_corrupt])
        padded_output = torch.zeros(max_size, *output.shape[1:]).to(output.device)
        padded_output[:output.shape[0]] = output

        dist.gather(padded_output, dst=0)


def init_process(rank, world_size):
    # If you are running on multipule GPUs and the script does not exit properly,
    # you may need to set a different MASTER_PORT before running again.

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, joined_extended, batches, run_num):
    window_size = 32
    # Initialize distributed environment
    init_process(rank, world_size)
    rank = dist.get_rank()
    size = dist.get_world_size()
    torch.cuda.set_device(rank)
    # Load model
    # model = torch.load(ddm_load_path, map_location='cpu') # for cpu benchmark
    model = torch.load(ddm_load_path, map_location='cuda:'+str(rank), weights_only=False)
    
    num_iterations = len(batches)
    
    # Initial random data
    mask = torch.zeros((1, 1, window_size, window_size, window_size))
    sample_corrupt = torch.randn((1, 1, window_size, window_size, window_size))
    broadcast_list = [sample_corrupt, mask]
    if rank == 0:
        joined_mask = np.zeros(joined_extended.shape)
        sample_corrupt, mask, joined_mask = create_new_input(sample_corrupt, joined_extended, None, batches[0], 0, joined_mask, run_num)
        broadcast_list = [sample_corrupt, mask]
    dist.broadcast_object_list(broadcast_list, 0)
    sample_corrupt, mask = broadcast_list
    # print("rank", rank, "received", sample_corrupt.shape, mask.shape)
    if rank == 0:
        start_time = time()

    for i in range(num_iterations):
        print("Rank", rank, "Batch", i+1)
        # Check the batch size

        # Distribute data to all processes
        
        sample_corrupt_batches = split_data_for_devices(sample_corrupt, size)
        mask_batches = split_data_for_devices(mask, size)
        if rank != 0:
            run(rank, size, model, sample_corrupt_batches, mask_batches, i)
        else:
            gathered_results = run(rank, size, model, sample_corrupt_batches, mask_batches, i)
            # Gather the results at rank 0
            # gathered_results = [torch.zeros_like(results) for _ in range(size)]
            # print("gathered")
            # dist.gather(results, gather_list=gathered_results, dst=0)
            # print("dist_gathered")
            # Process results and create new data for next iteration
            if i == num_iterations - 1:
                sample_corrupt, mask, joined_mask = create_new_input(gathered_results, joined_extended, batches[i], None, i, joined_mask, run_num)
                broadcast_list = [sample_corrupt, mask]
            else:
                sample_corrupt, mask, joined_mask = create_new_input(gathered_results, joined_extended, batches[i], batches[i+1], i, joined_mask, run_num)
                broadcast_list = [sample_corrupt, mask]
            print(f"Iteration {i + 1} results shape: {gathered_results.shape}")
        dist.broadcast_object_list(broadcast_list, 0)
        sample_corrupt, mask = broadcast_list
        # print("rank", rank, "received", sample_corrupt.shape, mask.shape)

    if rank == 0:
        elapsed_time = time() - start_time
        
        print(f"Total time: , {elapsed_time} seconds")
    
    # Cleanup
    dist.destroy_process_group()

def create_new_input(processed_results, joined_extended, curr_batch, nxt_batch, batch_num = 0, joined_mask = None, run_num = 0):
    window_size = 32
    
    if curr_batch is not None:
        coords = tile_gen.batch_to_coords(curr_batch, window_size)
        # if batch_num == 0:
        #     old_data = np.load('run_103_batch_0.npy')
        #     processed_results = []
        #     for j in range(len(coords)):
        #         processed_results.append(old_data[coords[j][0]:coords[j][0]+window_size, 
        #                                             coords[j][1]:coords[j][1]+window_size, 
        #                                             coords[j][2]:coords[j][2]+window_size])
        #     processed_results = torch.tensor(processed_results).unsqueeze(1)
        processed_results = processed_results.cpu().numpy()

        # Iterate over each 3D tensor slice

        for j in range(len(coords)):
            # Get the current 3D tensor slice
            img = processed_results[j,0,:,:,:]

            # Calculate the midpoint threshold
            threshold = (img.min() + img.max()) / 2.0

            # Check if image is unconditioned
         
            if joined_mask[coords[j][0]:coords[j][0]+window_size, 
                                       coords[j][1]:coords[j][1]+window_size, 
                                       coords[j][2]:coords[j][2]+window_size].sum() == 0:
                img -= threshold
            # img[img <= threshold] = 0
            # img[img > threshold] = 1
            
            # Update the processed_results tensor
            processed_results[j,0,:,:,:] = img
        processed_results = torch.tensor(processed_results)
    if curr_batch == 0:
        joined_mask = np.zeros(joined_extended.shape)
    # Generate new data based on processed results
    
    if curr_batch is not None:
        coords = tile_gen.batch_to_coords(curr_batch, window_size)
        for j in range(len(coords)):
            reverse_mask = joined_mask[coords[j][0]:coords[j][0]+window_size, 
                                       coords[j][1]:coords[j][1]+window_size, 
                                       coords[j][2]:coords[j][2]+window_size] == 0
            # print(coords[j][0],coords[j][0]+window_size, coords[j][1],coords[j][1]+window_size, coords[j][2],coords[j][2]+window_size)
            joined_extended[coords[j][0]:coords[j][0]+window_size, 
                            coords[j][1]:coords[j][1]+window_size, 
                            coords[j][2]:coords[j][2]+window_size]*=np.invert(reverse_mask)
            
            joined_extended[coords[j][0]:coords[j][0]+window_size, 
                            coords[j][1]:coords[j][1]+window_size, 
                            coords[j][2]:coords[j][2]+window_size] += processed_results[j,0,:,:,:].cpu().numpy()*reverse_mask
            
            joined_mask[coords[j][0]:coords[j][0]+window_size, 
                        coords[j][1]:coords[j][1]+window_size, 
                        coords[j][2]:coords[j][2]+window_size] += processed_results[j,0,:,:,:].cpu().numpy()*reverse_mask
        
        plt.imshow(joined_extended[:,:,round(joined_extended.shape[2]/2)])
        plt.savefig('run_{}/run_{}_batch_{}.png'.format(run_num, run_num, batch_num+1))
        np.save('run_{}/run_{}_batch_{}.npy'.format(run_num, run_num, batch_num+1), joined_extended)
    # plt.close()

    if nxt_batch is None:
        return None, None, None
    else:
        coords = tile_gen.batch_to_coords(nxt_batch, window_size)
        mask_list = []
        sample_corrupt_list = []
        for j in range(len(coords)):
            mask_list.append(torch.tensor((joined_extended[coords[j][0]:coords[j][0]+window_size, 
                                                           coords[j][1]:coords[j][1]+window_size, 
                                                           coords[j][2]:coords[j][2]+window_size] != 0).astype(int)))
            sample_corrupt_list.append(torch.randn((1,1,window_size,window_size,window_size)))
            sample_corrupt_list[-1][0,0,:,:,:] = torch.tensor(joined_extended[coords[j][0]:coords[j][0]+window_size, 
                                                                              coords[j][1]:coords[j][1]+window_size, 
                                                                              coords[j][2]:coords[j][2]+window_size]*mask_list[-1].cpu().numpy())
        mask = torch.stack(mask_list, dim=0).unsqueeze(1)
        sample_corrupt = torch.cat(sample_corrupt_list, dim=0)
        # batch_size = 15
    
    # mask = torch.zeros((batch_size, 1, 16, 16, 16))
    # sample_corrupt = torch.randn((batch_size, 1, 16, 16, 16))
    batch_size = sample_corrupt.size(0)
    world_size = dist.get_world_size()
    # If batch size is smaller than the number of GPUs
    if batch_size < world_size:
        # Calculate how many times to duplicate the last sample
        diff = world_size - batch_size

        # Duplicate the last sample of sample_corrupt 'diff' times
        last_sample_corrupt = sample_corrupt[-1].unsqueeze(0).repeat((diff, 1, 1, 1, 1))
        sample_corrupt = torch.cat([sample_corrupt, last_sample_corrupt], dim=0)

        # Duplicate the last sample of mask 'diff' times
        last_mask = mask[-1].unsqueeze(0).repeat((diff, 1, 1, 1, 1))
        mask = torch.cat([mask, last_mask], dim=0)
    
    return sample_corrupt, mask, joined_mask

ddm_load_path = "model_checkpoints/ddm32_big_250.ckpt"
# for the anisotropic model, use
# ddm_load_path = "model_checkpoints/ddm32_big_250_aniso.ckpt"
if __name__ == '__main__':

    # choose numbers to label the runs of the model with. This will create a new directory
    # for each run. The script will error if the directory already exists!
    for run_num in range(238,239):
    
        os.mkdir('run_{}'.format(run_num))
        window_size = 32 # block size, n.b. this only controls the plan

        # If the plan takes a long time to generate, which may happen if the plan is large,
        # you can save the plan to a file and load it later.
        # with open('turbo-blade_batches.pickle', 'rb') as handle:
        #     batches = pickle.load(handle)

        # size of generated area. Note that due to the size of the blocks,
        # the actual area will be larger. You can check the actual area by
        # running the plotting functions in tile_gen6.py
        lims = (80,80,80) 

        generation_plan = 'grid' # 'grid' or 'center', grid recommended for isotrpic,
        # center recommended for anisotropic
        if generation_plan == 'center':
            batches = tile_gen.gen_center(lims, window_size, 24)
        elif generation_plan == 'grid':
            batches = tile_gen.gen_grid(lims, window_size, 24)
        # print(batches)
        all_batches = []
        for batch in batches:
            for coord in tile_gen.batch_to_coords(batch, window_size):
                all_batches.append(coord)
        max_coord = np.array(all_batches).max(axis=0)+32
        joined_extended = np.zeros(max_coord)

        # world_size controls the number of GPUs used. You can use torch.cuda.device_count()
        # to use all available GPUs. or you can use a specific number of GPUs.
        world_size = torch.cuda.device_count()
        # world_size = 1
        print("Generation plan has", len(batches), "iterations")
        torch.multiprocessing.spawn(main, args=(world_size, joined_extended, batches, run_num), nprocs=world_size, join=True)


