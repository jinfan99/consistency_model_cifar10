from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import math
import os.path as osp
import torch

from jcm import checkpoints
from jcm import losses
from jcm import sde_lib
from tqdm import tqdm

# Keep the import below for registering all model definitions
from jcm.models import ddpm, ncsnv2, ncsnpp

from jcm.models import utils as mutils

import matplotlib.pyplot as plt
from jcm import datasets

def load_seed_dir(seed_dir):
    seed_list = []
    for i in range(10):
        seed_path = osp.join(seed_dir, 'seed'+str(i)+'.pth')
        seed = torch.load(seed_path)
        seed_list.append(seed)
    seed_tensor = torch.cat(seed_list, dim=0)
    seed_np = seed_tensor.permute(0, 2, 3, 1).numpy()[None, :]
    seed_jax = jax.numpy.asarray(seed_np)
    return seed_jax

def load_seed_file(seed_path):
    seed = torch.load(seed_path)
    seed_np = seed.permute(0, 2, 3, 1).numpy()[None, :]

    return seed_jax

def plot_grid(samples):
    samples = samples / 2. + 0.5
    samples = samples.transpose((0, 2, 1, 3, 4))
    samples = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2] * samples.shape[3], samples.shape[4]))
    plt.figure(figsize=(10,10))
    plt.imshow(samples)
    plt.axis('off')

def get_onestep_sampler(sde, model, init_std=None):
    def sampler(state, z):        
        x = z
        model_fn = mutils.get_distiller_fn(
            sde,
            model,
            state.params_ema,
            state.model_state,
            train=False,
            return_state=False,
        )
        samples = model_fn(x, jnp.ones((x.shape[0],)) * init_std)
        return samples, 1

    return jax.pmap(sampler, axis_name="batch")


def cd_lpips_generation(init_z):
    ## Load CD models
    from configs.cifar10_ve_cd import get_config
    config = get_config()
    rng = hk.PRNGSequence(42)
    model, init_model_state, initial_params = mutils.init_model(next(rng), config)
    optimizer, optimize_fn = losses.get_optimizer(config)
    state = mutils.State(
        step=0,
        lr=4e-4,
        ema_rate=config.model.ema_rate,
        params=initial_params,
        params_ema=initial_params,
        model_state=init_model_state,
        opt_state=optimizer.init(initial_params),
        rng_state=rng.internal_state,
    )

    ## You need to manually download the checkpoint first.
    checkpoint_dir = "/home/ubuntu/consistency_models_cifar10/checkpoints/cd_lpips/"
    state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=80)

    sde = sde_lib.get_sde(config)

    pstate = flax.jax_utils.replicate(state)
    onestep_sampler = get_onestep_sampler(sde, model, init_std=80.)

    output_images = jax.numpy.zeros_like(init_z.squeeze())
    print('output shape: ', output_images.shape)
    bs = 100
    assert init_z.shape[1] % bs == 0, "can't be divided!"

    num_iter = int(init_z.shape[1] / bs)

    for i in tqdm(range(num_iter)):
        onestep_samples = onestep_sampler(pstate, init_z[:, bs*i: bs*(i+1)] * 80.)[0].clip(-1,1)
        
        # output_images[bs*i: bs*(i+1)] = onestep_samples
        output_images = output_images.at[bs*i: bs*(i+1)].set(((onestep_samples.squeeze()+1)/2)*255)
        

    output_dir = '/home/ubuntu/tmp_images/sscd'

    output_path = osp.join(output_dir, 'cd_lpips.pth')
    torch.save(output_images, output_path)


def ct_lpips_generation(init_z):
    # Use the following for loading CT models
    from configs.cifar10_ve_ct_adaptive import get_config
    config = get_config()
    rng = hk.PRNGSequence(42)
    model, init_model_state, initial_params = mutils.init_model(next(rng), config)
    optimizer, optimize_fn = losses.get_optimizer(config)
    state = mutils.StateWithTarget(
        step=0,
        lr=2e-4,
        ema_rate=config.model.ema_rate,
        params=initial_params,
        target_params=initial_params,
        params_ema=initial_params,
        model_state=init_model_state,
        opt_state=optimizer.init(initial_params),
        rng_state=rng.internal_state,
    )

    checkpoint_dir = "/home/ubuntu/consistency_models_cifar10/checkpoints/ct_lpips"
    state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=74)

    sde = sde_lib.get_sde(config)

    pstate = flax.jax_utils.replicate(state)
    onestep_sampler = get_onestep_sampler(sde, model, init_std=80.)

    output_images = jax.numpy.zeros_like(init_z.squeeze())
    print('output shape: ', output_images.shape)
    bs = 100
    assert init_z.shape[1] % bs == 0, "can't be divided!"

    num_iter = int(init_z.shape[1] / bs)

    for i in tqdm(range(num_iter)):
        onestep_samples = onestep_sampler(pstate, init_z[:, bs*i: bs*(i+1)] * 80.)[0].clip(-1,1)
        
        # output_images[bs*i: bs*(i+1)] = onestep_samples
        output_images = output_images.at[bs*i: bs*(i+1)].set(((onestep_samples.squeeze()+1)/2)*255)

        # print(output_images[0][0])
    output_dir = '/home/ubuntu/tmp_images/sscd'

    output_path = osp.join(output_dir, 'ct_lpips.pth')
    torch.save(output_images, output_path)



def main():
    seed_dir = '/home/ubuntu/download/seed'
    seed = load_seed_dir(seed_dir)
    
    # seed = seed[None, :]

    cd_lpips_generation(seed)
    ct_lpips_generation(seed)


if __name__ == '__main__':
    main()