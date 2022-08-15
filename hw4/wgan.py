import torch
from torch import Tensor
from typing import Callable
from torch.optim.optimizer import Optimizer
from hw4.gan import Generator, Discriminator


def train_wgan_batch(dsc_model: Discriminator,
                     gen_model: Generator,
                     critic_loss_fn: Callable,
                     gen_loss_fn: Callable,
                     dsc_optimizer: Optimizer,
                     gen_optimizer: Optimizer,
                     critic_iterations,
                     weight_clip_limit,
                     x_data: Tensor):
    batch_size = x_data.shape[0]

    for i in range(critic_iterations):
        dsc_model.zero_grad()

        for p in dsc_model.parameters():
            p.data.clamp_(-weight_clip_limit, weight_clip_limit)

        generated = gen_model.sample(batch_size)

        dsc_y_real = dsc_model(x_data)
        dsc_y_fake = dsc_model(generated)

        critic_loss = critic_loss_fn(dsc_y_real, dsc_y_fake)

        critic_loss.backward(retain_graph=True)

        dsc_optimizer.step()

    generated = gen_model.sample(batch_size, True)

    dsc_y_fake = dsc_model(generated)

    loss_gen = gen_loss_fn(dsc_y_fake)

    gen_model.zero_grad()

    loss_gen.backward()

    gen_optimizer.step()

    return critic_loss.item(), loss_gen.item()


def critic_loss_function(y_data, y_generated):
    return (y_generated - y_data).mean()


def generator_loss_function(y_generated):
    return -torch.mean(y_generated)
