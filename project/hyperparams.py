def sn_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            # You an add extra args for the optimizer here
            betas=(0.5, 0.999),
            weight_decay=0.001
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0004,
            # You an add extra args for the optimizer here,
            betas=(0.5, 0.999),
            weight_decay=0.001
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 10
    hypers["z_dim"] = 200
    hypers["data_label"] = 0
    hypers['label_noise'] = 0.1

    # ========================
    return hypers


def wgan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        discriminator_optimizer=dict(
            type="RMSprop",  # Any name in nn.optim like SGD, Adam
            lr=5e-5,
        ),
        generator_optimizer=dict(
            type="RMSprop",
            lr=5e-3,
        ),
        critic_iterations=5,
        clip_value=0.01
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 10
    hypers["z_dim"] = 200
    hypers['critic_iterations'] = 3

    # ========================
    return hypers
