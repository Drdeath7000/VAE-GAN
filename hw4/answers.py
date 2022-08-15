r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""



# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0.0, learn_rate=0.00, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    #Best
    # hypers["batch_size"] = 5
    # hypers["h_dim"] = 1024
    # hypers["z_dim"] = 100
    # hypers["x_sigma2"] = 0.4
    # hypers["learn_rate"] = 0.001
    # hypers["betas"] = (0.5, 0.99)

    hypers["batch_size"] = 25
    hypers["h_dim"] = 1024
    hypers["z_dim"] = 100
    hypers["x_sigma2"] = 0.4
    hypers["learn_rate"] = 0.001
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**The parameter $$\sigma^{2}$$ effects the importance of the differnce between the original image and the
reconstructed one. For low values the expression $$\frac{1}{\sigma^{2} d_{x}}$$ is large and the
reconstruction loss becomes more important than the kl-divergence loss. For large values the expression
shrinks and the loss is more effectted by the kl-divergence loss. **

"""

part2_q2 = r"""
**Your answer:
1)
Reconstruction loss: The porpouse of the reconstruction loss is the make the model learn how to generate 
sample simillar to the training samples. The model tries to minimize the distance between the generated
sample and the instance it trained on.
KL-loss: The porpouse of the KL-loss is to normalize the training, avoid over-fitting and insure good properties of the
latent space (close samples will have simmilar generated results).

2)
The KL-loss term is forces the learnt distribution to be close to the standart normal distribution.
This happens because KL-divergence measures the distance between two distributions and will increase
the loss if the mean and variance of the model are far from the identity covariance and 0 mean.

3) The benefit of the effect is that because the learnt distributions covariances are close to the identity
and the mean is close to zero. leart distributions are less more "spread out" instead of a small part of the 
latent space to produce meaningfull results and the area around it to produce noise. Additionaly differnect
distribution will be closer to each other because of the 0 mean.

"""

part2_q3 = r"""
**We start by maximizing the the evidence distribution, $$p(X)$$ because it indirectly leads to the
minimization of the KL-Divergence :**
"""

part2_q4 = r"""
**We model the log of the latent space variance for ease of training. If we wouldn't have modeld as
log, we would have to calculate the log of the variance. That would result in very small numbers and
more un-necessary computation:**
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
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
            lr=0.0002,
            # You an add extra args for the optimizer here,
            betas=(0.5, 0.999),
            weight_decay=0.001
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 10
    hypers["z_dim"] = 100
    hypers["data_label"] = 0
    hypers['label_noise'] = 0.1

    # ========================
    return hypers


part3_q1 = r"""
**During training, we use the generator to create samples, feed them to the discriminator
and then do back-propagation on the loss of the discriminator.
We maintatin the gradients of the samples when training the generator because we want to backward pass
to update the generator.
We do not maintain the graditents of the sample when training the descriminator because the parameters
of the generator are not updated.
 :**

"""

part3_q2 = r"""
**
1) We should not decide to stop if the generator loss function is below a threshold because the value of 
the loss is decided by the quality of the disciminator and it could be the case that it is not
good enough at distinguisihing bewteen fake and real samples and that the reasons the generator
loss function is below the treshold.

2) If the discriminator loss is a constant value and the generator loss decreases it means that
the discriminator is making random guesses (giving probability 0.5 to all samples) and that the generator
is creating sample good enough to fool it. 

:**

"""

part3_q3 = r"""
**VAE outputs are more blurry this can be attributed to the fact that the loss function optimized
a lower bound on the likelihood of a sample being generated and not the likelihood itself.:**

"""

# ==============


def part4_wgan_hyperparams():
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
    hypers["z_dim"] = 50
    hypers['critic_iterations'] = 3

    # ========================
    return hypers