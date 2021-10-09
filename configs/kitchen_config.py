import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.temperature = 0.5
    config.dropout_rate = 0.1

    config.tau = 0.005  # For soft target updates.

    return config
