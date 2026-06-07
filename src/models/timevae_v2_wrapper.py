"""TimeVAE_v2: TSGM BetaVAE with the loss actually parameterised by beta."""

from __future__ import annotations

import os
from typing import Any

import keras
import numpy as np
from keras import layers, ops

from tsgm.models.architectures.zoo import Sampling
from tsgm.models.cvae import BetaVAE

from src.models._validation import (
    OutputActivation,
    ScalerFamily,
    profile_to_scaler_family,
    validate_scaler_activation,
)


def _build_encoder_vae_conv5(
    seq_len: int, feat_dim: int, latent_dim: int,
) -> keras.Model:
    """Mirror of TSGM ''VAE_CONV5Architecture._build_encoder''."""
    encoder_inputs = keras.Input(shape=(seq_len, feat_dim))
    x = encoder_inputs
    for kernel in (10, 2, 2, 2, 4):
        x = layers.Conv1D(64, kernel, activation="relu", padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


def _build_decoder_vae_conv5(
    seq_len: int,
    feat_dim: int,
    latent_dim: int,
    output_activation: OutputActivation,
) -> keras.Model:
    """Mirror of TSGM ''VAE_CONV5Architecture._build_decoder'' with configurable output activation."""
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation="relu")(latent_inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    dense_shape = 64 * seq_len
    x = layers.Dense(dense_shape, activation="relu")(x)
    x = layers.Reshape((seq_len, dense_shape // seq_len))(x)
    for kernel in (2, 2, 2, 2, 10):
        x = layers.Conv1D(64, kernel, activation="relu", padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
    out_act = None if output_activation == "linear" else output_activation
    decoder_outputs = layers.Conv1D(
        feat_dim, 3, activation=out_act, padding="same",
    )(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


class TimeVAEv2(BetaVAE):
    """Fixed-loss ''BetaVAE'' with reconstruction_wt and KL annealing."""

    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        beta: float = 1.0,
        reconstruction_wt: float = 3.0,
        kl_anneal_epochs: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder, decoder, beta=beta, **kwargs)
        self.reconstruction_wt = float(reconstruction_wt)
        self.kl_anneal_epochs = int(kl_anneal_epochs)
        self.current_epoch = 0

    def _kl_weight(self) -> float:
        if self.kl_anneal_epochs <= 0:
            return float(self.beta)
        ramp = min(1.0, (self.current_epoch + 1) / float(self.kl_anneal_epochs))
        return float(self.beta) * float(ramp)

    def train_step_tf(self, tf, data):  # type: ignore[override]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = self._get_reconstruction_loss(data, reconstruction)
            kl = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl = ops.mean(ops.sum(kl, axis=1))
            total = self.reconstruction_wt * recon_loss + self._kl_weight() * kl
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step_torch(self, torch, data):  # type: ignore[override]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        recon_loss = self._get_reconstruction_loss(data, reconstruction)
        kl = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl = ops.mean(ops.sum(kl, axis=1))
        total = self.reconstruction_wt * recon_loss + self._kl_weight() * kl
        if hasattr(total, "shape") and len(total.shape) > 0:
            total = ops.mean(total)

        self.zero_grad()
        total.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step_jax(self, jax, data):  # type: ignore[override]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        recon_loss = self._get_reconstruction_loss(data, reconstruction)
        kl = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl = ops.mean(ops.sum(kl, axis=1))
        total = self.reconstruction_wt * recon_loss + self._kl_weight() * kl
        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class _EpochCounter(keras.callbacks.Callback):
    """Sets ''model.current_epoch'' at the start of every epoch."""

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        model = self.model
        if hasattr(model, "current_epoch"):
            model.current_epoch = int(epoch)


def build_timevae_v2(
    seq_len: int,
    feat_dim: int,
    latent_dim: int,
    *,
    beta: float = 1.0,
    reconstruction_wt: float = 3.0,
    kl_anneal_epochs: int = 0,
    output_activation: OutputActivation = "linear",
    learning_rate: float = 1e-3,
    scaler_family: ScalerFamily | None = None,
) -> TimeVAEv2:
    """Build and compile a :class:`TimeVAEv2` with a ''vae_conv5''-style backbone."""
    if scaler_family is not None:
        validate_scaler_activation(scaler_family, output_activation)

    os.environ.setdefault("KERAS_BACKEND", "torch")
    encoder = _build_encoder_vae_conv5(seq_len, feat_dim, latent_dim)
    decoder = _build_decoder_vae_conv5(
        seq_len, feat_dim, latent_dim, output_activation,
    )
    model = TimeVAEv2(
        encoder,
        decoder,
        beta=beta,
        reconstruction_wt=reconstruction_wt,
        kl_anneal_epochs=kl_anneal_epochs,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate))
    return model


def fit_timevae_v2(
    model: TimeVAEv2,
    x: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    verbose: int = 0,
) -> None:
    """Fit with deterministic shuffle and a per-epoch KL-annealing counter."""
    keras.utils.set_random_seed(seed)
    model.fit(
        x,
        epochs=epochs,
        batch_size=min(batch_size, max(1, x.shape[0])),
        verbose=verbose,
        shuffle=True,
        callbacks=[_EpochCounter()],
    )


def generate_numpy(
    model: TimeVAEv2,
    n: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Generate ''n'' synthetic windows as a ''float32'' numpy array ''(n, L, F)''."""
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    latent_dim = int(model.latent_dim)
    if seed is not None:
        seed_gen = keras.random.SeedGenerator(seed=int(seed))
        z = keras.random.normal((n, latent_dim), seed=seed_gen)
    else:
        z = keras.random.normal((n, latent_dim))
    out = model.decoder(z, training=False)
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float32)


__all__ = [
    "OutputActivation",
    "TimeVAEv2",
    "_EpochCounter",
    "build_timevae_v2",
    "fit_timevae_v2",
    "generate_numpy",
    "profile_to_scaler_family",
]
