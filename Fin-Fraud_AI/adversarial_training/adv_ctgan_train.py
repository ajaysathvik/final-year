import os

import numpy as np
import pandas as pd
import torch
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import CTGAN, Discriminator, Generator
from torch import nn
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_IN = os.path.join(ROOT, "original_dataset", "final1.csv")
DATA_OUT = os.path.join(ROOT, "adversarial_training", "adv_balanced_data.csv")
NOVEL_SCAM_SEEDS = os.path.join(ROOT, "original_dataset", "new_scam_seeds.csv")
SYNTHETIC_NOVEL_OUT = os.path.join(ROOT, "adversarial_training", "synthetic_new_scams_adv_ctgan.csv")
SYNTHETIC_BATCH_OUT = os.environ.get(
    "ADV_CTGAN_SYNTHETIC_OUTPUT_PATH",
    os.path.join(ROOT, "adversarial_training", "synthetic_adv_batch.csv"),
)

TEXT_COLS = ["title", "body"]
MIN_TARGET_ROWS = 20


def _prepare_training_slice(df):
    working = df.drop(columns=[col for col in TEXT_COLS if col in df.columns], errors="ignore").copy()
    working["amount_numeric"] = pd.to_numeric(working.get("amount_numeric"), errors="coerce")
    if "amount_numeric" in working.columns:
        if working["amount_numeric"].notna().any():
            working["amount_numeric"] = working["amount_numeric"].fillna(working["amount_numeric"].median())
        else:
            working["amount_numeric"] = 0.0
    return working


def _related_fraud_rows(full_df, seeds_df):
    fraud_df = full_df[full_df["is_fraud"] == 1].copy()
    if seeds_df.empty:
        return fraud_df.iloc[0:0].copy()

    types = set(seeds_df.get("annotation.fraud_type", pd.Series(dtype=str)).dropna().astype(str))
    channels = set(seeds_df.get("annotation.key_features.fraud_channel", pd.Series(dtype=str)).dropna().astype(str))
    payments = set(seeds_df.get("annotation.key_features.payment_method", pd.Series(dtype=str)).dropna().astype(str))

    mask = pd.Series(False, index=fraud_df.index)
    if types:
        mask |= fraud_df.get("annotation.fraud_type", pd.Series(index=fraud_df.index, dtype=object)).astype(str).isin(types)
    if channels:
        mask |= fraud_df.get("annotation.key_features.fraud_channel", pd.Series(index=fraud_df.index, dtype=object)).astype(str).isin(channels)
    if payments:
        mask |= fraud_df.get("annotation.key_features.payment_method", pd.Series(index=fraud_df.index, dtype=object)).astype(str).isin(payments)
    return fraud_df[mask].copy()


def _choose_generation_target(full_df):
    forced_mode = os.environ.get("ADV_CTGAN_FORCE_MODE", "").strip().lower()
    if forced_mode == "legacy_non_fraud":
        return full_df[full_df["is_fraud"] == 0].copy(), "legacy_non_fraud", 0, 500
    if forced_mode == "novel_scam_fraud":
        seeds_df = pd.read_csv(NOVEL_SCAM_SEEDS) if os.path.exists(NOVEL_SCAM_SEEDS) else pd.DataFrame()
        seeds_df = seeds_df[seeds_df.get("is_fraud", 0) == 1].copy()
        target_df = pd.concat([seeds_df, _related_fraud_rows(full_df, seeds_df)], ignore_index=True).drop_duplicates()
        return target_df, "novel_scam_fraud", 1, 500

    if os.path.exists(NOVEL_SCAM_SEEDS):
        seeds_df = pd.read_csv(NOVEL_SCAM_SEEDS)
        seeds_df = seeds_df[seeds_df.get("is_fraud", 0) == 1].copy()
    else:
        seeds_df = pd.DataFrame()

    if not seeds_df.empty:
        target_df = pd.concat([seeds_df, _related_fraud_rows(full_df, seeds_df)], ignore_index=True).drop_duplicates()
        mode = "novel_scam_fraud"
        synthetic_label = 1
        sample_size = min(500, max(100, len(seeds_df) * 4))
        if len(target_df) >= MIN_TARGET_ROWS:
            return target_df, mode, synthetic_label, sample_size
        print(
            f"Novel scam seed set too small for stable Adv-CTGAN ({len(target_df)} rows). "
            "Falling back to legacy non-fraud augmentation."
        )

    fallback_df = full_df[full_df["is_fraud"] == 0].copy()
    return fallback_df, "legacy_non_fraud", 0, 500


class AdversarialCTGAN(CTGAN):
    def __init__(self, adv_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_weight = adv_weight
        self.adversary = None
        self.adv_optimizer = None

    def _build_adversary(self, data_dim):
        self.adversary = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(self._device)
        self.adv_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=2e-4, weight_decay=1e-5)
        self.adv_criterion = nn.BCELoss()

    def fit(self, train_data, discrete_columns=(), epochs=None):
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )
        data_dim = self._transformer.output_dimensions
        self._build_adversary(data_dim)

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)
        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        optimizer_g = torch.optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizer_d = torch.optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        print("Starting Adversarial CTGAN Training...")
        for epoch in range(epochs):
            for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                for _ in range(self._discriminator_steps):
                    fakez = torch.normal(
                        mean=0.0,
                        std=1.0,
                        size=(self._batch_size, self._embedding_dim),
                        device=self._device,
                    )

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)
                        perm = np.random.permutation(self._batch_size)
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[torch.as_tensor(perm, device=self._device)]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        fake_cat = fakeact
                        real_cat = real

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)
                    penalty = self._discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizer_d.zero_grad()
                    penalty.backward(retain_graph=True)
                    loss_d.backward()
                    optimizer_d.step()

                    self.adv_optimizer.zero_grad()
                    pred_real = self.adversary(real)
                    pred_fake = self.adversary(fakeact.detach())
                    loss_adv_real = self.adv_criterion(pred_real, torch.ones_like(pred_real))
                    loss_adv_fake = self.adv_criterion(pred_fake, torch.zeros_like(pred_fake))
                    loss_adv = (loss_adv_real + loss_adv_fake) / 2
                    loss_adv.backward()
                    self.adv_optimizer.step()

                fakez = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=(self._batch_size, self._embedding_dim),
                    device=self._device,
                )
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1 = None, None
                else:
                    c1, m1, _, _ = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                    cross_entropy = self._cond_loss(fake, c1, m1)
                else:
                    y_fake = self._discriminator(fakeact)
                    cross_entropy = 0

                loss_g_core = -torch.mean(y_fake) + cross_entropy
                pred_fake_adv = self.adversary(fakeact)
                loss_g_adv = self.adv_criterion(pred_fake_adv, torch.ones_like(pred_fake_adv))
                loss_g = loss_g_core + self.adv_weight * loss_g_adv

                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1:03d} | Loss D: {loss_d.item():.4f} | "
                    f"Loss G (Base): {loss_g_core.item():.4f} | "
                    f"Loss G (Adv): {loss_g_adv.item():.4f} | Adv Agent Loss: {loss_adv.item():.4f}"
                )


def run_adv_ctgan():
    print("Loading Original Dataset...")
    df = pd.read_csv(DATA_IN)
    df.replace("unknown", pd.NA, inplace=True)

    target_df, mode, synthetic_label, sample_size = _choose_generation_target(df)
    if target_df.empty:
        raise RuntimeError("No rows available for Adv-CTGAN training.")

    train_df = _prepare_training_slice(target_df)

    discrete_columns = []
    for col in train_df.columns:
        if col == "amount_numeric":
            continue
        if train_df[col].dtype == "object" or train_df[col].dtype == "bool" or train_df[col].nunique(dropna=False) < 10:
            discrete_columns.append(col)

    print(f"Training Adversarial CTGAN on {len(train_df)} rows ({mode})...")
    model = AdversarialCTGAN(
        epochs=300,
        batch_size=20,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        adv_weight=2.0,
    )
    model.fit(train_df, discrete_columns=discrete_columns)

    requested_size = int(os.environ.get("ADV_CTGAN_SAMPLE_SIZE", sample_size) or sample_size)
    sample_size = max(1, requested_size)
    print(f"Generating {sample_size} synthetic adversarial rows for mode={mode}...")
    synthetic_adv = model.sample(sample_size)
    synthetic_adv["is_fraud"] = synthetic_label
    for col in TEXT_COLS:
        synthetic_adv[col] = pd.NA
    synthetic_adv.to_csv(SYNTHETIC_BATCH_OUT, index=False)
    print(f"Saved generated batch to {SYNTHETIC_BATCH_OUT}")

    if mode == "novel_scam_fraud":
        synthetic_adv.to_csv(SYNTHETIC_NOVEL_OUT, index=False)
        print(f"Saved synthetic novel scam rows to {SYNTHETIC_NOVEL_OUT}")
    elif os.path.exists(SYNTHETIC_NOVEL_OUT):
        os.remove(SYNTHETIC_NOVEL_OUT)

    adv_balanced_df = pd.concat([df, synthetic_adv], ignore_index=True)
    adv_balanced_df.to_csv(DATA_OUT, index=False)
    print(f"Saved to {DATA_OUT}!")


if __name__ == "__main__":
    run_adv_ctgan()
