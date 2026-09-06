"""Check B1.2 using a seeded synthetic tone plus background noise."""

import numpy as np

from audio_sentinel.config import NoiseReductionSettings
from audio_sentinel.noise_reduction import reduce_noise


def main() -> None:
    rate = 16_000
    clean = np.zeros(rate * 2)
    clean[8_000:24_000] = 0.3 * np.sin(2 * np.pi * 1_000 * np.arange(rate) / rate)
    noise = np.random.default_rng(22).normal(0, 0.03, clean.size)
    samples = (clean + noise).astype(np.float32)[:, None]
    disabled = reduce_noise(samples, rate, NoiseReductionSettings(), max_decoded_bytes=1_000_000)
    np.testing.assert_array_equal(disabled.samples, samples)
    result = reduce_noise(samples, rate, NoiseReductionSettings(enabled=True, reduction_strength=0.8),
                          max_decoded_bytes=1_000_000)
    assert result.samples.shape == samples.shape
    original_snr = 10 * np.log10(np.sum(clean ** 2) / np.sum((samples[:, 0] - clean) ** 2))
    output_snr = 10 * np.log10(np.sum(clean ** 2) / np.sum((result.samples[:, 0] - clean) ** 2))
    assert output_snr - original_snr > 2
    print(f"Synthetic benchmark at strength 0.8: SNR {original_snr:.2f} dB before, {output_snr:.2f} dB after.")
    print(f"Preserved {len(samples)} frames at {rate} Hz. Disabled mode is an exact copy.")
    print("No recordings, downloads, or output files used. This is not a real-dataset evaluation.")


if __name__ == "__main__":
    main()
