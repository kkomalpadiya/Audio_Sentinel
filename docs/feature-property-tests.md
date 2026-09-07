# B2.2: Feature shape, range, and determinism tests

`tests/test_feature_properties.py` adds 24 tests to the existing Log-Mel generator
and storage suites. These check numerical properties beyond the original default
examples. No audio-processing implementation changes were required.

| Coverage | Independent expectation |
| --- | --- |
| Four recipes at 8, 16, 44.1, and 48 kHz | Enumerated complete FFT frame starts, expected Mel axis, float32 C order, finite values, and short-input padding |
| FFT boundary and seeded lengths | 27 length cases per recipe, including just below/at FFT and hop boundaries; 108 cases total |
| Dynamic range | Narrow-range output equals broad-range output floored at its peak minus the configured top_db |
| Fixed reference power | Changing the reference shifts every cell by minus 10 log10 of the reference ratio, without peak normalization |
| Silence floors | Configured amin/reference values give the known absolute dB floor, including very small positive amin |
| Increased minimum power | Only bins below the new absolute power floor rise |
| Positive output values | A prepared-scale tone can produce positive power dB, so values must not be forced into [-80, 0] |
| Polarity and array layout | Negating samples or using an equivalent strided read-only view preserves power features |
| Call independence | Mutating earlier result arrays, interleaving another recipe, and reloading recipe JSON cannot change the next result |
| Fresh-process reproducibility | Two clean Python interpreters process the same saved samples with different call orders and produce exactly the parent's NPY bytes |

Shape expectations enumerate frame starts instead of calling the implementation's
shape helper. The range checks use mathematical relationships rather than
duplicating the production FFT pipeline. Existing B2.1 tests still compare
numerical output against independently constructed FFT frames, Mel triangles,
and dB calculations. A2.2 tests still verify persistence and source linkage.

## Run

```powershell
python -m pytest tests/test_feature_properties.py -q
.\scripts\verify_project.ps1
```

Tests use fixed seeds and generated signals. The process test saves its samples
and recipe in the test's temporary directory, starts Python directly without a
shell, and removes PYTHONPATH so the checkout is selected explicitly. No audio
recordings, model downloads, or manual setup are needed.

Exact byte equality is verified in the current software/platform environment,
including fresh processes and different call orders. It is not a promise of
identical bits across different NumPy, librosa, FFT/BLAS libraries, or platforms.
The tests do not establish acoustic-model accuracy or real-world detection quality.

## In plain language

These tests check that feature tables have the right dimensions, respond to
volume and scaling settings correctly, and give the same numbers when the same
audio is processed again. Next is A2.3: connect preparation, feature generation,
and storage into one pipeline service.
