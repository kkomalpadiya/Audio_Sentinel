# A1.5: Preparation integration tests and sample-clip smoke test

The offline preparation path now has 13 additional integration tests and a
standalone smoke-test script. These run the actual loader, transforms, segmentation,
and persistence through `AudioPreparationService`, then reopen the saved files.

## Integration coverage

| Scenario | What is checked |
| --- | --- |
| Eight format/recipe combinations | WAV and FLAC; 8, 16, 44.1, and 48 kHz inputs; mono and stereo; up/downsampling; noise reduction on/off; pad/drop tails |
| Configuration and metadata | Load the recipe from JSON; verify the stored settings, source hash, permission, dataset, annotations, and duration |
| Encoded output | Expected shape, approximate target loudness, peak limit, tone frequency, PCM16 windows, and exact window-to-full-clip sample equality |
| Fresh service and repeated input | Reload configuration and create a new service; matching output is reused with unchanged bytes and timestamps |
| Silence | Enabled noise reduction and normalization leave silence silent, including padded windows |
| Late write failure and retry | Inject a disk error after a window write; discard partial staging, preserve an earlier completed bundle and the raw source, then retry successfully |
| Conflicting annotations | Reject changed metadata at the same output identity; the original request still reuses its original result |
| Permission scope | Acoustic-only permission rejects language-level annotations but still allows acoustic preparation |
| Standalone command | Launch the smoke script in another working directory with no `PYTHONPATH`; parse its success report |

The matrix includes eight parametrized tests; the remaining scenarios add five
tests. Samples and output comparisons are generated locally. Expected dimensions,
tone frequency, loudness, and sample-slice relationships are checked independently
of calling the preparation functions again to produce an expected result.

The disk failure uses a temporary patched write method after an actual write.
Other processing remains real. Tests restore that method before the retry.
These checks cover integration behavior; they do not establish model accuracy,
real-world denoising quality, or suitability for live monitoring.

## Run the sample-clip smoke test

From the project root in PowerShell:

```powershell
python scripts/smoke_test_preparation.py
```

The script finds this checkout's `src` directory itself. It needs the project's
existing Python dependencies, but no editable installation or `PYTHONPATH` setup.
No new setup or download is needed in the current environment.

It generates a 1.6-second stereo tone at 48 kHz in a temporary directory and loads
`configs/preprocessing.example.json`. The service prepares 16 kHz mono audio,
saves one full WAV and five window WAVs, and writes the validated manifest. The
script checks the explicit window positions, real sample slices, zero padding,
approximately -20 dBFS RMS loudness, source hash, and repeat-save reuse. It removes
the temporary files before printing a JSON report with `"status": "passed"`.

This is synthetic permission evidence for a generated tone only. It grants no
permission for a real recording. The script does not open a microphone, download
data, or save audio in the project's raw/interim directories. A failed check raises
an error and returns a nonzero process exit code; checks also run under `python -O`.

## Project verification

```powershell
python -m pytest tests/test_preparation_integration.py -q
.\scripts\verify_project.ps1
```

The standard script compiles the package, runs all project tests, and runs the
smoke test. It now explicitly checks each Python command's exit code, so a failed
check cannot be hidden by a later successful command. `-SkipTests` runs compilation
only and skips both pytest and the smoke test.

No manual recording or installation is required. Review, commit, and push this
checkpoint when ready. The next task is A2.1: define the Log-Mel feature contract
and metadata schema.

## In plain language

Unit tests check separate tools. Integration tests check that those tools pass the
right data to each other and produce usable files together. A smoke test is a quick
run of the complete path with one sample: it tells you whether the main workflow
is working in your current environment.
