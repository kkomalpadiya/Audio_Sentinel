# B3.1 — Isolated YAMNet model loader

## What this task adds

`audio_sentinel.acoustic_loader.load_yamnet` loads the pinned YAMNet v1
SavedModel from `models/yamnet/1`. It does not download anything and does not run
inference. Before TensorFlow parses the model, the loader verifies the complete
four-file payload, its combined SHA-256 digest, the official 521-row vocabulary,
and path containment without links or junctions.

After loading, it verifies the `serving_default` contract: one float32 waveform
input shaped `(samples,)`, then float32 score, embedding, and model-owned Log-Mel
outputs shaped `(patches, 521)`, `(patches, 1024)`, and `(frames, 64)`. The result
contains the callable model plus JSON-ready version metadata. The metadata records
the model/version/handle, local relative path, artifact and vocabulary digests,
label-mapping version, runtime version, SavedModel exporter version, and the
verified tensor contract. It never exposes an absolute local path.

The model artifact is Google YAMNet v1 from its official Kaggle mirror. The old
TF Hub identity remains the origin of this version, but local setup now uses the
Kaggle handle `google/yamnet/tensorFlow2/yamnet/1`. The pinned payload digest is
`2aee541e6039364299c90cfe5a715d239097aafb38aa4ce50d805a5445993b82`.
The model was exported by TensorFlow 2.3.0 and is loaded here with TensorFlow
2.21.0 on Python 3.13.

Official references:

- [Google YAMNet model](https://www.kaggle.com/models/google/yamnet/tensorFlow2/yamnet/1)
- [TensorFlow YAMNet tutorial](https://www.tensorflow.org/hub/tutorials/yamnet)
- [TensorFlow pip installation](https://www.tensorflow.org/install/pip)
- [KaggleHub model downloads](https://github.com/Kaggle/kagglehub#download-model)

## One-time local setup

From the project root, run:

```powershell
.\scripts\setup_yamnet.ps1
```

The script creates `.venv/yamnet`, installs the pinned optional `yamnet`
dependencies, downloads the public model into `models/yamnet/1`, verifies its
digest, and runs the loader smoke test. Both the environment and model weights
are ignored by Git. A future clean checkout or another machine must run this
command once; normal source-only tests do not require TensorFlow or model files.

To rerun just the real loader check:

```powershell
.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_loader.py
```

## Failure behavior

The loader returns stable error codes for a missing model, invalid paths,
unexpected files, changed bytes, missing or mismatched TensorFlow, a changed
vocabulary asset, and signature drift. Verification occurs before runtime import
where possible. The loader performs no network access and no inference; those
boundaries keep ordinary project imports and tests lightweight.

In plain language: setup downloads the known recognizer once. Every later load
checks that the recognizer is still the exact expected copy and that its input and
output plugs have not changed. The next task can safely send prepared waveforms
through it without mixing model installation, identity checks, and inference.
