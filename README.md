<p align="center">
  <img src="docs/icon.png" width="128" height="128" alt="TellWell icon">
</p>

<h1 align="center">TellWell for Android</h1>

<p align="center">
  Free, private voice typing that runs entirely on your phone.
</p>

<p align="center">
  <a href="https://github.com/ayushpatnaikgit/tellwell-android/releases/latest/download/TellWell.apk"><img src="https://img.shields.io/badge/Download_for_Android-0b8a87?style=for-the-badge&logo=android&logoColor=white" height="56" alt="Download TellWell for Android"></a>
</p>

<p align="center">
  <a href="https://ayushpatnaikgit.github.io/tellwell/android/">Website</a> ·
  <a href="https://github.com/ayushpatnaikgit/tellwell">TellWell for Mac</a>
</p>

TellWell is a voice keyboard that works alongside the keyboard you already use. Type as usual; when you'd rather
talk, tap the globe key. TellWell starts listening at once. Tap when you're done: it types what you meant, without
the "um"s, repeated words and false starts, and hands you straight back to your keyboard.

TellWell is free. If it helps you, a star on GitHub helps other people find it.

## Install

1. On your phone, tap **Download for Android** above (or get the APK from the
   [latest release](https://github.com/ayushpatnaikgit/tellwell-android/releases/latest)) and open it. Android asks you to allow installs from your browser or file manager the first time.
2. Open TellWell and follow the three setup steps:
   - **Download the models** (about 1 GB, once; use Wi-Fi). The download keeps going if you lock the phone.
   - **Allow the microphone.**
   - **Turn on the keyboard.** Your current keyboard stays the main one.
3. In any app, tap the globe key on your keyboard (or the keyboard icon at the bottom of the screen) and pick TellWell.

Using TellWell:

- It starts listening as soon as it opens. **Tap** the waveform to finish; it types the text and switches back.
- **Hold** the waveform instead for push-to-talk.
- The keyboard key cancels and goes back without typing anything.

## Privacy

- Speech recognition and cleanup run on the phone. What you say is never sent anywhere.
- The microphone is on only while TellWell is open and listening (at most two minutes at a time). Audio stays in
  memory and is discarded; nothing is saved or logged.
- TellWell refuses to listen in password fields.
- The internet permission is used for one thing: downloading the two models over HTTPS from
  [tellwell-models](https://github.com/ayushpatnaikgit/tellwell-models) (copies of pinned Hugging Face versions, which
  are the fallback). Each file is kept only if its SHA-256 matches. All network code is in
  [`ModelDownloader.kt`](app/src/main/java/io/github/ayushpatnaikgit/tellwell/ModelDownloader.kt).
- No analytics, no crash reporting, no accounts.

## Phones

- Android 10 or newer, 64-bit (arm64). That covers nearly every phone from the last five years.
- About 1.1 GB of free memory while TellWell is open. Phones with 6 GB of RAM or more are comfortable.
- Speed: on a Motorola Edge 70 (Snapdragon 7 Gen 4), the text appears about one second after you tap.

TellWell picks the fastest part of the phone that gives correct results, checking each with a known test sentence:

1. the NPU on recent Snapdragon chips (Hexagon v73 and newer),
2. otherwise the GPU on Snapdragon (Adreno),
3. otherwise the CPU, which works on every phone but is slower.

Only the Motorola Edge 70 has been tested so far. Reports from other phones are welcome in the issues.

## How it works

| Step | Model | Runs with |
|---|---|---|
| Speech to text | [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (English), int8 ([sherpa-onnx export](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8)) | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and ONNX Runtime, CPU |
| Cleanup | [S1-mini FreeFlow Cleanup, Q4_0 GGUF](https://huggingface.co/ayushpatnaik/S1-mini-FreeFlow-cleanup-GGUF), a fine-tune of S1-mini by Superwhisper | [llama.cpp](https://github.com/ggml-org/llama.cpp): NPU, GPU or CPU |

The cleanup instructions stay cached in the model's memory between dictations, so each dictation only processes
your words. On the CPU, cleanup also uses copy-ahead decoding: it guesses the next few words by copying them from
the transcript and checks them in one pass, which gives the same text about 20% faster.

## Building

Requirements: JDK 17 or newer, the Android SDK (platform 36, build-tools 36), NDK 29 and CMake 3.31.

```sh
echo "sdk.dir=/path/to/android/sdk" > local.properties
gradle :app:assembleDebug
```

For a signed release build, add to `local.properties` (never commit it):

```properties
tellwell.keystore=/path/to/release.jks
tellwell.keystorePasswordFile=/path/to/password
```

and run `gradle :app:assembleRelease`.

The prebuilt native libraries in `app/src/main/jniLibs/arm64-v8a` are:

- **llama.cpp**, built with the Snapdragon toolchain (`ghcr.io/snapdragon-toolchain/arm64-android:v0.8`) from the
  integer-HMX branch of [ggml-org/llama.cpp#29740](https://github.com/ggml-org/llama.cpp/pull/29740), with
  `GGML_BACKEND_DL=ON` and `GGML_CPU_ALL_VARIANTS=ON`. Current llama.cpp master gives wrong results on the NPU of
  chips without FP16 matrix units, such as the Snapdragon 7 Gen 4.
- **sherpa-onnx** JNI and **ONNX Runtime**, from the sherpa-onnx Android release.

## Licence

TellWell for Android is released under the [MIT licence](LICENSE). It includes llama.cpp (MIT), sherpa-onnx
(Apache 2.0), ONNX Runtime (MIT), the Inter typeface (SIL Open Font License) and Material Design icons
(Apache 2.0); their licences are in the `THIRD_PARTY_*` files. The models are downloaded, not bundled: Parakeet is
licensed CC BY 4.0 by NVIDIA, and S1-mini is by Superwhisper under Apache 2.0 with a naming clause (see its model
card).
