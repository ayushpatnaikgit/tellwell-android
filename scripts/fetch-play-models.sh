#!/bin/sh
# Fills the Play asset packs (models_common, models_cpu, models_npu) with the model files, from TellWell's model
# release, keeping each file only if its SHA-256 matches. Run once before `./gradlew :app:bundlePlayRelease`.
# A file already present with the right hash is kept. MODELS_CACHE=<dir> copies from a local folder first.
set -eu
cd "$(dirname "$0")/.."
BASE=https://github.com/ayushpatnaikgit/tellwell-models/releases/download/v1

fetch() {  # fetch <pack> <file name in the pack> <release asset> <sha256>
  out="models_$1/src/main/assets/$2"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(shasum -a 256 "$out" | cut -d' ' -f1)" = "$4" ]; then echo "ok    $out"; return; fi
  if [ -n "${MODELS_CACHE:-}" ] && [ -f "$MODELS_CACHE/$3" ]; then cp "$MODELS_CACHE/$3" "$out.part"
  else curl -fL --retry 3 -C - -o "$out.part" "$BASE/$3"; fi
  [ "$(shasum -a 256 "$out.part" | cut -d' ' -f1)" = "$4" ] || { rm -f "$out.part"; echo "SHA-256 mismatch: $3" >&2; exit 1; }
  mv "$out.part" "$out"
  echo "fetched $out"
}

fetch common tokens.txt parakeet-tdt-0.6b-v2-int8-tokens.txt ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d
fetch common decoder.int8.onnx parakeet-tdt-0.6b-v2-int8-decoder.int8.onnx 4d097bae3bac5c96018d5293ba39f1b657296102b142affda40378f38b020f41
fetch common joiner.int8.onnx parakeet-tdt-0.6b-v2-int8-joiner.int8.onnx 77a29f3e5ac7d37074e0af465391b80f314015d5b97f0f90c6c7767f1d56f428
fetch common s1-mini-freeflow-cleanup-q4_0.gguf s1-mini-freeflow-cleanup-q4_0.gguf ff7dfbc19d7220ac7e1f7cd88bc95da3db7e7e7893a35ea552895727605abddc
fetch cpu encoder.int8.onnx parakeet-tdt-0.6b-v2-int8-encoder.int8.onnx 92a10bd9c23936ea0566c29f963bc57c2a5df4261b6a8921c39abea66557ca8b
fetch npu enc-w8a16-1000.onnx parakeet-tdt-0.6b-v2-npu-w8a16-10s.onnx e50e3a70a6b8ae12596a3c709fe4633a154e3ce9ac889a233a0a8eaf74e96848
fetch npu enc-w8a16-1000.onnx.data parakeet-tdt-0.6b-v2-npu-w8a16-10s.onnx.data 92538fb20215e4b237d46daf4b082cc43dd478e90ebc6c8c06f8fbdb881a326e
