// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "../api/chat-api";

const GGUF_REPO_SUFFIX_RE = /-GGUF(?:$|-)/i;
// Kept in step with `isMlxId` in model-selector/recommended-fit.ts, which gates
// the picker's MLX rows. Copied rather than imported so this module keeps no
// value imports: it is loaded standalone by its contract tests, and a test pins
// the two literals against each other.
const MLX_RE = /-MLX(?:$|-)/i;
// The publisher LM Studio rows carry for MLX builds, whose folder name alone
// often has no -MLX token. Mirrors `_looks_like_mlx_repo` in routes/models.py.
const MLX_REPO_PREFIX = "mlx-community/";
// Only shard 1 of a split GGUF is loadable; llama.cpp finds the siblings itself.
const GGUF_TAIL_SHARD_RE = /-(\d{3,})-of-\d{3,}\.gguf$/i;

/** Path segments, tolerating both separators and a trailing slash. */
function pathSegments(value: string): string[] {
  return value.replace(/\\/g, "/").split("/").filter(Boolean);
}

/** GGUF detection by backend format hint, name, or path. Name matching is scoped
 * to the entry's own segment so `/mnt/my-GGUF-drive/` can't misclassify it. */
export function localModelIsGguf(model: LocalModelInfo): boolean {
  const segments = pathSegments(model.id);
  const name = segments[segments.length - 1] ?? model.id;
  return (
    model.model_format === "gguf" ||
    GGUF_REPO_SUFFIX_RE.test(name) ||
    GGUF_REPO_SUFFIX_RE.test(model.display_name) ||
    model.path.toLowerCase().endsWith(".gguf")
  );
}

export function isDirectGgufPath(path: string): boolean {
  return path.toLowerCase().endsWith(".gguf");
}

/** A companion GGUF, not a main model: an mmproj vision adapter, a separate MTP
 * drafter (`mtp-*.gguf` or an `MTP/` folder), or a tail shard of a split.
 * Mirrors hub/utils/gguf.py `is_mtp_drafter_path` / `_is_mmproj_filename`. */
export function isGgufCompanionPath(path: string): boolean {
  const segments = pathSegments(path);
  const name = (segments[segments.length - 1] ?? "").toLowerCase();
  const parent = (segments[segments.length - 2] ?? "").toLowerCase();
  if (
    name.includes("mmproj") ||
    name.startsWith("mtp-") ||
    name === "mtp" ||
    parent === "mtp"
  ) {
    return true;
  }
  const shard = GGUF_TAIL_SHARD_RE.exec(name);
  return shard !== null && Number(shard[1]) > 1;
}

/** Local models outside the HF cache that auto-load should consider. Companions
 * and partial downloads are never loadable. The companion rules apply to GGUF
 * rows only, like the backend predicates they mirror (which return False for any
 * non-`.gguf` path): a safetensors checkpoint in an `mtp-...` folder is real.
 *
 * The classification has to be positive rather than "not a known non-GGUF
 * format". `/api/models/local` reports only `"gguf"` or nothing: every
 * `model_format` in `routes/models.py` is the literal `"gguf"` or comes from
 * `_dir_model_format`, which returns `"gguf"` or `None`. The richer vocabulary
 * ("safetensors", "adapter", "unknown") belongs to the hub inventory schema,
 * which feeds the picker, not this path, so keying off it never matches a real
 * row and leaves genuine checkpoints under `mtp-.../MTP/mmproj...` unloadable.
 *
 * The case that gives up: a directory the scanner left unclassified because its
 * only GGUFs are mmproj projectors, and which is also named like a companion.
 * That one becomes a candidate whose load fails and is skipped, rather than a
 * real checkpoint that can never be auto-loaded at all. Directories holding a
 * main GGUF (MTP drafters included) are classified `"gguf"` and still refused. */
export function isAutoLoadLocalModel(model: LocalModelInfo): boolean {
  if (model.partial) {
    return false;
  }
  // A folder holding both safetensors and GGUF weights. Loading it resolves to a
  // GGUF whichever kind is asked for, because ModelConfig.from_identifier runs
  // detect_gguf_model on any local path first and that takes the largest file, so
  // the checkpoint path would silently load the biggest quant and then record the
  // run as non-GGUF with a 4096-token cap. Which one the user wants is genuinely
  // ambiguous, and this sweep runs unattended, so it is left to an explicit pick
  // where the variant is chosen deliberately. The picker still lists it.
  if (model.model_format === "mixed") {
    return false;
  }
  if (localModelIsGguf(model) && isGgufCompanionPath(model.path)) {
    return false;
  }
  return (
    model.source === "custom" ||
    model.source === "models_dir" ||
    model.source === "lmstudio"
  );
}

/** An MLX build on a host that cannot run one. `core/inference/worker.py` picks
 * the MLX runner purely from the detected device, so anywhere else the checkpoint
 * falls through to the transformers worker and cannot load; /validate has no MLX
 * preflight, so the candidate passes the guard, spends one of the three auto-load
 * attempts and only then fails.
 *
 * `hostRunsMlx` is narrower than "is a Mac", which is what the picker gates on:
 * `/api/health` derives `device_type` from `sys.platform` alone, so an Intel Mac
 * and an Apple Silicon Mac whose MLX stack is missing or broken both report
 * "mac" while `detect_hardware` falls through to CPU. Name-based matching, like
 * the picker: the inventory carries no runtime field. */
export function isUnsupportedMlxLocalModel(
  model: LocalModelInfo,
  hostRunsMlx: boolean,
): boolean {
  if (hostRunsMlx) {
    return false;
  }
  const modelId = model.model_id ?? "";
  // Scoped to the entry's own segment, like localModelIsGguf, so a parent such
  // as /mnt/my-MLX-models/ cannot condemn every checkpoint underneath it.
  const segments = pathSegments(model.id);
  const name = segments[segments.length - 1] ?? model.id;
  return (
    MLX_RE.test(name) ||
    MLX_RE.test(model.display_name) ||
    MLX_RE.test(modelId) ||
    modelId.toLowerCase().startsWith(MLX_REPO_PREFIX)
  );
}

export function findLocalModel(
  models: LocalModelInfo[],
  id: string,
): LocalModelInfo | undefined {
  const normalized = id.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }
  return models.find((model) => {
    if (model.id.toLowerCase() === normalized) {
      return true;
    }
    if (model.path.toLowerCase() === normalized) {
      return true;
    }
    // The Hub-style alias, only for a cache snapshot. LM Studio rows carry a
    // publisher/model-name model_id of the same shape (and Ollama rows an
    // ollama/name:tag one) for an independent copy with its own files, so a
    // remembered `org/model` must not resolve to one of those. Their own memory
    // is the absolute path, matched above.
    if (model.active_cache !== false) {
      return false;
    }
    const modelId = model.model_id?.trim().toLowerCase();
    return Boolean(modelId && modelId === normalized);
  });
}

/** Prefer recently touched local models, then stable name order. */
export function sortLocalModelsForAutoLoad(
  models: LocalModelInfo[],
): LocalModelInfo[] {
  const name = (model: LocalModelInfo) =>
    model.model_id ?? model.display_name ?? model.id;
  return [...models].sort((a, b) => {
    const updatedDiff = (b.updated_at ?? 0) - (a.updated_at ?? 0);
    if (updatedDiff !== 0) {
      return updatedDiff;
    }
    return name(a).localeCompare(name(b));
  });
}
