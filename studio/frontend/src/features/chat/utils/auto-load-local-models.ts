// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "../api/chat-api";

const GGUF_REPO_SUFFIX_RE = /-GGUF(?:$|-)/i;
// Only shard 1 of a split GGUF is loadable; llama.cpp rejects the rest with
// "model must be loaded with the first split" and finds the siblings itself.
const GGUF_TAIL_SHARD_RE = /-(\d{3,})-of-\d{3,}\.gguf$/i;

/** Last path segment, tolerating both separators and a trailing slash. */
function pathSegments(value: string): string[] {
  return value.replace(/\\/g, "/").split("/").filter(Boolean);
}

/** GGUF detection for a local model by backend format hint, name, or file path.
 * Name matching is scoped to the entry's own segment so a parent folder like
 * `/mnt/my-GGUF-drive/` can't misclassify a safetensors checkpoint. */
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

/** A GGUF that is a companion, not a main model: an mmproj vision adapter, a
 * separate MTP drafter (`mtp-*.gguf` or an `MTP/` folder), or a tail shard of a
 * split. Mirrors hub/utils/gguf.py `is_mtp_drafter_path` / `_is_mmproj_filename`,
 * which the backend applies everywhere main-model selection happens. */
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
 * and half-downloaded copies are never loadable, so they can't be auto-picked. */
export function isAutoLoadLocalModel(model: LocalModelInfo): boolean {
  if (model.partial || isGgufCompanionPath(model.path)) {
    return false;
  }
  return (
    model.source === "custom" ||
    model.source === "models_dir" ||
    model.source === "lmstudio"
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
