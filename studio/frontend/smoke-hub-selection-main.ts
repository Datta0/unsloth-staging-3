// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Cross-engine harness: runs the On Device selection logic inside a real
// browser, because the Node suite proves nothing about how Chromium, Firefox
// and WebKit handle percent-decoding, URIError, or non-ASCII case folding.

import {
  dedupeSameSourceHubCacheRows,
} from "./src/features/hub/inventory/inventory-dedupe";
import {
  buildCachedInventoryRow,
  buildLocalInventoryRows,
  cachedInventoryId,
  optimisticInventoryId,
} from "./src/features/hub/inventory/view-models";
import {
  resolveDownloadedSelection,
  resolveSelectionUrlSync,
} from "./src/features/hub/lib/selection-resolution";

type Case = { name: string; ok: boolean; detail?: string };
const results: Case[] = [];

function check(name: string, fn: () => boolean | string) {
  try {
    const outcome = fn();
    results.push(
      outcome === true
        ? { name, ok: true }
        : { name, ok: false, detail: String(outcome) },
    );
  } catch (error) {
    results.push({ name, ok: false, detail: `threw: ${String(error)}` });
  }
}

// deno-lint-ignore no-explicit-any
type Row = any;

function cachedRow(repoId: string, modelFormat: Row, over: Row = {}): Row {
  return buildCachedInventoryRow(
    {
      repo_id: repoId,
      inventory_id: cachedInventoryId(modelFormat, repoId),
      model_format: modelFormat,
      size_bytes: 100,
      ...over,
    } as Row,
    modelFormat,
  );
}

function unknownLocal(repoId: string, transport: string | null): Row {
  return buildLocalInventoryRows([
    {
      id: repoId,
      inventory_id: `hf_cache:unknown:${encodeURIComponent(repoId)}`,
      load_id: repoId,
      display_name: repoId.split("/").at(-1) ?? repoId,
      path: `/cache/models--${repoId.replace("/", "--")}`,
      source: "hf_cache",
      model_id: repoId,
      model_format: "unknown",
      partial: true,
      partial_transport: transport,
    },
  ] as Row)[0];
}

function resolve(id: string | null, cached: Row[], local: Row[]) {
  return resolveDownloadedSelection({
    selectedId: id,
    cachedRows: cached,
    localRows: local,
    filteredCachedRows: cached,
    filteredLocalRows: local,
  });
}

// --- encoding round trip, the engine's own encodeURIComponent -----------------
check("canonical IDs round-trip for ASCII repos", () => {
  const row = cachedRow("unsloth/gemma-3-270m-it", "gguf");
  if (row.id !== "cache:gguf:unsloth%2Fgemma-3-270m-it") return row.id;
  return resolve(row.id, [row], []).selectedId === row.id || "not resolved";
});

check("canonical IDs round-trip for non-ASCII repos", () => {
  for (const repoId of ["组织/模型", "org/модель", "org/modèle-café", "org/e-\u{1F600}"]) {
    const row = cachedRow(repoId, "safetensors");
    if (resolve(row.id, [row], []).selectedId !== row.id) return repoId;
  }
  return true;
});

check("legacy unencoded deep links still resolve", () => {
  const row = cachedRow("unsloth/gemma-3-270m-it", "gguf");
  return (
    resolve("cache:gguf:unsloth/gemma-3-270m-it", [row], []).selectedId ===
      row.id || "legacy ID lost"
  );
});

check("lowercase percent escapes resolve", () => {
  const row = cachedRow("unsloth/gemma-3-270m-it", "gguf");
  return (
    resolve("cache:gguf:unsloth%2fgemma-3-270m-it", [row], []).selectedId ===
      row.id || "%2f not accepted"
  );
});

// --- URIError: the engine-specific one ---------------------------------------
check("malformed percent escapes never throw", () => {
  const row = cachedRow("org/repo", "gguf");
  for (const id of [
    "cache:gguf:%",
    "cache:gguf:%2",
    "cache:gguf:%ZZ",
    "cache:gguf:%E0%A4%A",
    "cache:gguf:%C3%28",
    "cache:gguf:%ED%A0%80",
  ]) {
    const out = resolve(id, [row], []);
    if (out.selectedId !== null) return `${id} selected ${out.selectedId}`;
  }
  return true;
});

check("decodeURIComponent really does throw here", () => {
  // Guards the guard: if an engine stopped throwing, the try/catch above would
  // be passing for the wrong reason.
  try {
    decodeURIComponent("%");
    return "engine did not throw on '%'";
  } catch {
    return true;
  }
});

// --- case folding ------------------------------------------------------------
check("repo keys fold consistently for non-ASCII", () => {
  const row = cachedRow("ORG/MODÈLE", "gguf");
  return (
    resolve(`cache:gguf:${encodeURIComponent("org/modèle")}`, [row], [])
      .selectedId === row.id || "case folding mismatch"
  );
});

// --- the behaviour the PR is for ---------------------------------------------
check("selection survives unknown -> live gguf -> cache", () => {
  const repoId = "unsloth/gemma-3-270m-it";
  const selected = `hf_cache:unknown:${encodeURIComponent(repoId)}`;
  const live = cachedRow(repoId, "gguf", { partial: true, optimistic: true });
  if (resolve(selected, [live], []).selectedId !== live.id) return "lost at live";
  const done = cachedRow(repoId, "gguf");
  if (resolve(selected, [done], []).selectedId !== done.id) return "lost at complete";
  return true;
});

check("a gguf download does not adopt a proven snapshot partial", () => {
  const repoId = "unsloth/hybrid-repo";
  const snapshot = unknownLocal(repoId, "xet");
  return (
    resolve(optimisticInventoryId("gguf", repoId), [], [snapshot]).selectedId ===
      null || "adopted the wrong family"
  );
});

check("a transport-less partial is still adopted", () => {
  const repoId = "unsloth/gguf-repo";
  const partial = unknownLocal(repoId, null);
  return (
    resolve(optimisticInventoryId("gguf", repoId), [], [partial]).selectedId ===
      partial.id || "lost its own partial"
  );
});

check("dedupe keeps an unrelated hybrid partial", () => {
  const repoId = "unsloth/hybrid-repo";
  const snapshot = unknownLocal(repoId, "xet");
  const gguf = cachedRow(repoId, "gguf", { partial: true });
  const out = dedupeSameSourceHubCacheRows({
    cachedRows: [gguf],
    localRows: [snapshot],
  });
  return out.localRows.length === 1 || `kept ${out.localRows.length}`;
});

check("a Windows path is not read as a repo ID", () => {
  const row = cachedRow("org/repo", "gguf");
  for (const id of ["C:\\Users\\me\\models\\x", "\\\\server\\share\\m"]) {
    if (resolve(id, [row], []).selectedId !== null) return id;
  }
  return true;
});

check("the gguf file query survives canonicalization", () => {
  const sync = resolveSelectionUrlSync({
    isDiscoverTab: false,
    urlModel: "cache:gguf:org/repo",
    selectionInputId: "cache:gguf:org/repo",
    resolvedSelectedId: "cache:gguf:org%2Frepo",
    resolvedModelFormat: "gguf",
  });
  return (
    (sync?.action === "replace" && sync.preserveGgufFile === true) ||
    JSON.stringify(sync)
  );
});

check("resolution is idempotent over a generated sweep", () => {
  const repoId = "org/repo";
  const formats = ["gguf", "safetensors", "adapter", "checkpoint", "unknown"];
  let n = 0;
  for (const rowFormat of formats) {
    for (const transport of [null, "xet"]) {
      for (const partial of [false, true]) {
        const cached = [
          cachedRow(repoId, rowFormat, {
            partial,
            partial_transport: transport,
          }),
        ];
        for (const source of ["cache", "download", "hf_cache"]) {
          for (const selFormat of formats) {
            const id = `${source}:${selFormat}:${encodeURIComponent(repoId)}`;
            const once = resolve(id, cached, []).selectedId;
            const twice = resolve(once, cached, []).selectedId;
            if (once !== twice) return `${id}: ${once} -> ${twice}`;
            n += 1;
          }
        }
      }
    }
  }
  return n >= 300 || `only ${n} combinations`;
});

const failed = results.filter((r) => !r.ok);
const payload = {
  total: results.length,
  passed: results.length - failed.length,
  failed,
  userAgent: navigator.userAgent,
};
// deno-lint-ignore no-explicit-any
(window as any).__hubSelectionResults = payload;
const out = document.getElementById("out");
if (out) out.textContent = JSON.stringify(payload, null, 2);
