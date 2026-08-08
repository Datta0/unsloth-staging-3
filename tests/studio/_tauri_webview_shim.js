// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Test double for the Tauri v2 webview runtime plus the parts of
// studio/src-tauri/src/main.rs the desktop frontend talks to, so the REAL built
// frontend can be driven in Chromium on a runner with no Tauri shell.
//
// Injected with `context.add_init_script` BEFORE the bundle loads, because
// src/lib/api-base.ts decides `isTauri` at module scope from the presence of
// window.__TAURI_INTERNALS__.
//
// Only three things here are behaviour rather than plumbing:
//   * desktop_preflight answers `owned_ready`, which is what a normal launch
//     against an already-running desktop-owned backend returns.
//   * desktop_auth hands back tokens, which is what Rust does after logging in.
//   * plugin:window|close prevents the close and emits `app-closing`, which is
//     where main.rs's quit_sequence raises the overlay: after the quit
//     confirmations, before cleanup_child_processes. The reap then never
//     finishes, which is the wedged case the Force quit button exists for.
(() => {
  const CFG = window.__UNSLOTH_SHIM_CFG__ || {};
  const callbacks = new Map();
  let nextCallbackId = 1;
  const listeners = new Map();
  let nextEventId = 1;

  function transformCallback(callback, once = false) {
    const id = nextCallbackId++;
    callbacks.set(id, { fn: callback, once });
    return id;
  }

  function emit(eventName, payload) {
    const forEvent = listeners.get(eventName);
    if (!forEvent) return 0;
    for (const [eventId, callbackId] of [...forEvent]) {
      const entry = callbacks.get(callbackId);
      if (!entry) continue;
      if (entry.once) callbacks.delete(callbackId);
      entry.fn({ event: eventName, id: eventId, payload });
    }
    return forEvent.size;
  }

  const STUBS = {
    desktop_preflight: {
      disposition: "owned_ready",
      reason: null,
      port: CFG.backendPort,
      can_auto_repair: false,
      managed_bin: null,
    },
    desktop_auth: {
      access_token: CFG.accessToken || "",
      refresh_token: CFG.refreshToken || "",
    },
    desktop_update_policy: {
      mode: "in_app",
      releasePageBaseUrl: "https://github.com/unslothai/unsloth/releases/tag/",
      releaseTagPrefix: "desktop-v",
    },
    was_launched_hidden: false,
    check_health: true,
    check_desktop_update: null,
    check_desktop_manual_update: null,
    has_initialized_app_window_layout: true,
    has_saved_window_state: true,
    drain_native_intents: [],
    set_renderer_activity: null,
    set_training_active: null,
    "plugin:window|scale_factor": 1,
    "plugin:window|inner_size": { width: CFG.width || 1440, height: CFG.height || 900 },
    "plugin:window|outer_size": { width: CFG.width || 1440, height: CFG.height || 900 },
    "plugin:window|inner_position": { x: 0, y: 0 },
    "plugin:window|outer_position": { x: 0, y: 0 },
    "plugin:window|is_maximized": false,
    "plugin:window|is_minimized": false,
    "plugin:window|is_fullscreen": false,
    "plugin:window|is_visible": true,
    "plugin:window|is_focused": true,
    "plugin:window|is_decorated": false,
    "plugin:window|is_resizable": true,
    "plugin:window|theme": "light",
    "plugin:window|title": "Unsloth",
    "plugin:window|current_monitor": {
      name: "test",
      size: { width: 1920, height: 1080 },
      position: { x: 0, y: 0 },
      scaleFactor: 1,
    },
    "plugin:window|available_monitors": [],
  };

  async function invoke(cmd, args = {}) {
    if (cmd === "plugin:event|listen") {
      const eventId = nextEventId++;
      if (!listeners.has(args.event)) listeners.set(args.event, new Map());
      listeners.get(args.event).set(eventId, args.handler);
      return eventId;
    }
    if (cmd === "plugin:event|unlisten") {
      const forEvent = listeners.get(args.event);
      if (forEvent) forEvent.delete(args.eventId);
      return null;
    }
    if (cmd === "plugin:window|close" || cmd === "plugin:window|destroy") {
      // api.prevent_close(): the window stays up while the quit thread runs.
      window.__SHIM_CLOSE_REQUESTED__ = (window.__SHIM_CLOSE_REQUESTED__ || 0) + 1;
      setTimeout(() => {
        window.__SHIM_APP_CLOSING_HEARD_BY__ = emit("app-closing", null);
      }, 0);
      return null;
    }
    if (cmd === "force_quit") {
      window.__SHIM_FORCE_QUIT__ = (window.__SHIM_FORCE_QUIT__ || 0) + 1;
      return null;
    }
    if (cmd in STUBS) return STUBS[cmd];
    if (cmd.startsWith("plugin:window|") || cmd.startsWith("plugin:webview|")) return null;
    if (cmd.startsWith("plugin:updater|")) return null;
    // Real Tauri rejects an unregistered command and the frontend has fallbacks
    // for that; inventing a return shape is how a stub silently breaks the tree.
    throw new Error(`Command ${cmd} not found`);
  }

  window.__TAURI_INTERNALS__ = {
    invoke,
    transformCallback,
    unregisterCallback: (id) => callbacks.delete(id),
    convertFileSrc: (p) => p,
    metadata: {
      currentWindow: { label: "main" },
      currentWebview: { windowLabel: "main", label: "main" },
    },
  };
  window.__TAURI_EVENT_PLUGIN_INTERNALS__ = {
    unregisterListener(eventName, eventId) {
      const forEvent = listeners.get(eventName);
      if (forEvent) forEvent.delete(eventId);
    },
  };
  window.__SHIM__ = {
    emit,
    listenerCount: (e) => (listeners.get(e) || new Map()).size,
  };

  // The overlay is raised for Windows only, so report the platform it ships on.
  // shouldUseCustomWindowTitlebar() accepts win/linux/x11 alike, so this changes
  // no layout.
  try {
    Object.defineProperty(navigator, "userAgentData", {
      configurable: true,
      get: () => ({ platform: "Windows", mobile: false, brands: [] }),
    });
  } catch (error) {
    /* keep the real platform rather than half-spoofing it */
  }
})();
