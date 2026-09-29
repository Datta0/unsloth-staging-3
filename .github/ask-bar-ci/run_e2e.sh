#!/usr/bin/env bash
# End-to-end check of the macOS Ask bar on a real Apple Silicon runner with an MLX model.
# Needs: a debug desktop binary at $BIN, Unsloth installed at ~/.unsloth/studio, $MODEL cached,
# ask_driver built at $DRIVER. Writes evidence to $OUT (screenshots, window lists, logs, metrics).
set -uo pipefail
: "${BIN:?}" "${DRIVER:?}" "${MODEL:?}" "${OUT:?}"
mkdir -p "$OUT"
CONFIG_DIR="$HOME/Library/Application Support/ai.unsloth.studio"
STUDIO_LOGS="$HOME/.unsloth/studio/logs"
FAIL=0
fail() { echo "::error::$*"; FAIL=1; }
now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }
panel_up() { "$DRIVER" windows "$APP" | python3 -c '
import json,sys
rows=json.load(sys.stdin)
print("yes" if any(r["layer"]==101 and r["onscreen"] for r in rows) else "no")'; }
panel_exists() { "$DRIVER" windows "$APP" | python3 -c '
import json,sys
print("yes" if any(r["layer"]==101 for r in json.load(sys.stdin)) else "no")'; }
server_hits() { cat "$STUDIO_LOGS"/server/*.log 2>/dev/null | grep -c "$1" || true; }

launch() {
  "$BIN" > "$OUT/app-$1.log" 2>&1 &
  APP=$!
  echo "launched $1 pid=$APP"
  # Ready once the main window's app has made authenticated calls to its backend.
  for _ in $(seq 1 180); do
    kill -0 "$APP" 2>/dev/null || { fail "app exited during launch $1"; return 1; }
    if [ "$(server_hits 'request_completed')" -gt 5 ]; then
      sleep 15
      return 0
    fi
    sleep 2
  done
  fail "backend never served the app in launch $1"
  return 1
}

stop_app() {
  kill -TERM "$APP" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$APP" 2>/dev/null || break; sleep 1; done
  kill -KILL "$APP" 2>/dev/null || true
  wait "$APP" 2>/dev/null || true
}

echo "accessibility: $("$DRIVER" trusted)"

# 1) Off by default: no panel window, no extra WebKit process, the shortcut does nothing.
rm -f "$CONFIG_DIR/ask-bar-enabled"
launch off || true
if [ -n "${APP:-}" ] && kill -0 "$APP" 2>/dev/null; then
  pgrep -f com.apple.WebKit.WebContent | wc -l | tr -d ' ' > "$OUT/webcontent-off.txt"
  [ "$(panel_exists)" = no ] || fail "the Ask window exists while the bar is off"
  "$DRIVER" hotkey; sleep 2
  [ "$(panel_up)" = no ] || fail "Option+Space opened the panel while the bar is off"
  "$DRIVER" windows "$APP" > "$OUT/windows-off.json"
  stop_app
fi

# 2) On: the shortcut opens the panel over whatever is frontmost and it answers with the MLX model.
mkdir -p "$CONFIG_DIR"
echo true > "$CONFIG_DIR/ask-bar-enabled"
launch on || { echo "FAIL=$FAIL"; exit 1; }
pgrep -f com.apple.WebKit.WebContent | wc -l | tr -d ' ' > "$OUT/webcontent-on.txt"
grep -q "Ask bar: enabled" "$OUT/app-on.log" || fail "the app never enabled the Ask bar from the saved setting"
open -a TextEdit || true
sleep 3

t0=$(now_ms)
"$DRIVER" hotkey
shown=""
for _ in $(seq 1 50); do
  [ "$(panel_up)" = yes ] && { shown=$(( $(now_ms) - t0 )); break; }
  sleep 0.1
done
[ -n "$shown" ] || fail "Option+Space did not show the panel"
echo "hotkey_to_panel_ms=${shown:-none}" | tee -a "$OUT/metrics.txt"
"$DRIVER" windows "$APP" > "$OUT/windows-shown.json"
screencapture -x "$OUT/1-panel-open.png" || true

loads0=$(server_hits '/api/inference/load')
chats0=$(server_hits '/v1/chat/completions')
"$DRIVER" type "What is the capital of France? Answer in one short sentence."
sleep 0.5
screencapture -x "$OUT/2-question-typed.png" || true
t1=$(now_ms)
"$DRIVER" key return
answered=""
for _ in $(seq 1 600); do
  if [ "$(server_hits '/v1/chat/completions')" -gt "$chats0" ]; then
    answered=$(( $(now_ms) - t1 ))
    break
  fi
  sleep 0.5
done
sleep 5
screencapture -x "$OUT/3-answer.png" || true
[ -n "$answered" ] || fail "no /v1/chat/completions request reached the backend"
[ "$(server_hits '/api/inference/load')" -gt "$loads0" ] || fail "nothing was loaded, but no model was loaded before asking"
echo "enter_to_completion_ms=${answered:-none}" | tee -a "$OUT/metrics.txt"

# Follow-up in the same conversation.
chats1=$(server_hits '/v1/chat/completions')
"$DRIVER" type "And of Germany?"
"$DRIVER" key return
for _ in $(seq 1 240); do
  [ "$(server_hits '/v1/chat/completions')" -gt "$chats1" ] && break
  sleep 0.5
done
sleep 5
screencapture -x "$OUT/4-follow-up.png" || true
[ "$(server_hits '/v1/chat/completions')" -gt "$chats1" ] || fail "the follow-up never reached the backend"

# Escape hides it; the shortcut brings it back fresh; a click elsewhere dismisses it.
"$DRIVER" key escape; sleep 1
[ "$(panel_up)" = no ] || fail "Escape did not hide the panel"
"$DRIVER" hotkey; sleep 1.5
[ "$(panel_up)" = yes ] || fail "the panel did not come back on the second Option+Space"
screencapture -x "$OUT/5-reopened-fresh.png" || true
"$DRIVER" click 20 600; sleep 1.5
[ "$(panel_up)" = no ] || fail "a click in another app did not dismiss the panel"

# With Unsloth itself frontmost, a click in its own main window dismisses the panel too.
main_center=$("$DRIVER" windows "$APP" | python3 -c '
import json,sys
rows=[r for r in json.load(sys.stdin) if r["layer"]==0 and r["onscreen"]]
b=max(rows,key=lambda r:r["bounds"]["Width"]*r["bounds"]["Height"])["bounds"] if rows else None
print(f"{b[\"X\"]+b[\"Width\"]/2:.0f} {b[\"Y\"]+b[\"Height\"]/2:.0f}" if b else "")')
if [ -n "$main_center" ]; then
  "$DRIVER" click $main_center; sleep 1
  "$DRIVER" hotkey; sleep 1.5
  [ "$(panel_up)" = yes ] || fail "the panel did not open over Unsloth's own window"
  "$DRIVER" click $main_center; sleep 1.5
  [ "$(panel_up)" = no ] || fail "a click in Unsloth's main window did not dismiss the panel"
else
  fail "no on-screen main window to click"
fi

stop_app
cp "$STUDIO_LOGS"/server/*.log "$OUT/" 2>/dev/null || true
grep -h '"/v1/chat/completions"\|/api/inference/load\|/api/settings/last-local-model' "$OUT"/server-*.log \
  | tail -20 > "$OUT/ask-requests.log" || true
echo "FAIL=$FAIL"
exit "$FAIL"
