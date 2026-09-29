// Drives the macOS Ask bar the way a user does: global key events and the window server's view.
// Usage: ask_driver hotkey | type "<text>" | key return|escape | windows <pid> | click <x> <y>
import AppKit
import CoreGraphics
import Foundation

func post(_ code: CGKeyCode, flags: CGEventFlags = []) {
    let source = CGEventSource(stateID: .hidSystemState)
    for down in [true, false] {
        let event = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: down)!
        event.flags = flags
        event.post(tap: .cghidEventTap)
        usleep(30_000)
    }
}

func typeText(_ text: String) {
    let source = CGEventSource(stateID: .hidSystemState)
    for scalar in text.utf16 {
        var unit = scalar
        for down in [true, false] {
            let event = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: down)!
            event.keyboardSetUnicodeString(stringLength: 1, unicodeString: &unit)
            event.post(tap: .cghidEventTap)
        }
        usleep(15_000)
    }
}

func windows(pid: Int32) {
    let list = CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]] ?? []
    var rows: [[String: Any]] = []
    for info in list where (info[kCGWindowOwnerPID as String] as? Int32) == pid {
        rows.append([
            "name": info[kCGWindowName as String] as? String ?? "",
            "layer": info[kCGWindowLayer as String] as? Int ?? -1,
            "onscreen": info[kCGWindowIsOnscreen as String] as? Bool ?? false,
            "bounds": info[kCGWindowBounds as String] ?? [:],
        ])
    }
    let data = try! JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

let args = CommandLine.arguments
switch args.count > 1 ? args[1] : "" {
case "trusted":
    print(AXIsProcessTrusted() ? "trusted" : "untrusted")
case "hotkey":
    post(49, flags: .maskAlternate)  // Option+Space
case "type":
    typeText(args[2])
case "key":
    post(args[2] == "return" ? 36 : 53)
case "windows":
    windows(pid: Int32(args[2])!)
case "click":
    let point = CGPoint(x: Double(args[2])!, y: Double(args[3])!)
    for type in [CGEventType.leftMouseDown, .leftMouseUp] {
        CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: point, mouseButton: .left)!
            .post(tap: .cghidEventTap)
        usleep(50_000)
    }
default:
    FileHandle.standardError.write("unknown command\n".data(using: .utf8)!)
    exit(2)
}
