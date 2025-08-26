import csv
import time
import math
import threading
from collections import defaultdict, deque

import psutil
from pynput import mouse, keyboard

import ctypes
from ctypes import wintypes

import win32gui
import win32process
import win32con

import argparse
from pathlib import Path
import statistics

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

def get_screen_size():
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def get_foreground_process():
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        return None, None, None
    try:
        tid, pid = win32process.GetWindowThreadProcessId(hwnd)
        name = None
        try:
            name = psutil.Process(pid).name()
        except Exception:
            name = f"pid_{pid}"
        return pid, name, hwnd
    except Exception:
        return None, None, None

def get_window_show_cmd(hwnd):
    try:
        placement = win32gui.GetWindowPlacement(hwnd)
        return placement[1]
    except Exception:
        return None

class TelemetryAggregator:
    def __init__(self, cols=16, rows=9, pause_speed_thresh=5.0, window_seconds=60):
        self.cols = cols
        self.rows = rows
        self.pause_speed_thresh = pause_speed_thresh
        self.window_seconds = window_seconds

        self.screen_w, self.screen_h = get_screen_size()

        self.lock = threading.Lock()

        self.last_move_t = None
        self.last_x = None
        self.last_y = None
        self.instant_speeds = []

        self.heat_counts = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        self.left_down_times = deque()
        self.right_down_times = deque()

        self.left_holds = []
        self.right_holds = []

        self.click_times = []

        self.pause_times = []

        self.key_down_times = {}
        self.key_down_sequence = []
        self.key_release_sequence = []
        self.key_holds = []
        self.key_press_count = 0

        self.gui_last_pid_name = None
        self.gui_switch_count = 0
        self.gui_focus_accum = defaultdict(float)
        self.gui_last_change_t = None
        self.gui_last_hwnd = None
        self.gui_last_showcmd = None
        self.gui_window_event_count = 0

        self.session_t0 = time.monotonic()
        self.window_start = self.session_t0

    @staticmethod
    def r3(x):
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "0.000"

    def _bin_pos(self, x, y):
        xi = max(0, min(self.cols - 1, int(x / max(1, self.screen_w) * self.cols)))
        yi = max(0, min(self.rows - 1, int(y / max(1, self.screen_h) * self.rows)))
        return xi, yi

    def on_move(self, x, y):
        t = time.monotonic()
        with self.lock:
            if self.last_x is not None and self.last_y is not None and self.last_move_t is not None:
                dt = t - self.last_move_t
                if dt > 0:
                    dist = math.hypot(x - self.last_x, y - self.last_y)
                    spd = dist / dt
                    self.instant_speeds.append(spd)
                    if spd < self.pause_speed_thresh:
                        self.pause_times.append(t)
            self.last_x, self.last_y, self.last_move_t = x, y, t
            xi, yi = self._bin_pos(x, y)
            self.heat_counts[yi][xi] += 1

    def on_click(self, x, y, button, pressed):
        t = time.monotonic()
        with self.lock:
            if pressed:
                if hasattr(button, "name") and button.name == "left":
                    self.left_down_times.append(t)
                elif hasattr(button, "name") and button.name == "right":
                    self.right_down_times.append(t)
            else:
                if hasattr(button, "name") and button.name == "left":
                    if self.left_down_times:
                        dur = t - self.left_down_times.popleft()
                        self.left_holds.append(dur)
                    self.click_times.append(t)
                elif hasattr(button, "name") and button.name == "right":
                    if self.right_down_times:
                        dur = t - self.right_down_times.popleft()
                        self.right_holds.append(dur)
                    self.click_times.append(t)

    def on_scroll(self, x, y, dx, dy):
        with self.lock:
            xi, yi = self._bin_pos(x, y)
            self.heat_counts[yi][xi] += 1

    def on_key_press(self, key):
        t = time.monotonic()
        with self.lock:
            if key not in self.key_down_times:
                self.key_down_times[key] = t
                self.key_down_sequence.append(t)
                self.key_press_count += 1

    def on_key_release(self, key):
        t = time.monotonic()
        with self.lock:
            t0 = self.key_down_times.pop(key, None)
            if t0 is not None:
                self.key_holds.append(t - t0)
            self.key_release_sequence.append(t)

    def poll_gui(self):
        now = time.monotonic()
        pid, name, hwnd = get_foreground_process()
        showcmd = get_window_show_cmd(hwnd) if hwnd else None

        with self.lock:
            if self.gui_last_pid_name is None:
                self.gui_last_pid_name = name
                self.gui_last_change_t = now
            else:
                if name != self.gui_last_pid_name:
                    if self.gui_last_change_t is not None:
                        self.gui_focus_accum[self.gui_last_pid_name] += now - self.gui_last_change_t
                    self.gui_switch_count += 1
                    self.gui_last_pid_name = name
                    self.gui_last_change_t = now
                else:
                    pass

            if hwnd and showcmd is not None:
                if self.gui_last_hwnd != hwnd:
                    self.gui_last_hwnd = hwnd
                    self.gui_last_showcmd = showcmd
                else:
                    if self.gui_last_showcmd is not None and showcmd != self.gui_last_showcmd:
                        if showcmd in (win32con.SW_MINIMIZE, win32con.SW_MAXIMIZE, win32con.SW_SHOWMINIMIZED, win32con.SW_SHOWMAXIMIZED):
                            self.gui_window_event_count += 1
                        self.gui_last_showcmd = showcmd

    def _compute_mouse_features(self):
        avg_speed = statistics.fmean(self.instant_speeds) if self.instant_speeds else 0.0
        delays = []
        if self.click_times and self.pause_times:
            p = 0
            pauses = list(self.pause_times)
            for ct in self.click_times:
                prevs = [pt for pt in pauses if pt <= ct]
                if prevs:
                    delays.append(ct - prevs[-1])
        click_pause_mean = statistics.fmean(delays) if delays else 0.0
        click_pause_std  = statistics.pstdev(delays) if len(delays) > 1 else 0.0

        left_mean = statistics.fmean(self.left_holds) if self.left_holds else 0.0
        left_std  = statistics.pstdev(self.left_holds) if len(self.left_holds) > 1 else 0.0
        right_mean = statistics.fmean(self.right_holds) if self.right_holds else 0.0
        right_std  = statistics.pstdev(self.right_holds) if len(self.right_holds) > 1 else 0.0

        return avg_speed, click_pause_mean, click_pause_std, left_mean, left_std, right_mean, right_std

    def _compute_keyboard_features(self, elapsed_s):
        n = self.key_press_count
        holds = list(self.key_holds)
        dd = []
        rp = []
        rr = []

        if len(self.key_down_sequence) >= 2:
            for i in range(1, len(self.key_down_sequence)):
                dd.append(self.key_down_sequence[i] - self.key_down_sequence[i-1])
        if self.key_down_sequence and self.key_release_sequence:
            m = min(len(self.key_down_sequence), len(self.key_release_sequence))
            for i in range(1, m):
                rp.append(self.key_down_sequence[i] - self.key_release_sequence[i-1])
        if len(self.key_release_sequence) >= 2:
            for i in range(1, len(self.key_release_sequence)):
                rr.append(self.key_release_sequence[i] - self.key_release_sequence[i-1])

        avg_hold = statistics.fmean(holds) if holds else 0.0
        std_hold = statistics.pstdev(holds) if len(holds) > 1 else 0.0
        avg_dd = statistics.fmean(dd) if dd else 0.0
        std_dd = statistics.pstdev(dd) if len(dd) > 1 else 0.0
        avg_rp = statistics.fmean(rp) if rp else 0.0
        std_rp = statistics.pstdev(rp) if len(rp) > 1 else 0.0
        avg_rr = statistics.fmean(rr) if rr else 0.0
        cpm = (n / elapsed_s) * 60.0 if elapsed_s > 0 else 0.0

        return n, avg_hold, std_hold, avg_dd, std_dd, avg_rp, std_rp, avg_rr, cpm

    def _compute_gui_features(self, window_end_t):
        if self.gui_last_pid_name and self.gui_last_change_t is not None:
            self.gui_focus_accum[self.gui_last_pid_name] += (window_end_t - self.gui_last_change_t)
            self.gui_last_change_t = window_end_t

        focus_time = max(self.gui_focus_accum.values()) if self.gui_focus_accum else 0.0
        unique_apps = len(self.gui_focus_accum)
        return focus_time, self.gui_switch_count, unique_apps, self.gui_window_event_count

    def _compute_heatmap(self):
        flat = []
        max_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                max_count = max(max_count, self.heat_counts[r][c])
        for r in range(self.rows):
            for c in range(self.cols):
                v = (self.heat_counts[r][c] / max_count) if max_count > 0 else 0.0
                flat.append(v)
        return flat

    def roll_window(self):
        window_end = time.monotonic()
        with self.lock:
            heat = self._compute_heatmap()
            avg_speed, cp_mean, cp_std, l_mean, l_std, r_mean, r_std = self._compute_mouse_features()
            elapsed = window_end - self.window_start
            (npress, k_hold, k_hold_std, k_dd, k_dd_std,
                k_rp, k_rp_std, k_rr, k_cpm) = self._compute_keyboard_features(elapsed)
            gui_focus_time, gui_switches, gui_unique, gui_winevt = self._compute_gui_features(window_end)

            start_rel = self.window_start - self.session_t0
            end_rel = window_end - self.session_t0

            self._reset_for_next_window(window_end)

        return (start_rel, end_rel, heat,
                avg_speed, cp_mean, cp_std, l_mean, l_std, r_mean, r_std,
                npress, k_hold, k_hold_std, k_dd, k_dd_std, k_rp, k_rp_std, k_rr, k_cpm,
                gui_focus_time, gui_switches, gui_unique, gui_winevt)

    def _reset_for_next_window(self, new_start_t):
        self.window_start = new_start_t
        # Mouse
        self.instant_speeds.clear()
        self.heat_counts = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.left_down_times.clear()
        self.right_down_times.clear()
        self.left_holds.clear()
        self.right_holds.clear()
        self.click_times.clear()
        self.pause_times.clear()
        # Keyboard
        self.key_down_times.clear()
        self.key_down_sequence.clear()
        self.key_release_sequence.clear()
        self.key_holds.clear()
        self.key_press_count = 0
        # GUI
        self.gui_switch_count = 0
        self.gui_focus_accum.clear()

def header_names(cols=16, rows=9):
    names = ["window_start_s", "window_end_s"]
    for r in range(rows):
        for c in range(cols):
            names.append(f"mouse_heatmap_r{r}_c{c}")
    names += [
        "mouse_avg_speed","mouse_click_pause_mean","mouse_click_pause_std",
        "mouse_left_hold_mean","mouse_left_hold_std","mouse_right_hold_mean","mouse_right_hold_std",
        "key_press_count","key_avg_hold","key_std_hold","key_avg_dd","key_std_dd",
        "key_avg_rp","key_std_rp","key_avg_rr","key_cpm",
        "gui_focus_time","gui_switch_count","gui_unique_apps","gui_window_event_count"
    ]
    return names

def write_row(csv_path, header, row_tuple, precision=3):
    def fmt(v):
        if isinstance(v, (int,)) and not isinstance(v, bool):
            return str(v)
        try:
            return f"{float(v):.{precision}f}"
        except Exception:
            return "0.000"

    (start_rel, end_rel, heat,
        avg_speed, cp_mean, cp_std, l_mean, l_std, r_mean, r_std,
        npress, k_hold, k_hold_std, k_dd, k_dd_std, k_rp, k_rp_std, k_rr, k_cpm,
        gui_focus_time, gui_switches, gui_unique, gui_winevt) = row_tuple

    values = [start_rel, end_rel] + heat + [
        avg_speed, cp_mean, cp_std, l_mean, l_std, r_mean, r_std,
        npress, k_hold, k_hold_std, k_dd, k_dd_std, k_rp, k_rp_std, k_rr, k_cpm,
        gui_focus_time, gui_switches, gui_unique, gui_winevt
    ]

    int_indices = set()
    for idx, name in enumerate(header):
        if name in ("key_press_count","gui_switch_count","gui_unique_apps","gui_window_event_count"):
            int_indices.add(idx)

    formatted = []
    for idx, v in enumerate(values):
        if idx in int_indices:
            try:
                formatted.append(str(int(v)))
            except Exception:
                formatted.append("0")
        else:
            formatted.append(fmt(v))

    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(formatted)

def mouse_worker(agg: TelemetryAggregator, stop_event: threading.Event):
    with mouse.Listener(
        on_move=lambda x, y: agg.on_move(x, y),
        on_click=lambda x, y, button, pressed: agg.on_click(x, y, button, pressed),
        on_scroll=lambda x, y, dx, dy: agg.on_scroll(x, y, dx, dy)
    ) as listener:
        while not stop_event.is_set():
            time.sleep(0.01)
        listener.stop()

def keyboard_worker(agg: TelemetryAggregator, stop_event: threading.Event):
    with keyboard.Listener(
        on_press=lambda key: agg.on_key_press(key),
        on_release=lambda key: agg.on_key_release(key)
    ) as listener:
        while not stop_event.is_set():
            time.sleep(0.01)
        listener.stop()

def gui_poll_worker(agg: TelemetryAggregator, stop_event: threading.Event, hz=5.0):
    interval = 1.0 / hz
    while not stop_event.is_set():
        agg.poll_gui()
        time.sleep(interval)

def window_roll_worker(agg: TelemetryAggregator, csv_path: str, stop_event: threading.Event, window_seconds: int):
    header = header_names(agg.cols, agg.rows)
    next_roll = time.monotonic() + window_seconds
    while not stop_event.is_set():
        now = time.monotonic()
        if now >= next_roll:
            row = agg.roll_window()
            write_row(csv_path, header, row, precision=3)
            next_roll = now + window_seconds
        time.sleep(0.2)
    row = agg.roll_window()
    write_row(csv_path, header, row, precision=3)

def main():
    parser = argparse.ArgumentParser(description="Windows HCI telemetry logger -> CSV (1 row per window).")
    parser.add_argument("--out", type=str, default="hci_telemetry.csv", help="Output CSV path")
    parser.add_argument("--window-seconds", type=int, default=60, help="Window size in seconds (default: 60)")
    args = parser.parse_args()

    agg = TelemetryAggregator(cols=16, rows=9, window_seconds=args.window_seconds)

    stop_event = threading.Event()

    t_mouse = threading.Thread(target=mouse_worker, args=(agg, stop_event), daemon=True)
    t_key = threading.Thread(target=keyboard_worker, args=(agg, stop_event), daemon=True)
    t_gui = threading.Thread(target=gui_poll_worker, args=(agg, stop_event, 5.0), daemon=True)
    t_roll = threading.Thread(target=window_roll_worker, args=(agg, args.out, stop_event, args.window_seconds), daemon=True)

    t_mouse.start()
    t_key.start()
    t_gui.start()
    t_roll.start()

    print(f"Logging started. Writing rows every {args.window_seconds}s to {args.out}. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        t_mouse.join(timeout=2)
        t_key.join(timeout=2)
        t_gui.join(timeout=2)
        t_roll.join(timeout=2)
        print("Done.")

if __name__ == "__main__":
    main()
