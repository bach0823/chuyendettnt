#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
GIMP Guard - Chặn tính năng Color Picker khi ấn giữ phím Ctrl trong GIMP:
========================================================================
1. Lắng nghe sự kiện chuột cấp thấp (WH_MOUSE_LL) trên Windows.
2. Khi cửa sổ GIMP đang Active và người dùng Click chuột trái (vẽ nét):
   Nếu phím Ctrl đang bị đè -> Tạm thời ngắt cờ Ctrl để GIMP nhận cú click vẽ bình thường.
3. Không ảnh hưởng đến các phím tắt bàn phím: Ctrl+Z, Ctrl+S, Ctrl+E vẫn hoạt động 100%.
"""

import sys
import time
import ctypes
from ctypes import wintypes
import psutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WH_MOUSE_LL = 14
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
VK_CONTROL = 0x11
KEYEVENTF_KEYUP = 0x0002

HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", wintypes.POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulonglong)
    ]

hook_handle = None
was_ctrl_released_for_click = False


def is_gimp_active() -> bool:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return False
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return False
    try:
        proc = psutil.Process(pid.value)
        return "gimp" in proc.name().lower()
    except Exception:
        return False


def low_level_mouse_proc(nCode, wParam, lParam):
    global was_ctrl_released_for_click

    if nCode >= 0:
        if wParam == WM_LBUTTONDOWN:
            if is_gimp_active():
                # Kiểm tra phím Ctrl có đang bị đè không
                is_ctrl_down = bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)
                if is_ctrl_down:
                    # Gửi tín hiệu nhả Ctrl ngay trước khi GIMP xử lý click chuột
                    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
                    was_ctrl_released_for_click = True
                    # In thông báo log nhẹ
                    print("[GIMP Guard] Đã chặn Color Picker: Chuyển cú click thành nét vẽ Pencil bình thường.")
                    sys.stdout.flush()

        elif wParam == WM_LBUTTONUP:
            if was_ctrl_released_for_click:
                was_ctrl_released_for_click = False
                # Nếu phím Ctrl vật lý trên bàn phím vẫn đang bị giữ, khôi phục lại trạng thái đè
                if bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000):
                    user32.keybd_event(VK_CONTROL, 0, 0, 0)

    return user32.CallNextHookEx(hook_handle, nCode, wParam, lParam)


def main():
    global hook_handle

    print("=" * 65)
    print("=== GIMP GUARD - BẢO VỆ CHỐNG COLOR PICKER KHI VẼ BẰNG PEN ===")
    print("=" * 65)
    print("Trạng thái: Đang chạy ngầm và theo dõi cửa sổ GIMP...")
    print("- Bấm Ctrl+C trong cửa sổ này để tắt Guard.")
    print("=" * 65)
    sys.stdout.flush()

    callback = HOOKPROC(low_level_mouse_proc)
    hook_handle = user32.SetWindowsHookExW(
        WH_MOUSE_LL,
        callback,
        kernel32.GetModuleHandleW(None),
        0
    )

    if not hook_handle:
        print("[LỖI] Không thể cài đặt Windows Hook:", ctypes.GetLastError())
        sys.exit(1)

    try:
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    except KeyboardInterrupt:
        pass
    finally:
        if hook_handle:
            user32.UnhookWindowsHookEx(hook_handle)
            print("\n[GIMP Guard] Đã gỡ bỏ Hook thành công. Tạm biệt!")


if __name__ == "__main__":
    main()
