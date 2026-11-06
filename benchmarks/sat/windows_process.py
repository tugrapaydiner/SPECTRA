"""Windows job supervision for bounded development campaigns; standard library only."""
from __future__ import annotations

import ctypes as c
from ctypes import wintypes as w
import msvcrt
import os
from pathlib import Path
import subprocess
import threading
import time


class _JobLimits(c.Structure):
    _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                ("flags", w.DWORD), ("min_working_set", c.c_size_t),
                ("max_working_set", c.c_size_t), ("active_processes", w.DWORD),
                ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class _ExtendedLimits(c.Structure):
    _fields_ = [("basic", _JobLimits), ("io", c.c_uint64 * 6),
                ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                ("peak_process_memory", c.c_size_t), ("peak_job_memory", c.c_size_t)]


class _Accounting(c.Structure):
    _fields_ = [("user", c.c_int64), ("kernel", c.c_int64),
                ("period_user", c.c_int64), ("period_kernel", c.c_int64),
                ("faults", w.DWORD), ("total", w.DWORD),
                ("active", w.DWORD), ("terminated", w.DWORD)]


class _Memory(c.Structure):
    _fields_ = [("size", w.DWORD), ("faults", w.DWORD),
                ("peak_working_set", c.c_size_t), ("working_set", c.c_size_t),
                ("peak_paged", c.c_size_t), ("paged", c.c_size_t),
                ("peak_nonpaged", c.c_size_t), ("nonpaged", c.c_size_t),
                ("pagefile", c.c_size_t), ("peak_pagefile", c.c_size_t)]


class _Startup(c.Structure):
    _fields_ = [("size", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR),
                ("title", w.LPWSTR), ("x", w.DWORD), ("y", w.DWORD),
                ("x_size", w.DWORD), ("y_size", w.DWORD), ("x_chars", w.DWORD),
                ("y_chars", w.DWORD), ("fill", w.DWORD), ("flags", w.DWORD),
                ("show", w.WORD), ("reserved_size", w.WORD), ("reserved_bytes", c.c_void_p),
                ("stdin", w.HANDLE), ("stdout", w.HANDLE), ("stderr", w.HANDLE)]


class _StartupEx(c.Structure):
    _fields_ = [("startup", _Startup), ("attributes", c.c_void_p)]


class _Process(c.Structure):
    _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE),
                ("pid", w.DWORD), ("tid", w.DWORD)]


_kernel = c.WinDLL("kernel32", use_last_error=True)


def _api(name, restype, *argtypes):
    function = getattr(_kernel, name)
    function.restype, function.argtypes = restype, argtypes
    return function


_create_job = _api("CreateJobObjectW", w.HANDLE, c.c_void_p, w.LPCWSTR)
_set_job = _api("SetInformationJobObject", w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD)
_query_job = _api("QueryInformationJobObject", w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p)
_assign_job = _api("AssignProcessToJobObject", w.BOOL, w.HANDLE, w.HANDLE)
_terminate_job = _api("TerminateJobObject", w.BOOL, w.HANDLE, w.UINT)
_close = _api("CloseHandle", w.BOOL, w.HANDLE)
_init_attributes = _api("InitializeProcThreadAttributeList", w.BOOL, c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t))
_update_attributes = _api("UpdateProcThreadAttribute", w.BOOL, c.c_void_p, w.DWORD, c.c_size_t,
                          c.c_void_p, c.c_size_t, c.c_void_p, c.c_void_p)
_delete_attributes = _api("DeleteProcThreadAttributeList", None, c.c_void_p)
_create_process = _api("CreateProcessW", w.BOOL, w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p,
                       w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR, c.POINTER(_StartupEx), c.POINTER(_Process))
_resume = _api("ResumeThread", w.DWORD, w.HANDLE)
_wait = _api("WaitForSingleObject", w.DWORD, w.HANDLE, w.DWORD)
_exit_code = _api("GetExitCodeProcess", w.BOOL, w.HANDLE, c.POINTER(w.DWORD))
_terminate = _api("TerminateProcess", w.BOOL, w.HANDLE, w.UINT)
_memory = _api("K32GetProcessMemoryInfo", w.BOOL, w.HANDLE, c.POINTER(_Memory), w.DWORD)
_KILLED = 0xFFFFFFF7


def _checked(result):
    if not result:
        raise c.WinError(c.get_last_error())
    return result


def run_windows(command: list[str], destination: Path, workdir: Path, environment: dict,
                timeout: float, memory_mib: int, output_mib: int,
                output_directory: Path | None = None) -> dict:
    """Assign a suspended process to a memory-limited job before any code runs.

    Logs are hard capped. Other newly written files in the working directory are
    polled every 10 ms; their size can overshoot between polls. This is not a disk
    quota or a security sandbox. Job memory is committed memory, not RSS.
    """
    record = {"timed_out": False, "output_limit_exceeded": False,
              "resource_scope": "job CPU and committed memory; root-process peak working set",
              "peak_job_memory_kib": None}
    cap = output_mib << 20
    watch = destination if workdir == Path(__file__).resolve().parents[2] else workdir
    if output_directory is not None:
        watch = output_directory.resolve(strict=True)
    initial = {p: p.stat().st_size for p in watch.rglob("*") if p.is_file()}
    overflow = threading.Event()
    errors = []
    threads, descriptors = [], set()
    process, attributes, job = _Process(), None, None
    started = time.perf_counter()

    def copy_stream(fd, path):
        try:
            with os.fdopen(fd, "rb", buffering=0) as incoming, path.open("xb") as outgoing:
                remaining = cap
                while chunk := incoming.read(64 * 1024):
                    outgoing.write(chunk[:remaining])
                    if len(chunk) > remaining:
                        overflow.set()
                    remaining = max(0, remaining - len(chunk))
        except OSError as error:
            errors.append(error)
            overflow.set()

    def check_files():
        for path in watch.rglob("*"):
            try:
                size = path.stat().st_size if path.is_file() else 0
            except FileNotFoundError:
                continue
            if size > cap and size > initial.get(path, 0):
                overflow.set()
                return

    try:
        job = _checked(_create_job(None, None))
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
        limits.job_memory = memory_mib << 20
        _checked(_set_job(job, 9, c.byref(limits), c.sizeof(limits)))
        handles = []
        stdin = os.open(os.devnull, os.O_RDONLY)
        descriptors.add(stdin)
        handles.append(msvcrt.get_osfhandle(stdin))
        for name in ("stdout", "stderr"):
            read_fd, write_fd = os.pipe()
            descriptors.update((read_fd, write_fd))
            handles.append(msvcrt.get_osfhandle(write_fd))
            thread = threading.Thread(target=copy_stream, args=(read_fd, destination / (name + ".txt")), daemon=True)
            thread.start()
            descriptors.remove(read_fd)  # The reader owns and closes this end.
            threads.append(thread)
        for handle in handles:
            os.set_handle_inheritable(handle, True)
        size = c.c_size_t()
        _init_attributes(None, 1, 0, c.byref(size))
        storage = c.create_string_buffer(size.value)
        _checked(_init_attributes(storage, 1, 0, c.byref(size)))
        attributes = storage
        inherited = (w.HANDLE * 3)(*handles)
        _checked(_update_attributes(attributes, 0, 0x20002, inherited, c.sizeof(inherited), None, None))
        startup = _StartupEx()
        startup.startup.size = c.sizeof(startup)
        startup.startup.flags = 0x100  # STARTF_USESTDHANDLES
        startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = handles
        startup.attributes = c.cast(attributes, c.c_void_p)
        env = c.create_unicode_buffer("\0".join(f"{k}={v}" for k, v in sorted(environment.items(), key=lambda item: item[0].upper())) + "\0")
        argv = c.create_unicode_buffer(subprocess.list2cmdline(command))
        # Suspended, hidden, Unicode environment, and an explicit inherited-handle list.
        _checked(_create_process(None, argv, None, None, True, 0x08080404, env,
                                 str(workdir), c.byref(startup), c.byref(process)))
        _checked(_assign_job(job, process.process))
        if _resume(process.thread) == 0xFFFFFFFF:
            raise c.WinError(c.get_last_error())
        for fd in descriptors:
            os.close(fd)
        descriptors.clear()
        while True:
            check_files()
            if overflow.is_set():
                break
            state = _wait(process.process, 10)
            if state == 0:
                break
            if state == 0xFFFFFFFF:
                raise c.WinError(c.get_last_error())
            if time.perf_counter() - started >= timeout:
                record["timed_out"] = True
                break
        # Includes descendants even when the root exited successfully.
        _checked(_terminate_job(job, _KILLED))
        _wait(process.process, 0xFFFFFFFF)
        for thread in threads:
            thread.join()
        check_files()
        record["output_limit_exceeded"] = overflow.is_set()
        if errors:
            raise errors[0]
        code = w.DWORD()
        _checked(_exit_code(process.process, c.byref(code)))
        record["returncode"] = c.c_int32(code.value).value
        memory = _Memory()
        memory.size = c.sizeof(memory)
        _checked(_memory(process.process, c.byref(memory), c.sizeof(memory)))
        record["peak_rss_kib"] = memory.peak_working_set / 1024
        _checked(_query_job(job, 9, c.byref(limits), c.sizeof(limits), None))
        record["peak_job_memory_kib"] = limits.peak_job_memory / 1024
        accounting = _Accounting()
        cleanup_deadline = time.perf_counter() + 5
        while True:
            _checked(_query_job(job, 1, c.byref(accounting), c.sizeof(accounting), None))
            if accounting.active == 0:
                break
            if time.perf_counter() >= cleanup_deadline:
                raise OSError("Windows job did not finish terminating")
            time.sleep(0.001)
        record.update(user_seconds=accounting.user / 10_000_000,
                      system_seconds=accounting.kernel / 10_000_000)
    except OSError as error:
        record["spawn_error"] = f"{type(error).__name__}: {error}"
    finally:
        if process.process:
            _terminate(process.process, _KILLED)
        if job:
            _close(job)  # Also closes the job on assignment or supervision failure.
        for fd in descriptors:
            os.close(fd)
        for thread in threads:
            thread.join()
        if attributes is not None:
            _delete_attributes(attributes)
        for handle in (process.thread, process.process):
            if handle:
                _close(handle)
        for name in ("stdout.txt", "stderr.txt"):
            (destination / name).touch(exist_ok=True)
    return record
