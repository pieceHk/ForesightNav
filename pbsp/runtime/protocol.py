"""Newline-delimited JSON mapping protocol, shared by workers and clients."""

import base64
import json
import os
from pathlib import Path
import socket
import stat
import sys
import threading
import traceback


PROTOCOL_VERSION = 1


def encode_message(message, *, versioned=True):
    payload = dict(message)
    if versioned:
        payload["protocol_version"] = PROTOCOL_VERSION
    return json.dumps(payload, separators=(",", ":")).encode() + b"\n"


def decode_message(line):
    payload = json.loads(line.decode())
    if not isinstance(payload, dict):
        raise ValueError("Protocol message must be a JSON object")
    version = payload.get("protocol_version", PROTOCOL_VERSION)
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported mapping protocol version: {version!r}")
    return payload


def receive_line(connection, buffer=b""):
    while b"\n" not in buffer:
        chunk = connection.recv(65536)
        if not chunk:
            raise RuntimeError("ES Mapper Server closed connection")
        buffer += chunk
    return buffer.split(b"\n", 1)


def send_message(connection, message, *, versioned=True):
    connection.sendall(encode_message(message, versioned=versioned))


def request(socket_path, message, *, timeout_s=120.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_s)
        connection.connect(str(socket_path))
        send_message(connection, message)
        line, _ = receive_line(connection)
    return decode_message(line)


def decode_binary_payload(payload):
    if isinstance(payload, str):
        return base64.b64decode(payload.encode("ascii"), validate=True)
    if isinstance(payload, (list, tuple)):
        return bytes(payload)
    return payload


def handle_connection(connection, server, *, on_shutdown=None):
    stream = None
    try:
        stream = connection.makefile("rb")
        while True:
            line = stream.readline()
            if not line:
                break
            command = ""
            try:
                message = decode_message(line)
                command = message.get("cmd", "")
                arguments = {key: value for key, value in message.items() if key != "cmd"}
                if command == "reset_episode":
                    response = server.handle_reset_episode(**message)
                elif command == "push_frame":
                    response = server.handle_push_frame(**arguments)
                elif command == "push_live_frame":
                    response = server.handle_push_live_frame(**arguments)
                elif command == "query":
                    response = server.handle_query(**arguments)
                elif command == "batch_query":
                    response = server.handle_batch_query(**arguments)
                elif command == "status":
                    response = server.handle_status()
                elif command == "flush":
                    response = server.handle_flush()
                elif command == "shutdown":
                    try:
                        server.handle_flush()
                    except Exception as exc:
                        print(f"[ESMapperServer] shutdown flush failed: {exc}", flush=True)
                    send_message(connection, {"ok": True, "message": "shutting down"})
                    if on_shutdown is not None:
                        on_shutdown()
                    return
                else:
                    response = {"ok": False, "message": f"Unknown command: {command}"}
                send_message(connection, response)
                del message, response, arguments, line
            except Exception as exc:
                print(f"[ESMapperServer] cmd={command!r} ERROR: {exc}", flush=True)
                traceback.print_exc(file=sys.stdout)
                send_message(
                    connection,
                    {"ok": False, "message": str(exc), "traceback": traceback.format_exc()},
                )
    finally:
        if stream is not None:
            stream.close()
        connection.close()


def serve(server, socket_path, *, stop_event=None, on_ready=None):
    """Own the listener and client threads; the worker owns the Mapper."""
    path = Path(socket_path)
    if path.exists():
        if not stat.S_ISSOCK(path.stat().st_mode):
            raise FileExistsError(f"Socket path is not a socket: {path}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            try:
                probe.connect(str(path))
            except ConnectionRefusedError:
                path.unlink()
            else:
                raise RuntimeError(f"Mapping socket already has a listener: {path}")
    stopped = stop_event if stop_event is not None else threading.Event()
    clients, threads = set(), set()
    lock = threading.Lock()

    def handle(client):
        try:
            handle_connection(client, server, on_shutdown=stopped.set)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with lock:
                clients.discard(client)
                threads.discard(threading.current_thread())

    inode = None
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        inode = path.stat().st_ino
        os.chmod(path, 0o600)
        listener.listen(5)
        listener.settimeout(0.2)
        print(f"[ESMapperServer] Listening on {path}", flush=True)
        if on_ready is not None:
            on_ready()
        while not stopped.is_set():
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            thread = threading.Thread(target=handle, args=(client,), daemon=True)
            with lock:
                clients.add(client)
                threads.add(thread)
            thread.start()
    finally:
        stopped.set()
        listener.close()
        with lock:
            active_clients, active_threads = tuple(clients), tuple(threads)
        for client in active_clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for thread in active_threads:
            thread.join()
        if inode is not None and path.exists() and path.stat().st_ino == inode:
            path.unlink()
