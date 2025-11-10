import socket
import struct
import time

class Msg:
    CGetSimulationState = 1
    CSetInputState = 2
    CIsInMenus = 3

def _read_exact(sock, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data

def _read_int32(sock) -> int:
    return struct.unpack("<i", _read_exact(sock, 4))[0]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 5400))
print("Connected to plugin!")

while True:
    # Ask if we're in menus
    sock.sendall(struct.pack("<i", Msg.CIsInMenus))
    length = _read_int32(sock)
    is_menu = struct.unpack("<i", _read_exact(sock, length))[0]
    if is_menu:
        print("In menus — waiting...")
        time.sleep(1)
        continue

    # Request simulation state
    sock.sendall(struct.pack("<i", Msg.CGetSimulationState))
    length = _read_int32(sock)
    if length > 0:
        sim_data = _read_exact(sock, length)

    # Dummy input: steer = 0.0, gas = 1, brake = 0
    steer = 0.0
    accelerate = 1
    brake = 0
    msg = struct.pack("<id2B", Msg.CSetInputState, steer, accelerate, brake)
    sock.sendall(msg)

    time.sleep(0.05)
