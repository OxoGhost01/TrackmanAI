import socket
import struct
import time

class Msg:
    CGetSimulationState = 1
    CSetInputState = 2
    CIsInMenus = 3
    SIsInMenus = 4

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 5400))
print("Connected to plugin!")

while True:
    # Check if in menus
    sock.sendall(struct.pack("<i", Msg.CIsInMenus))
    msgType = struct.unpack("<i", sock.recv(4))[0]
    if msgType == Msg.SIsInMenus:
        (isMenu,) = struct.unpack("<i", sock.recv(4))
        if isMenu:
            print("In menus — waiting...")
            time.sleep(1)
            continue

    # Request sim state
    sock.sendall(struct.pack("<i", Msg.CGetSimulationState))
    (length,) = struct.unpack("<i", sock.recv(4))
    data = sock.recv(length)
    print(f"Received {len(data)} bytes")

    # Send dummy input (gas only)
    payload = struct.pack("<i4B", Msg.CSetInputState, 0, 0, 1, 0)
    sock.sendall(payload)

    time.sleep(0.05)
