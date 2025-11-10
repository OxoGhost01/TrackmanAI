# server_enum.py
import socket

HOST = "127.0.0.1"
PORT = 5400

class MsgType:
    Msg_None = 0
    Msg_CarState = 1
    Msg_Action = 2
    Msg_Reset = 3

def handle_connection(conn):
    with conn:
        buf = ""
        while True:
            data = conn.recv(4096)
            if not data:
                print("client closed")
                break
            buf += data.decode('utf-8', errors='ignore')
            # process full lines
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if not line.strip():
                    continue
                # parse envelope: mtype|payload
                if "|" not in line:
                    print("bad envelope:", line)
                    continue
                mtype_s, payload = line.split("|", 1)
                mtype = int(mtype_s)
                if mtype == MsgType.Msg_CarState:
                    toks = payload.split(",")
                    # vx, vy, vz, yaw, pitch, roll
                    vx, vy, vz, yaw, pitch, roll = map(float, toks[:6])
                    # --- here you'd run your model to decide actions ---
                    # For now: simple dummy policy
                    steer = 0.0   # centered
                    gas   = 1.0   # full throttle

                    # send action message: 2|steer,gas\n
                    reply = f"{MsgType.Msg_Action}|{steer:.6f},{gas:.6f}\n"
                    conn.sendall(reply.encode('utf-8'))
                # add handling for other message types if needed

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"listening on {HOST}:{PORT}")
    conn, addr = s.accept()
    print("connected by", addr)
    handle_connection(conn)

if __name__ == "__main__":
    main()
