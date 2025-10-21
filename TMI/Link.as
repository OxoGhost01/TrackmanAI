enum MessageType {
    CGetSimulationState = 1,
    CSetInputState      = 2,
    CIsInMenus          = 3,
    SIsInMenus          = 4
}

Net::Socket@ sock;
Net::Socket@ clientSock;

// debug toggle
bool debug = true;

void Main() {
    @sock = Net::Socket();

    // start listening, same as Linesight
    bool ok = sock.Listen("127.0.0.1", 5400);
    if (debug) print("DRLBridge listening on 5400 ok=" + ok);
}

// OnRunStep is called every frame/tick for a live run
void OnRunStep(SimulationManager@ sim) {
    // 1) Accept connection if we don't have a client yet
    if (clientSock is null) {
        @clientSock = sock.Accept();      // returns null if none waiting
        if (clientSock !is null) {
            if (debug) print("DRLBridge: client connected!");
            return; // wait next tick to start handling messages
        }
    }

    if (clientSock is null) return; // still no client

    // 2) If the client has data, handle it.
    uint avail = clientSock.get_Available(); // use get_Available()
    if (avail == 0) return;

    // NOTE: ReadInt/ReadUint8 are blocking only if not enough bytes are available.
    // We checked get_Available() > 0; we still must be careful when reading multiple fields.
    int msgType = clientSock.ReadInt32(); // use ReadInt() in this TMI runtime

    switch (msgType) {
        case MessageType::CGetSimulationState: {
            auto@ state = sim.SaveState();
            if (state !is null) {
                const array<uint8>@ data = state.ToArray();
                // send length (int) then raw bytes
                clientSock.Write(int(data.Length));
                clientSock.Write(data);
            } else {
                // send zero length to indicate no state
                clientSock.Write(0);
            }
            break;
        }

        case MessageType::CSetInputState: {
            // read four uint8 booleans (0/1)
            bool left       = clientSock.ReadUint8() > 0;
            bool right      = clientSock.ReadUint8() > 0;
            bool accelerate = clientSock.ReadUint8() > 0;
            bool brake      = clientSock.ReadUint8() > 0;

            if (sim.InRace) {
                sim.SetInputState(InputType::Left,  left ? 1 : 0);
                sim.SetInputState(InputType::Right, right ? 1 : 0);
                sim.SetInputState(InputType::Up,    accelerate ? 1 : 0);
                sim.SetInputState(InputType::Down,  brake ? 1 : 0);
            }
            break;
        }

        case MessageType::CIsInMenus: {
            int isInMenus = (GetCurrentGameState() == TM::GameState::Menus ? 1 : 0);
            // reply with SIsInMenus code then the int value
            clientSock.Write(int(MessageType::SIsInMenus));
            clientSock.Write(isInMenus);
            break;
        }

        default: {
            if (debug) print("DRLBridge: unknown msgType " + msgType);
            break;
        }
    }
}

PluginInfo@ GetPluginInfo() {
    PluginInfo info;
    info.Name = "DRLBridge";
    info.Author = "you";
    info.Description = "RL bridge using TMInterface 2.1.x binary socket API";
    info.Version = "0.7"; // string + semicolon
    return info;
}
