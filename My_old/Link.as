enum MessageType {
    CGetSimulationState = 1,
    CSetInputState      = 2,
    CIsInMenus          = 3,
}

Net::Socket@ sock;
Net::Socket@ clientSock;
bool debug = true;

void Main() {
    @sock = Net::Socket();
    bool ok = sock.Listen("127.0.0.1", 5400);
    if (debug) print("DRLBridge listening on 5400 ok=" + ok);
}

void OnRunStep(SimulationManager@ sim) {
    if (clientSock is null) {
        @clientSock = sock.Accept();
        if (clientSock !is null && debug)
            print("DRLBridge: client connected!");
        return;
    }

    if (clientSock is null) return;

    uint avail = clientSock.get_Available();
    if (avail < 4) return;

    int msgType = clientSock.ReadInt32();

    switch (msgType) {
        case MessageType::CGetSimulationState: {
            auto@ state = sim.SaveState();
            if (state !is null) {
                const array<uint8>@ data = state.ToArray();
                clientSock.Write(int(data.Length));
                clientSock.Write(data);
            } else {
                clientSock.Write(0);
            }
            break;
        }

        case MessageType::CSetInputState: {
            double steer = float(clientSock.ReadDouble());
            bool accelerate = clientSock.ReadUint8() > 0;
            bool brake = clientSock.ReadUint8() > 0;

            if (sim.InRace) {
                int steerInt = int(steer * 65536.0);
                sim.SetInputState(InputType::Steer, steerInt);
                sim.SetInputState(InputType::Up, accelerate ? 1 : 0);
                sim.SetInputState(InputType::Down, brake ? 1 : 0);
            }
            break;
        }

        case MessageType::CIsInMenus: {
            int isInMenus = (GetCurrentGameState() == TM::GameState::Menus ? 1 : 0);
            clientSock.Write(4);
            clientSock.Write(isInMenus);
            break;
        }
    }
}

PluginInfo@ GetPluginInfo() {
    PluginInfo info;
    info.Name = "Link";
    info.Author = "OxoGhost";
    info.Description = "RL bridge using TMI 2.1.x with binary protocol";
    info.Version = "0.91";
    return info;
}
