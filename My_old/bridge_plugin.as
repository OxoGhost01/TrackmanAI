// Documents/TMInterface/Plugins/RLBridge/main.as
// Corrected: use SimulationManager@ in OnRunStep signature (not RunSimulationManager)

Net::Socket@ sock;

enum MsgType {
    Msg_None = 0,
    Msg_CarState = 1,
    Msg_Action = 2,
    Msg_Reset = 3
}

PluginInfo@ GetPluginInfo() {
    PluginInfo info;
    info.Name = "RLBridge";
    info.Author = "you";
    info.Description = "Bridge using OnRunStep and enum protocol (Steer+Gas)";
    return info;
}

// Utility: remove CR/LF trailing characters by cutting at first occurrence
string StripNewline(const string &in s) {
    int idx = s.FindFirstOf("\r\n");
    if (idx >= 0) return s.Substr(0, idx);
    return s;
}

void Main() {
    @sock = Net::Socket();
    print("RLBridge: loaded — will attempt connect during OnRunStep.");
}

// OnRunStep used for live runs. Use SimulationManager@ (correct type).
void OnRunStep(SimulationManager@ simManager, bool userCancelled) {
    if (simManager is null) return;

    // Attempt (fast, non-blocking) connect each frame until successful.
    if (sock.get_RemoteIP() == "") {
        if (sock.Connect("127.0.0.1", 5400, 0)) {
            sock.set_NoDelay(true);
            print("RLBridge: connected to Python server");
        } else {
            // Not connected yet - skip this frame to avoid blocking.
            return;
        }
    }

    // Obtain vehicle object for the main car in the run
    TM::SceneVehicleCar@ car = simManager.get_SceneVehicleCar();
    if (car is null) return;

    // Telemetry: local speed vector (vx,vy,vz)
    float vx = car.CurrentLocalSpeed.x;
    float vy = car.CurrentLocalSpeed.y;
    float vz = car.CurrentLocalSpeed.z;

    // Try to get orientation (yaw, pitch, roll) from dyna state safely
    float yaw = 0.0, pitch = 0.0, roll = 0.0;
    if (simManager.get_Dyna() !is null) {
        auto st = simManager.get_Dyna().GetCurrentState();
        if (st !is null) {
            st.Location.Rotation.GetYawPitchRoll(yaw, pitch, roll);
        }
    }

    // Build message: "1|vx,vy,vz,yaw,pitch,roll\n"
    string payload = Text::FormatFloat(vx, "", 0, 6) + "," +
                    Text::FormatFloat(vy, "", 0, 6) + "," +
                    Text::FormatFloat(vz, "", 0, 6) + "," +
                    Text::FormatFloat(yaw, "", 0, 6) + "," +
                    Text::FormatFloat(pitch, "", 0, 6) + "," +
                    Text::FormatFloat(roll, "", 0, 6);

    string out = Text::FormatInt(MsgType::Msg_CarState) + "|" + payload + "\n";
    sock.Write(out);

    // Non-blocking read of any reply bytes
    uint avail = sock.get_Available();
    if (avail == 0) return;

    string resp = sock.ReadString(avail);
    if (resp == "") return;

    // Process each newline-separated line; pick first non-empty valid envelope
    array<string>@ lines = resp.Split("\n");
    for (uint i = 0; i < lines.get_Length(); ++i) {
        string line = StripNewline(lines[i]);
        if (line == "") continue;

        // Envelope: "mtype|payload"
        array<string>@ parts = line.Split("|", 2);
        if (parts.get_Length() < 2) continue;

        int mtype = int(Text::ParseInt(parts[0]));
        string pay = parts[1];

        if (mtype == MsgType::Msg_Action) {
            // Expect "steer,gas" where steer/gas are floats in [-1..1]
            array<string>@ toks = pay.Split(",");
            if (toks.get_Length() >= 2) {
                float steer = Text::ParseFloat(toks[0]);
                float gas   = Text::ParseFloat(toks[1]);

                // Convert to TMInterface analog int range [-65536, 65536]
                int steer_i = int(steer * 65536.0);
                int gas_i   = int(gas   * 65536.0);

                // Clamp manually
                if (steer_i < -65536) steer_i = -65536;
                if (steer_i >  65536) steer_i =  65536;
                if (gas_i   < -65536) gas_i   = -65536;
                if (gas_i   >  65536) gas_i   =  65536;

                // Apply the inputs via SimulationManager::SetInputState
                simManager.SetInputState(InputType::Steer, steer_i);
                simManager.SetInputState(InputType::Gas,   gas_i);

                // If you prefer a digital brake, accept a third token:
                // if (toks.get_Length() >= 3) {
                //     int brake_flag = int(Text::ParseInt(toks[2])); // 0 or 1
                //     simManager.SetInputState(InputType::Down, brake_flag);
                // }
            }
        }
    }
}
