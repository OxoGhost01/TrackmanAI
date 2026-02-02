#!/bin/bash
# Launch script using single shared 32-bit Wine prefix for all ports
# Simpler than port-specific prefixes

PORT=${1:-8478}

# ============================================
# CONFIGURATION
# ============================================

# Source from Lutris
SOURCE_PREFIX="$HOME/Games/trackmania-nations-forever"
SOURCE_TM_DIR="$SOURCE_PREFIX/drive_c/Program Files (x86)/TmNationsForever"

# Shared 32-bit prefix for TrackMania
WINE_PREFIX="$HOME/.wine_trackmania_32bit"
TM_DIR="$WINE_PREFIX/drive_c/Program Files (x86)/TmNationsForever"

# ============================================
# ONE-TIME SETUP
# ============================================

if [ ! -d "$WINE_PREFIX" ]; then
    echo "=========================================="
    echo "FIRST-TIME SETUP: Creating 32-bit prefix"
    echo "=========================================="
    echo "This will take a minute..."
    
    # Create 32-bit prefix
    export WINEPREFIX="$WINE_PREFIX"
    export WINEARCH=win32
    export WINEDEBUG=-all
    
    wineboot -u
    sleep 3
    
    # Copy game files
    mkdir -p "$(dirname "$TM_DIR")"
    
    if [ -d "$SOURCE_TM_DIR" ]; then
        echo "Copying game files from Lutris..."
        cp -r "$SOURCE_TM_DIR" "$(dirname "$TM_DIR")/"
        echo "✓ Game files copied"
    else
        echo "ERROR: Lutris game not found at: $SOURCE_TM_DIR"
        exit 1
    fi
    
    # Copy TMInterface
    if [ -f "$HOME/.local/share/TMInterface/TMInterface.dll" ]; then
        cp "$HOME/.local/share/TMInterface/TMInterface.dll" "$TM_DIR/"
        cp "$HOME/.local/share/TMInterface"/*.dll "$TM_DIR/" 2>/dev/null
        echo "✓ TMInterface copied"
    else
        echo "WARNING: TMInterface.dll not found!"
        echo "Download from: https://github.com/donadigo/TMInterface/releases"
        echo "Extract to: ~/.local/share/TMInterface/"
    fi
    
    echo "=========================================="
    echo "Setup complete!"
    echo "=========================================="
fi

# ============================================
# LAUNCH GAME
# ============================================

export WINEPREFIX="$WINE_PREFIX"
export WINEARCH=win32
export TMI_PORT=$PORT
export WINE_DISABLE_UDEV=1
export WINEDEBUG=-all

if [ ! -f "$TM_DIR/TmForever.exe" ]; then
    echo "ERROR: Game not installed in 32-bit prefix"
    echo "Expected: $TM_DIR/TmForever.exe"
    exit 1
fi

cd "$TM_DIR"

# Kill existing instance on this port
pkill -f "custom_port $PORT" 2>/dev/null

# Launch
nohup wine TmForever.exe /configstring="set custom_port $PORT" \
    > /tmp/tm_port_${PORT}.log 2>&1 &

echo "Game launched on port $PORT"

# Verify
sleep 2
if ! pgrep -f "TmForever.exe" > /dev/null; then
    echo "ERROR: Game crashed! Log:"
    tail -30 /tmp/tm_port_${PORT}.log
    exit 1
fi

echo "✓ Game running"