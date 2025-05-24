#!/bin/bash

# Script to upload episode data to Vast.ai instance
# Usage: 
#   ./sync_episodes.sh [IP_ADDRESS] [PORT]
#   OR
#   ./sync_episodes.sh "ssh -p PORT root@IP_ADDRESS -L 8080:localhost:8080"

# Function to parse SSH command
parse_ssh_command() {
    local ssh_cmd="$1"
    # Extract port using regex: -p followed by digits
    PORT=$(echo "$ssh_cmd" | grep -oE '\-p [0-9]+' | grep -oE '[0-9]+')
    # Extract IP using regex: root@ followed by IP address
    IP_ADDRESS=$(echo "$ssh_cmd" | grep -oE 'root@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | cut -d'@' -f2)
    
    if [[ -z "$PORT" || -z "$IP_ADDRESS" ]]; then
        echo "Error: Could not parse SSH command. Expected format:"
        echo "ssh -p PORT root@IP_ADDRESS -L 8080:localhost:8080"
        return 1
    fi
    
    echo "Parsed from SSH command:"
    echo "  IP: $IP_ADDRESS"
    echo "  Port: $PORT"
    return 0
}

# Parse arguments
if [ $# -eq 1 ]; then
    # Single argument - assume it's an SSH command
    if ! parse_ssh_command "$1"; then
        exit 1
    fi
elif [ $# -eq 2 ]; then
    # Two arguments - assume IP and PORT
    IP_ADDRESS=$1
    PORT=$2
else
    echo "Usage: $0 <IP_ADDRESS> <PORT>"
    echo "   OR: $0 \"ssh -p PORT root@IP_ADDRESS -L 8080:localhost:8080\""
    echo "Example 1: $0 116.109.215.47 50472"
    echo "Example 2: $0 \"ssh -p 42614 root@188.243.117.214 -L 8080:localhost:8080\""
    exit 1
fi

echo "Uploading episode data to Vast.ai instance..."
echo "Target: root@$IP_ADDRESS:$PORT"
echo "This may take a while (1.1GB of data)..."

# Sync the episode data with compression
rsync -avz --progress -e "ssh -p $PORT" \
    ./mycobot_episodes/ root@$IP_ADDRESS:/workspace/FedVLA_Policy/mycobot_episodes/

echo "Episode data upload completed!" 