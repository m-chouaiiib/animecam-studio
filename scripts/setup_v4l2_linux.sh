#!/usr/bin/env bash
# ==============================================================================
# AnimeCam Studio - Safe v4l2loopback Virtual Camera Setup for Linux
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " AnimeCam Studio - v4l2loopback Virtual Camera Installer"
echo "=========================================================="

if [[ $EUID -ne 0 ]]; then
   echo "[-] Error: This script must be run as root (use sudo)." 1>&2
   exit 1
fi

echo "[+] Updating apt repositories..."
apt-get update -qq

echo "[+] Installing v4l2loopback-dkms and utils..."
apt-get install -y -qq v4l2loopback-dkms v4l2loopback-utils

# Unload existing module if present
if lsmod | grep -q "^v4l2loopback"; then
    echo "[+] Unloading previous v4l2loopback instance..."
    modprobe -r v4l2loopback || true
fi

echo "[+] Loading v4l2loopback module with label 'AnimeCam Virtual Camera'..."
modprobe v4l2loopback devices=1 video_nr=10 card_label="AnimeCam Virtual Camera" exclusive_caps=1

# Persist configuration
cat <<EOF > /etc/modprobe.d/animecam-v4l2loopback.conf
# Managed by AnimeCam Studio
options v4l2loopback devices=1 video_nr=10 card_label="AnimeCam Virtual Camera" exclusive_caps=1
EOF

if ! grep -q "v4l2loopback" /etc/modules 2>/dev/null; then
    echo "v4l2loopback" >> /etc/modules
fi

echo "[✓] Setup complete! /dev/video10 is ready as 'AnimeCam Virtual Camera'."
echo "[✓] Video conferencing apps (Zoom, Discord, Meet) can now select AnimeCam."
