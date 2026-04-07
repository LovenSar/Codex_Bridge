#!/usr/bin/env python3
"""
Codex Chat Bridge - Entry point
This file redirects to bridge/chat_bridge.py
"""
import sys
import os

# Add bridge directory to path
bridge_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bridge")
sys.path.insert(0, bridge_dir)

# Import and run the main from chat_bridge
from chat_bridge import main

if __name__ == "__main__":
    main()