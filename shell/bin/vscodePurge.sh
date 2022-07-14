#!/bin/bash

USER=ykliu

ps aux | grep -E "vscode.*$USER" | awk '{print $2}' | xargs kill -9
rm -rf /home/$USER/.vscode-server/bin/*

echo "Finish purging .vscode-server processes"
