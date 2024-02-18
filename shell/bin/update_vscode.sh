#!/bin/bash
# Script to update VScode

#grab the current commit id
COMMIT_ID=`ls -At  ~/.vscode-server/bin | head -n 1`
cd ~/.vscode-server/bin/$COMMIT_ID

#unlock it
rm ./vscode-remote-lock*

#download package and unpack it
wget https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-x64/stable
tar -xvzf ./stable --strip-components 1

if [ $? -eq 0 ]; then
    rm ./stable
    echo "Successfully updated VScode"
else
  echo "Failed to download and untar VScode update"
fi
