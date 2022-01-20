#!/usr/bin/bash

SCRIPTS_DIR=$(dirname ${BASH_SOURCE[0]})
DGL=$(realpath "$(find ${SCRIPTS_DIR}/../dgl/python -type d -name dgl*.egg)")
GNS_DGL=$(realpath "$(find ${SCRIPTS_DIR}/../dgl-GNS/python -type d -name dgl*.egg)")

case "$1" in
    custom*)
        if [[ -z $DGL ]]; then
            printf "Make sure you've built dgl\n"
        else
            new_path=${PYTHONPATH/"$GNS_DGL:"/}
            echo "$new_path" | grep -q "$DGL"
            if [[ $? -ne 0 ]]; then
                export PYTHONPATH="$DGL:$new_path"
            fi
        fi
    ;;
    gns*)
        if [[ -z $GNS_DGL ]]; then
            printf "Make sure you've built dgl-GNS\n"
        else
            new_path=${PYTHONPATH/"$DGL:"/}
            echo "$new_path" | grep -q "$GNS_DGL"
            if [[ $? -ne 0 ]]; then
                export PYTHONPATH="$GNS_DGL:$new_path"
            fi
        fi
    ;;
    system*)
        new_path=${PYTHONPATH/"$GNS_DGL:"/}
        new_path=${PYTHONPATH/"$DGL:"/}
        export PYTHONPATH="$new_path"
    ;;
    *)
        printf "Usage:\n\tsource env.sh {custom|gns|system}\n"
    ;;
esac

printf "PYTHONPATH=$PYTHONPATH\n"

