#!/usr/bin/bash

SCRIPTS_DIR=$(dirname ${BASH_SOURCE[0]})
ORIGIN_DGL=$(realpath "$(find ${SCRIPTS_DIR}/../dgl/python -type d -name dgl*.egg)")
CUSTOM_DGL=$(realpath "$(find ${SCRIPTS_DIR}/../dgl-GNS/python -type d -name dgl*.egg)")

case "$1" in
    orig*|offi*)
        if [[ -z $ORIGIN_DGL ]]; then
            printf "Make sure you've built dgl\n"
        else
            new_path=${PYTHONPATH/"$CUSTOM_DGL:"/}
            echo "$new_path" | grep -q "$ORIGIN_DGL"
            if [[ $? -ne 0 ]]; then
                export PYTHONPATH="$ORIGIN_DGL:$new_path"
            fi
        fi
    ;;
    cust*)
        if [[ -z $CUSTOM_DGL ]]; then
            printf "Make sure you've built dgl\n"
        else
            new_path=${PYTHONPATH/"$ORIGIN_DGL:"/}
            echo "$new_path" | grep -q "$CUSTOM_DGL"
            if [[ $? -ne 0 ]]; then
                export PYTHONPATH="$CUSTOM_DGL:$new_path"
            fi
        fi
    ;;
    clear)
        new_path=${PYTHONPATH/"$CUSTOM_DGL:"/}
        new_path=${PYTHONPATH/"$ORIGIN_DGL:"/}
        export PYTHONPATH="$new_path"
    ;;
    *)
        printf "PYTHONPATH=$PYTHONPATH\n"
        printf "Usage:\n\tsource env.sh {origin|custom|clear}\n"
    ;;
esac
