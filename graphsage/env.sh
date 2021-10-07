#!/usr/bin/bash

ORIGIN_DGL=$(realpath "$(find ../dgl/python -type d -name dgl*.egg)")
CUSTOM_DGL=$(realpath "$(find ../custom-dgl/python -type d -name dgl*.egg)")

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
        printf "Usage:\n\t$0 {origin|custom|clear}\n"
    ;;
esac
