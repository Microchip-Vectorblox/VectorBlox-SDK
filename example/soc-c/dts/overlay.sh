#!/bin/bash

set -e

OVERLAYS=/sys/kernel/config/device-tree/overlays

function add()
{
        if [ -e $OVERLAYS/$1 ]; then
                echo Overlay $1 already exists
                exit -1
        fi

        mkdir $OVERLAYS/$1
        cat $1 > $OVERLAYS/$1/dtbo
}

function remove()
{
        rmdir $OVERLAYS/$1
}

cmd=add

for i in $*; do
        case $i in
        add)
                cmd=add
                ;;

        rm)
                cmd=remove
                ;;

        *)
                $cmd $i
                ;;
        esac
done
