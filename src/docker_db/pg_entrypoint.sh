#!/bin/bash

trap 'true' SIGTERM
trap 'true' SIGINT

tail -f /dev/null &
wait $!

sudo chown duane:duane -R /var/lib/postgresql/data