#!/bin/sh
# Run a command that stops production only once production has been idle:
# its log unchanged for MIN minutes (default 10).
# Usage: when_idle.sh [MIN] -- COMMAND ...
set -u
MIN=10; [ "$1" != "--" ] && { MIN=$1; shift; }; shift
L=/var/log/observe/app/strata-flashnext.log
echo "$(date +%H:%M:%S) waiting for $MIN idle minutes"
until [ -n "$(find $L -mmin +$MIN)" ]; do sleep 30; done
echo "$(date +%H:%M:%S) idle, starting: $*"
exec "$@"
