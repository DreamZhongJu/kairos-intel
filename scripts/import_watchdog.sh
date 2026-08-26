#!/bin/bash
# Watchdog for ersi jsonl import: restart stalled importers, exit when complete.
LOG=/srv/games/kairos-import/watchdog.log

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

get_done() {
  docker exec kairos python -c "
import json, glob
n = 0
for f in glob.glob('/app/data/import_checkpoint_ersi_jsonl_v1.s*.json'):
    try:
        n += len(json.load(open(f)).get('done', {}))
    except Exception:
        pass
print(n)" 2>/dev/null || echo 0
}

alive_count() {
  docker exec kairos sh -c 'N=0; for p in /proc/[0-9]*/cmdline; do if tr "\0" " " < $p 2>/dev/null | grep -q "^python /app/import_chat_jsonl"; then N=$((N+1)); fi; done; echo $N'
}

fresh_lines() {
  docker exec kairos sh -c 'find /tmp/_imp_0.out /tmp/_imp_1.out /tmp/_imp_2.out /tmp/_imp_3.out /tmp/_imp_4.out /tmp/_imp_5.out -mmin -9 2>/dev/null | wc -l'
}

stale_files() {
  docker exec kairos sh -c 'N=0; for i in 0 1 2 3 4 5; do f=/tmp/_imp_$i.out; if [ -f $f ]; then A=$(stat -c %Y $f); T=$(date +%s); D=$((T-A)); if [ $D -gt 900 ]; then N=$((N+1)); fi; fi; done; echo $N'
}

restart_all() {
  docker exec kairos sh -c 'for p in /proc/[0-9]*/cmdline; do if tr "\0" " " < $p 2>/dev/null | grep -q "^python /app/import_chat_jsonl"; then kill -9 $(basename $(dirname $p)) 2>/dev/null; fi; done' >/dev/null 2>&1
  sleep 3
  docker exec -d -e EXTRACT_WORKERS=3 -e MODEL_MAX_RETRIES=0 kairos python /app/import_chat_jsonl.py --provider=nous --shards=6 --shard=0 --jobs=8
  docker exec -d -e EXTRACT_WORKERS=3 -e MODEL_MAX_RETRIES=0 kairos python /app/import_chat_jsonl.py --provider=openrouter --shards=6 --shard=1 --jobs=8
  docker exec -d -e EXTRACT_WORKERS=3 -e MODEL_MAX_RETRIES=0 kairos python /app/import_chat_jsonl.py --provider=zen --shards=6 --shard=2 --jobs=8
  docker exec -d -e EXTRACT_WORKERS=3 -e MODEL_MAX_RETRIES=0 kairos python /app/import_chat_jsonl.py --provider=nous --shards=6 --shard=3 --jobs=8
  docker exec -d -e EXTRACT_WORKERS=3 -e MODEL_MAX_RETRIES=0 kairos python /app/import_chat_jsonl.py --provider=openrouter --shards=6 --shard=4 --jobs=8
  docker exec -d -e EXTRACT_WORKERS=3 -e MODEL_MAX_RETRIES=0 kairos python /app/import_chat_jsonl.py --provider=zen --shards=6 --shard=5 --jobs=8
  log "restarted importers (6-lane layout)"
}

log "watchdog started"
while true; do
  DONE=$(get_done)
  if [ "$DONE" -ge 6253 ]; then
    log "COMPLETE done=$DONE"
    break
  fi
  A=$(alive_count)
  F=$(fresh_lines)
  S=$(stale_files)
  if [ "$A" -lt 5 ] || [ "$F" -eq 0 ] || [ "$S" -ge 1 ]; then
    log "stall detected alive=$A fresh_logs=$F stale_files=$S done=$DONE -> restarting"
    restart_all
  fi
  sleep 240
done
