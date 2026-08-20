#!/bin/bash
# Upload the release artifacts to the Zenodo draft, with visible progress.
#
# Safe to re-run: files already present in the deposition are skipped, so an
# interrupted run continues where it stopped. Nothing here publishes the record
# -- publishing stays a manual step in the web UI after the checksums are
# checked.
#
#   bash upload_to_zenodo.sh            # upload
#   bash upload_to_zenodo.sh --status   # just show what is on Zenodo now
set -u

DEPOSITION=22008658
BUCKET="https://zenodo.org/api/files/b39f794c-1e19-44b6-a103-9729f8414837"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/artifacts_external"
API="https://zenodo.org/api/deposit/depositions/$DEPOSITION"

if [[ -z "${ZENODO_TOKEN:-}" ]]; then
  if [[ -f "$HOME/.config/aser/zenodo.env" ]]; then
    set -a; . "$HOME/.config/aser/zenodo.env"; set +a
  else
    echo "ZENODO_TOKEN is not set and ~/.config/aser/zenodo.env does not exist." >&2
    exit 2
  fi
fi

PY=$(command -v python3 || command -v python)

status() {
  curl -sS --max-time 60 -H "Authorization: Bearer $ZENODO_TOKEN" "$API" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
print("deposition %s  state=%s  submitted=%s" % (d["id"], d.get("state"), d.get("submitted")))
files = d.get("files", [])
if not files:
    print("  (no files yet)")
for f in files:
    print("  %-34s %7.0f MB  md5=%s" % (f["filename"], f["filesize"] / 1e6, f.get("checksum", "")))
'
}

if [[ "${1:-}" == "--status" ]]; then status; exit 0; fi

echo "== already on Zenodo =="
uploaded=$(curl -sS --max-time 60 -H "Authorization: Bearer $ZENODO_TOKEN" "$API" \
           | "$PY" -c 'import json,sys; print(" ".join(f["filename"] for f in json.load(sys.stdin).get("files",[])))')
echo "  ${uploaded:-<none>}"
echo

total=$(ls -1 "$SRC"/*.pt 2>/dev/null | wc -l)
i=0
for f in "$SRC"/*.pt; do
  n=$(basename "$f")
  i=$((i + 1))
  size=$(du -h "$f" | cut -f1)
  case " $uploaded " in
    *" $n "*) echo "[$i/$total] $n ($size) -- already uploaded, skipping"; continue ;;
  esac

  for attempt in 1 2 3; do
    echo "[$i/$total] $n ($size) attempt $attempt -- uploading, progress below"
    # --progress-bar writes the meter to stderr so it is visible live;
    # -w prints the final status code on its own line.
    code=$(curl --progress-bar --max-time 7200 \
                -X PUT -H "Authorization: Bearer $ZENODO_TOKEN" \
                --upload-file "$f" "$BUCKET/$n" \
                -o /tmp/zenodo_resp_$n.txt -w '%{http_code}')
    if [[ "$code" == "200" || "$code" == "201" ]]; then
      echo "        -> HTTP $code  OK"
      break
    fi
    echo "        -> HTTP $code  FAILED" >&2
    head -c 200 /tmp/zenodo_resp_$n.txt >&2; echo >&2
    [[ $attempt == 3 ]] && { echo "giving up on $n" >&2; break; }
    sleep $((attempt * 15))
  done
done

echo
echo "== on Zenodo now =="
status
echo
echo "Next: compare the md5 values above against configs/manifest.json (which"
echo "records sha256), then publish from https://zenodo.org/deposit/$DEPOSITION"
