# Serve the reports from this checkout: rebuild site/index.html (reports/index.py), then serve site/ (only: it links
# results/ and outdir/) on localhost. Share it through a tunnel with login in front (Tailscale, Cloudflare Access), never
# a bare port-forward.
#   sh serve.sh               http://localhost:8000
#   sh serve.sh 8080          another port
#   sh serve.sh --regen       first regenerate every report (reports/<sweep>.py) from outdir/; a failing one is named
#                             at the end, the rest still run
set -e
cd "$(dirname "$0")"
PORT=8000
REGEN=0
for a in "$@"; do
    case "$a" in
        --regen) REGEN=1 ;;
        *) PORT=$a ;;
    esac
done
if [ "$REGEN" = 1 ]; then
    FAILED=""
    for f in reports/*.py; do
        n=$(basename "$f" .py)
        case "$n" in __init__|page|index) continue ;; esac  # the page shell and the index aren't reports
        uv run --extra analysis python -m "reports.$n" || FAILED="$FAILED reports/$n.py"
    done
    [ -z "$FAILED" ] || echo "FAILED to regenerate:$FAILED (see their errors above)" >&2
fi
uv run --extra analysis python -m reports.index
echo "serving site/ at http://localhost:$PORT"
uv run python -m http.server "$PORT" --bind 127.0.0.1 --directory site
