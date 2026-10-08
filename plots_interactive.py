"""Interactive counterparts of plots.py for report_page: Plotly charts (scroll or drag to zoom, pan, linear/log axis
buttons, hover tooltips, click a legend entry to hide it), click-to-sort tables, and image grids whose full-screen
viewer zooms (nearest neighbour), pans, steps across the grid with the arrow keys and toggles a cell's overlay layers.
Each helper returns an HTML fragment marked with MARK; report_page.write appends RUNTIME to any page holding one.
Stdlib only: figures are Plotly JSON built here and drawn by plotly.js from a CDN. The viewer is spearmint's lightbox
(spearmint/viz.py, from mia-muvit's report.py) with layer toggles in place of its j/k overlay cycling."""

import base64
import html
import itertools
import json
import mimetypes
from pathlib import Path

MARK = "data-interactive"  # attribute on every fragment's root element
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]  # plots.COLORS
GREY = "#999999"
DASHES = ["dash", "dot", "dashdot"]
THUMB = 220  # image-grid thumbnail width, px
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
CSS = r"""
table.sortable th{cursor:pointer;user-select:none} table.sortable th.asc::after{content:" \25B4"} table.sortable th.desc::after{content:" \25BE"}
table.imggrid td{vertical-align:top} table.imggrid img.zoom{cursor:zoom-in;border:1px solid #ddd}
#viewer{position:fixed;inset:0;background:rgba(0,0,0,.93);display:none;z-index:1000;cursor:grab;overflow:hidden}
#viewer.open{display:block} #viewer img{position:absolute;left:0;top:0;image-rendering:pixelated}
#viewer-bar{position:fixed;top:0;left:0;right:0;padding:6px 12px;color:#ddd;font-size:13px;background:rgba(0,0,0,.6);z-index:1001}
#viewer-bar label{margin-right:12px} #viewer-bar .hint{float:right;color:#999}
"""
JS = r"""
document.querySelectorAll("div.iplot").forEach(function(div) {
  var d = JSON.parse(document.getElementById(div.id + "-data").textContent);
  Plotly.newPlot(div, d.data, d.layout, {responsive: true, scrollZoom: true, displaylogo: false});
});

// Click a header to sort its column (descending first, then toggling); numbers compare as numbers and stay above
// text (e.g. "—" for missing) either way.
document.addEventListener("click", function(e) {
  var th = e.target.closest && e.target.closest("table.sortable th");
  if (!th) return;
  var t = th.closest("table"), i = th.cellIndex, dir = th.classList.contains("desc") ? 1 : -1;
  var body = t.tBodies[0], rows = Array.prototype.slice.call(body.rows);
  function key(r) { var s = r.cells[i] ? r.cells[i].textContent : "", n = parseFloat(s.replace(/,/g, "")); return isNaN(n) ? s : n; }
  rows.sort(function(a, b) {
    var x = key(a), y = key(b), nx = typeof x === "number", ny = typeof y === "number";
    if (nx !== ny) return nx ? -1 : 1;
    return dir * (nx ? x - y : String(x).localeCompare(String(y), undefined, {numeric: true}));
  });
  rows.forEach(function(r) { body.appendChild(r); });
  t.querySelectorAll("th").forEach(function(h) { h.classList.remove("asc", "desc"); });
  th.classList.add(dir === 1 ? "asc" : "desc");
});

// Image viewer: click a grid thumbnail to open its cell full-screen. Every layer of the cell is drawn at the first
// layer's size, stacked in order; the lowest visible layer is opaque, the ones above it at the slider's opacity.
// Sized via width/height, not a CSS transform, which the GPU bilinear-filters at high zoom.
(function() {
  var V = document.getElementById("viewer"), BAR = document.getElementById("viewer-bar");
  var scale = 1, tx = 0, ty = 0, drag = false, lx = 0, ly = 0, curW = 0, curH = 0;
  var grid = [], row = 0, col = 0, layers = [], on = {}, alpha = 0.5;  // on: layer index -> shown, kept across cells
  function esc(s) { return s.replace(/[&<>"]/g, function(c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
  function apply() {
    var w = layers[0].naturalWidth * scale, h = layers[0].naturalHeight * scale;
    layers.forEach(function(img) { img.style.left = tx + "px"; img.style.top = ty + "px"; img.style.width = w + "px"; img.style.height = h + "px"; });
  }
  function fit() {
    var nw = layers[0].naturalWidth || 1, nh = layers[0].naturalHeight || 1;
    scale = Math.min(V.clientWidth / nw, V.clientHeight / nh) * 0.95; tx = (V.clientWidth - nw * scale) / 2; ty = (V.clientHeight - nh * scale) / 2;
    apply();
  }
  function loaded() {  // same-size cell: keep zoom and pan, so stepping across the grid compares one spot
    var b = layers[0];
    if (b.naturalWidth === curW && b.naturalHeight === curH) apply(); else fit();
    curW = b.naturalWidth; curH = b.naturalHeight;
  }
  function paint() {
    var below = false;
    layers.forEach(function(img, i) {
      var shown = on[i] !== false;
      img.style.display = shown ? "" : "none";
      img.style.opacity = shown && below ? alpha : 1;
      below = below || shown;
    });
    var src = grid[row][col], h = "<b>" + esc(src[0].title) + "</b> ";
    if (src.length > 1) {
      h += src.map(function(s, i) { return "<label><input type='checkbox' data-i='" + i + "'" + (on[i] !== false ? " checked" : "") + "> " + (i + 1) + " " + esc(s.alt) + "</label>"; }).join("");
      h += "<label>overlay opacity <input type='range' min='0' max='1' step='0.05' value='" + alpha + "'></label>";
    }
    BAR.innerHTML = h + "<span class='hint'>arrows: grid · 1-9: layers · scroll: zoom · drag: pan · Esc: close</span>";
  }
  function show() {
    layers.forEach(function(img) { img.remove(); });
    layers = grid[row][col].map(function(s) {
      var img = document.createElement("img");
      img.className = "layer"; img.src = s.src; img.alt = s.alt; img.onload = apply;
      V.appendChild(img); return img;
    });
    paint();
    if (layers[0].complete) loaded(); else layers[0].onload = loaded;
  }
  function open(el) {
    var trs = Array.prototype.slice.call(el.closest("table").tBodies[0].rows);
    grid = trs.map(function(tr) {
      return Array.prototype.slice.call(tr.cells).map(function(td) { return Array.prototype.slice.call(td.querySelectorAll("img")); });
    });
    grid.forEach(function(cells, i) { cells.forEach(function(imgs, j) { if (imgs[0] === el) { row = i; col = j; } }); });
    curW = 0; curH = 0; V.classList.add("open"); show();
  }
  function move(dr, dc) {  // the next nonempty cell in that direction; stop at the edge
    for (var r = row + dr, c = col + dc; r >= 0 && r < grid.length && c >= 0 && c < grid[r].length; r += dr, c += dc) {
      if (grid[r][c].length) { row = r; col = c; show(); return; }
    }
  }
  function close() { V.classList.remove("open"); }
  document.addEventListener("click", function(e) { if (e.target.matches && e.target.matches("table.imggrid img.zoom")) open(e.target); });
  V.addEventListener("click", function(e) { if (e.target === V) close(); });
  BAR.addEventListener("input", function(e) {
    if (e.target.type === "checkbox") on[+e.target.dataset.i] = e.target.checked; else alpha = +e.target.value;
    paint();
  });
  document.addEventListener("keydown", function(e) {
    if (!V.classList.contains("open")) return;
    if (e.key === "Escape") { close(); return; }
    var i = "123456789".indexOf(e.key);
    if (i >= 0 && i < layers.length) { on[i] = on[i] === false; paint(); return; }
    var d = {ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0]}[e.key];
    if (d) { e.preventDefault(); move(d[0], d[1]); }
  });
  V.addEventListener("wheel", function(e) {  // zoom toward the cursor, ~8% per notch
    e.preventDefault();
    var f = Math.exp(-e.deltaY * 0.0008);
    tx = e.clientX - (e.clientX - tx) * f; ty = e.clientY - (e.clientY - ty) * f; scale *= f; apply();
  }, {passive: false});
  V.addEventListener("mousedown", function(e) {
    if (!e.target.classList.contains("layer")) return;
    drag = true; lx = e.clientX; ly = e.clientY; V.style.cursor = "grabbing"; e.preventDefault();
  });
  window.addEventListener("mousemove", function(e) {
    if (!drag) return;
    tx += e.clientX - lx; ty += e.clientY - ly; lx = e.clientX; ly = e.clientY; apply();
  });
  window.addEventListener("mouseup", function() { drag = false; V.style.cursor = "grab"; });
})();
"""
RUNTIME = (f"<style>{CSS}</style><div id='viewer'><div id='viewer-bar'></div></div>"
           f"<script src='{PLOTLY}'></script><script>{JS}</script>")

_ids = itertools.count()  # unique plot div ids within one process


def tooltip(row: dict) -> str:
    return "<br>".join(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}" for k, v in row.items())


def figure(data: list, layout: dict, logx: bool = False) -> str:
    """A Plotly figure: an empty div and its JSON, drawn by RUNTIME. Buttons above the plot switch each numeric axis
    between linear and log (a "category" axis, e.g. bar labels, gets none)."""
    layout = {"height": 380, "margin": {"t": 40, "r": 10, "b": 50, "l": 60}, "hovermode": "closest", "font": {"size": 11},
              "paper_bgcolor": "#fff", "plot_bgcolor": "#fff", "legend": {"font": {"size": 10}}, **layout}
    layout["xaxis"] = {"type": "log" if logx else "linear", "gridcolor": "#eee", **layout.get("xaxis", {})}
    layout["yaxis"] = {"type": "linear", "gridcolor": "#eee", **layout.get("yaxis", {})}
    numeric = [ax for ax in "xy" if layout[f"{ax}axis"]["type"] != "category"]
    layout["updatemenus"] = [
        {"type": "buttons", "direction": "right", "x": 0.35 * i, "y": 1.02, "xanchor": "left", "yanchor": "bottom",
         "active": int(layout[f"{ax}axis"]["type"] == "log"), "pad": {"t": 0, "b": 0}, "font": {"size": 10},
         "buttons": [{"label": f"{ax} {t}", "method": "relayout", "args": [{f"{ax}axis.type": t}]} for t in ("linear", "log")]}
        for i, ax in enumerate(numeric)]
    pid = f"iplot{next(_ids)}"
    payload = json.dumps({"data": data, "layout": layout}).replace("</", "<\\/")  # a string can't close the script
    return f"<div class='iplot' id='{pid}' {MARK}></div><script type='application/json' id='{pid}-data'>{payload}</script>"


def scatter(groups: dict, x: str, y: str, hlines: dict | None = None, logx: bool = False) -> str:
    """groups: label -> rows, dicts holding x and y; all of a row's fields show in its point's tooltip. hlines: name ->
    y, a gray baseline across the plot each (in the legend; dash styles cycle)."""
    data = [{"type": "scatter", "mode": "markers", "name": label, "x": [r[x] for r in rows], "y": [r[y] for r in rows],
             "text": [tooltip(r) for r in rows], "hovertemplate": "%{text}<extra></extra>",
             "marker": {"color": c, "size": 8, "opacity": 0.75}}
            for (label, rows), c in zip(groups.items(), itertools.cycle(COLORS))]
    shapes = [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": v, "y1": v, "name": name, "showlegend": True,
               "line": {"color": GREY, "width": 1.5, "dash": dash}}
              for (name, v), dash in zip((hlines or {}).items(), itertools.cycle(DASHES))]
    return figure(data, {"shapes": shapes, "xaxis": {"title": x}, "yaxis": {"title": y}}, logx=logx)


def lines(series: dict, xlabel: str, ylabel: str) -> str:
    """series: label -> (xs, ys), one line each. The tooltip lists every line's value at the hovered x."""
    data = [{"type": "scatter", "mode": "lines", "name": label, "x": list(xs), "y": list(ys), "line": {"color": c, "width": 1.8},
             "hovertemplate": "%{y:.4g}"}
            for (label, (xs, ys)), c in zip(series.items(), itertools.cycle(COLORS))]
    return figure(data, {"hovermode": "x unified", "xaxis": {"title": xlabel}, "yaxis": {"title": ylabel}})


def bar(rows: list, xlabel: str) -> str:
    """Horizontal bars coloured by group, top to bottom in the order given (plots.group_bar). rows: (label, value, group)."""
    groups = list(dict.fromkeys(g for _, _, g in rows))
    data = [{"type": "bar", "orientation": "h", "name": g, "marker": {"color": c},
             "y": [label for label, _, gr in rows if gr == g], "x": [v for _, v, gr in rows if gr == g],
             "hovertemplate": "%{y}: %{x:.4g}<extra></extra>"}
            for g, c in zip(groups, itertools.cycle(COLORS))]
    yaxis = {"type": "category", "categoryorder": "array", "categoryarray": [label for label, _, _ in rows], "autorange": "reversed", "automargin": True}
    return figure(data, {"height": 22 * len(rows) + 110, "xaxis": {"title": xlabel}, "yaxis": yaxis, "barmode": "group"})


def cell(c) -> str:
    return f"<td class='n'>{c:.4g}</td>" if isinstance(c, float) else f"<td>{html.escape(str(c))}</td>"


def table(header: list, rows: list) -> str:
    """Click a column header to sort by it. Floats show 4 significant digits; anything else as its str()."""
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    body = "".join("<tr>" + "".join(cell(c) for c in r) + "</tr>" for r in rows)
    return f"<table class='sortable' {MARK}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def img(path: Path, **attrs) -> str:
    assert path.is_file(), f"no image at {path}; run ./pull.sh?"
    mime = mimetypes.guess_type(path.name)[0]
    assert mime and mime.startswith("image/"), f"{path}: not an image"
    src = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    return "<img " + " ".join(f"{k}='{html.escape(str(v), quote=True)}'" for k, v in attrs.items()) + f" src='{src}'>"


def images(grid: dict) -> str:
    """grid: row label -> {column label: [image paths]}, embedded in the page. A cell's first image is its thumbnail;
    the rest are layers drawn over it in the viewer, toggled by checkbox or 1-9 (layer names: file stems). An empty or
    absent cell shows a dash."""
    cols = list(dict.fromkeys(c for cells in grid.values() for c in cells))
    head = "<tr><th></th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr>"
    rows = []
    for r, cells in grid.items():
        tds = []
        for c in cols:
            paths = cells.get(c, [])
            title = f"{r} · {c}"
            layers = [img(p, alt=p.stem, title=title, **({"class": "zoom", "width": THUMB} if i == 0 else {"hidden": ""}))
                      for i, p in enumerate(paths)]
            note = f"<br><small>{len(paths)} layers</small>" if len(paths) > 1 else ""
            tds.append(f"<td>{''.join(layers)}{note}</td>" if paths else "<td>–</td>")
        rows.append(f"<tr><th>{html.escape(str(r))}</th>{''.join(tds)}</tr>")
    return f"<table class='imggrid' {MARK}><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"
