def render_graph() -> str:
    """Interactive knowledge graph view — sigma.js WebGL renderer.

    The legacy canvas version recomputed O(n^2) repulsion on the main thread
    every frame and froze past ~1k nodes. This build renders with WebGL
    (sigma.js v2) and lays out with ForceAtlas2 running in a web worker
    (Barnes-Hut, O(n log n)), keeping thousands of entities fluid.
    """
    body = """
    <style>
      .graph-toolbar { display:flex; gap:10px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }
      .graph-toolbar input { padding:6px 10px; border:1px solid var(--border); border-radius:6px; font-size:13px; width:220px; }
      .graph-toolbar button { padding:6px 12px; }
      #graph-wrap { position:relative; height:72vh; border:1px solid var(--border); border-radius:10px; background:#fbfcfe; overflow:hidden; }
      #graph-empty { position:absolute; inset:0; display:none; align-items:center; justify-content:center; text-align:center; padding:20px; z-index:4; background:#fbfcfe; }
      #graph-search-list { position:absolute; top:52px; left:12px; z-index:6; background:#fff; border:1px solid var(--border);
        border-radius:8px; box-shadow:0 6px 24px rgba(15,25,40,.12); display:none; min-width:240px; max-height:280px; overflow:auto; }
      #graph-search-list .s-item { padding:7px 12px; font-size:13px; cursor:pointer; display:flex; gap:8px; align-items:center; }
      #graph-search-list .s-item:hover { background:#f0f4fa; }
      #graph-search-list .s-item .deg { color:var(--muted); font-size:11px; margin-left:auto; }
      .graph-legend { display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; }
      .legend-item { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); cursor:pointer;
        user-select:none; padding:2px 6px; border-radius:6px; }
      .legend-item:hover { background:#eef2f8; }
      .legend-item.off { opacity:.35; text-decoration:line-through; }
      .legend-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
      .graph-detail { margin-top:12px; padding:12px 14px; background:#fafbfc; border:1px solid var(--border); border-radius:8px; font-size:13px; min-height:20px; max-height:34vh; overflow:auto; }
      .graph-detail .detail-title { font-size:14px; margin-bottom:8px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
      .graph-detail .rel { padding:3px 0; border-bottom:1px dashed var(--border); cursor:pointer; }
      .graph-detail .rel:hover { background:#f0f4fa; }
      .graph-detail .rel .pred { color:var(--accent); font-weight:600; margin-right:6px; }
      #layout-status { font-size:12px; color:var(--muted); }
    </style>
    <div class="cards">
      <div class="card"><div class="label">文档</div><div class="value" id="stat-docs">—</div></div>
      <div class="card"><div class="label">实体</div><div class="value" id="stat-entities">—</div></div>
      <div class="card"><div class="label">关系</div><div class="value" id="stat-relations">—</div></div>
    </div>
    <div class="panel">
      <div class="graph-toolbar">
        <input id="graph-search" placeholder="搜索实体并定位…" autocomplete="off">
        <button class="btn" id="btn-layout" type="button">暂停布局</button>
        <button class="btn" id="btn-reheat" type="button">重新布局</button>
        <button class="btn" id="btn-reset" type="button">重置视图</button>
        <span id="layout-status"></span>
        <span class="muted">滚轮缩放 · 拖拽平移 · 拖节点调整 · 点击查看关联</span>
      </div>
      <div id="graph-wrap">
        <div id="graph-empty" class="warn"></div>
        <div id="graph-search-list"></div>
      </div>
      <div class="graph-legend" id="graph-legend"></div>
      <div class="graph-detail" id="graph-detail"><span class="muted">点击节点查看关联关系</span></div>
    </div>
    <script src="/graph/vendor/graphology.umd.min.js"></script>
    <script src="/graph/vendor/fa2.min.js"></script>
    <script src="/graph/vendor/fa2.supervisor.min.js"></script>
    <script src="/graph/vendor/sigma.min.js"></script>
    <script>
    (function(){
      function esc(s){ return String(s).replace(/[&<>"']/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
      var wrap = document.getElementById('graph-wrap');
      var empty = document.getElementById('graph-empty');
      var detail = document.getElementById('graph-detail');
      var legendEl = document.getElementById('graph-legend');
      var statusEl = document.getElementById('layout-status');

      fetch('/api/graph').then(function(r){ return r.json(); }).then(function(data){
        document.getElementById('stat-docs').textContent = data.stats.documents;
        document.getElementById('stat-entities').textContent = data.stats.entities;
        document.getElementById('stat-relations').textContent = data.stats.relations;
        if (!data.nodes || !data.nodes.length){
          empty.textContent = '知识图谱还是空的。';
          empty.style.display = 'flex';
          return;
        }
        init(data.nodes, data.edges);
      }).catch(function(e){
        empty.textContent = '加载图谱失败：' + esc(e && e.message ? e.message : '');
        empty.style.display = 'flex';
      });

      function init(rawNodes, rawEdges){
        var PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
        var typeColor = {}, paletteIdx = 0;

        var graph = new graphology.Graph({ multi: false, type: 'undirected' });
        rawNodes.forEach(function(n){
          if (!typeColor[n.type]) typeColor[n.type] = PALETTE[paletteIdx++ % PALETTE.length];
          var angle = Math.random() * Math.PI * 2, r = 80 * Math.sqrt(Math.random());
          try {
            graph.addNode(String(n.id), {
              label: n.name, name: n.name, etype: n.type, degree: n.degree || 0,
              x: Math.cos(angle) * r, y: Math.sin(angle) * r, size: 3
            });
          } catch (err) { /* duplicate id */ }
        });
        rawEdges.forEach(function(e){
          var s = String(e.source), t = String(e.target);
          if (graph.hasNode(s) && graph.hasNode(t) && s !== t && !graph.hasEdge(s, t)){
            try { graph.addEdge(s, t, { predicate: e.predicate, color: 'rgba(150,165,180,.35)', size: 1 }); } catch (err) {}
          }
        });

        // size by degree (sqrt scaling)
        graph.forEachNode(function(node, attrs){
          graph.setNodeAttribute(node, 'size', Math.max(3.5, Math.min(26, 3 + Math.sqrt(attrs.degree) * 2.2)));
        });

        // ---- layout: FA2 supervisor keeps matrices resident in a worker ----
        var order = graph.order;
        var settings = Object.assign(
          graphologyLayoutForceAtlas2.inferSettings(order),
          { barnesHutTheta: 0.9, scalingRatio: 6, slowDown: Math.max(4, 1 + Math.log(order)) }
        );
        statusEl.textContent = '初始布局中…（' + order + ' 节点）';
        // synchronous warm-up so the first paint is already structured
        graphologyLayoutForceAtlas2.assign(graph, {
          iterations: order > 4000 ? 60 : 120,
          settings: settings
        });

        var renderer = new Sigma(graph, wrap, {
          allowInvalidContainer: true,
          renderEdgeLabels: false,
          labelRenderedSizeThreshold: 13,
          labelDensity: 1.1,
          minCameraRatio: 0.02,
          maxCameraRatio: 12
        });

        // ---- interaction state ----
        var hoveredNode = null, selectedNode = null, searchHit = null;
        var offTypes = {};
        var nbSet = {};

        renderer.on('enterNode', function(e){ hoveredNode = e.node; nbSet = neighborsOf(e.node); refresh(); });
        renderer.on('leaveNode', function(){ hoveredNode = null; nbSet = selectedNode ? neighborsOf(selectedNode) : {}; refresh(); });
        renderer.on('clickNode', function(e){ selectNode(e.node); });
        renderer.on('clickStage', function(){
          if (selectedNode){ selectedNode = null; nbSet = {}; renderDetail(null); refresh(); }
        });
        renderer.on('downNode', function(){ setRunning(false); });   // dragging pauses layout for precision

        function neighborsOf(node){
          var set = {};
          graph.forEachNeighbor(node, function(nb){ set[nb] = true; });
          return set;
        }

        // ---- reducers (light table lookups; memoized by sigma) ----
        renderer.setSetting('nodeReducer', function(node, data){
          var res = Object.assign({}, data);
          res.color = typeColor[data.etype] || '#999';
          if (offTypes[data.etype]) { res.hidden = true; return res; }
          var active = hoveredNode || selectedNode || searchHit;
          if (active){
            if (node === active){ res.highlighted = true; res.zIndex = 3; res.size = data.size * 1.2; }
            else if (nbSet[node]){ res.zIndex = 2; }
            else {
              res.color = 'rgba(160,170,185,.22)';
              res.label = null;
              res.size = 2.5;
            }
          }
          if (searchHit === node && node !== active){ res.highlighted = true; }
          return res;
        });
        renderer.setSetting('edgeReducer', function(edge, data){
          var res = Object.assign({}, data);
          var active = hoveredNode || selectedNode;
          if (active){
            var s = graph.source(edge), t = graph.target(edge);
            if (s !== active && t !== active) res.hidden = true;
            else { res.color = '#5a7699'; res.size = 1.6; }
          } else {
            res.color = 'rgba(150,165,180,.35)';
            res.size = 1;
          }
          return res;
        });

        // ---- worker-driven layout ----
        var layout = new graphologyLayoutForceAtlas2Supervisor(graph, { settings: settings });

        var refreshQueued = false;
        function refresh(){
          if (!refreshQueued){
            refreshQueued = true;
            requestAnimationFrame(function(){ refreshQueued = false; renderer.refresh(); });
          }
        }
        // repaint loop only while the worker is feeding new positions
        (function paintLoop(){
          if (layout.isRunning()) renderer.refresh();
          requestAnimationFrame(paintLoop);
        })();

        function setRunning(v){
          if (v && !layout.isRunning()){ layout.start(); }
          else if (!v && layout.isRunning()){ layout.stop(); renderer.refresh(); }
          document.getElementById('btn-layout').textContent = v ? '暂停布局' : '继续布局';
          statusEl.textContent = v ? '布局运行中（worker 线程）' : '布局已暂停';
        }

        document.getElementById('btn-layout').addEventListener('click', function(){ setRunning(!layout.isRunning()); });
        document.getElementById('btn-reheat').addEventListener('click', function(){
          layout.stop();
          statusEl.textContent = '重新布局…';
          graphologyLayoutForceAtlas2.assign(graph, { iterations: 150, settings: settings });
          renderer.refresh();
          layout.start();
        });
        document.getElementById('btn-reset').addEventListener('click', function(){
          renderer.getCamera().animate({ x: 0, y: 0, ratio: 1, angle: 0 }, { duration: 300 });
        });

        // ---- legend with type filtering ----
        legendEl.innerHTML = Object.keys(typeColor).map(function(t){
          return '<span class="legend-item" data-type="' + esc(t) + '">' +
                 '<span class="legend-dot" style="background:' + typeColor[t] + '"></span>' + esc(t) + '</span>';
        }).join('');
        legendEl.addEventListener('click', function(ev){
          var item = ev.target.closest('.legend-item');
          if (!item) return;
          var t = item.getAttribute('data-type');
          offTypes[t] = !offTypes[t];
          item.classList.toggle('off', !!offTypes[t]);
          refresh();
        });

        // ---- detail panel ----
        function renderDetail(node){
          if (!node){
            detail.innerHTML = '<span class="muted">点击节点查看关联关系</span>';
            return;
          }
          var attrs = graph.getNodeAttributes(node);
          var rows = [];
          graph.forEachEdge(node, function(edge, eAttrs, s, t){
            var other = s === node ? t : s;
            rows.push('<div class="rel" data-node="' + esc(other) + '">' +
              '<span class="pred">' + esc(eAttrs.predicate || '相关') + '</span>' +
              esc(graph.getNodeAttribute(other, 'name')) +
              ' <span style="color:var(--muted)">(' + esc(graph.getNodeAttribute(other, 'etype')) + ')</span></div>');
          });
          detail.innerHTML =
            '<div class="detail-title"><strong>' + esc(attrs.name) + '</strong>' +
            '<span class="tag">' + esc(attrs.etype) + '</span>' +
            '<span class="muted">度数 ' + attrs.degree + '</span></div>' +
            (rows.length ? rows.join('') : '<span class="muted">无关联关系</span>');
        }
        detail.addEventListener('click', function(ev){
          var rel = ev.target.closest('.rel');
          if (!rel) return;
          focusNode(rel.getAttribute('data-node'));
        });

        function selectNode(node){
          selectedNode = node;
          nbSet = neighborsOf(node);
          renderDetail(node);
          var pos = renderer.getNodeDisplayData(node);
          if (pos){
            renderer.getCamera().animate(
              { x: pos.x, y: pos.y, ratio: Math.max(0.08, renderer.getCamera().getState().ratio * 0.55) },
              { duration: 320 }
            );
          }
          refresh();
        }
        function focusNode(node){
          if (!graph.hasNode(node)) return;
          selectNode(node);
        }

        // ---- search ----
        var input = document.getElementById('graph-search');
        var listEl = document.getElementById('graph-search-list');
        input.addEventListener('input', function(){
          var q = input.value.trim().toLowerCase();
          if (!q){ listEl.style.display = 'none'; searchHit = null; refresh(); return; }
          var hits = [];
          graph.forEachNode(function(node, attrs){
            if (hits.length >= 12) return;
            if (offTypes[attrs.etype]) return;
            if (attrs.name.toLowerCase().indexOf(q) !== -1) hits.push(node);
          });
          listEl.innerHTML = hits.map(function(node){
            var a = graph.getNodeAttributes(node);
            return '<div class="s-item" data-node="' + node + '">' +
                   '<span class="legend-dot" style="background:' + (typeColor[a.etype] || '#999') + '"></span>' +
                   esc(a.name) + '<span class="deg">度 ' + a.degree + '</span></div>';
          }).join('');
          listEl.style.display = hits.length ? 'block' : 'none';
        });
        listEl.addEventListener('click', function(ev){
          var item = ev.target.closest('.s-item');
          if (!item) return;
          listEl.style.display = 'none';
          input.value = '';
          selectNode(item.getAttribute('data-node'));
        });

        layout.start();
        statusEl.textContent = '布局运行中（worker 线程）· 可随时暂停';
      }
    })();
    </script>
    """
    return _page(body)
