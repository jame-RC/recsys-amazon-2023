#!/usr/bin/env python3
"""Training Monitor — real-time dashboard server.

Usage:
    python -m train_monitor.server [--port 8080] [--metrics training_metrics.json]
"""

import json
import os
import sys
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>训练监控面板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}
.container{max-width:1200px;margin:0 auto}
h1{color:#38bdf8;margin-bottom:20px;font-size:24px;display:flex;align-items:center;gap:8px}
.status-bar{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.status-item{background:#1e293b;padding:14px 18px;border-radius:10px}
.status-item .label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.status-item .value{font-size:22px;font-weight:700;color:#38bdf8}
.chart-wrap{background:#1e293b;border-radius:10px;padding:20px;margin-bottom:20px}
.chart-wrap canvas{max-height:350px}
.table-wrap{overflow-x:auto;background:#1e293b;border-radius:10px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#334155;padding:10px 14px;text-align:left;color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
td{padding:8px 14px;border-bottom:1px solid #1e293b;font-family:'SF Mono','Fira Code',monospace;white-space:nowrap}
tr:hover{background:#334155}
tr:last-child td{border-bottom:none}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot-green{background:#22c55e;box-shadow:0 0 6px #22c55e88}
.dot-gray{background:#64748b}
.footer{text-align:right;font-size:11px;color:#475569;margin-top:8px}
</style>
</head>
<body>
<div class="container">
<h1>📊 训练监控面板</h1>
<div class="status-bar">
<div class="status-item"><div class="label">状态</div><div class="value"><span class="dot dot-green" id="statusDot"></span><span id="statusText">等待数据…</span></div></div>
<div class="status-item"><div class="label">已用时间</div><div class="value" id="elapsedText">—</div></div>
<div class="status-item"><div class="label">当前轮次</div><div class="value" id="epochText">—</div></div>
<div class="status-item"><div class="label">数据点</div><div class="value" id="pointsText">0</div></div>
</div>
<div class="chart-wrap">
<canvas id="metricsChart"></canvas>
</div>
<div class="table-wrap">
<table>
<thead><tr><th>#</th><th>时间</th><th>轮次</th><th id="metricHeaders"></th></tr></thead>
<tbody id="tableBody"></tbody>
</table>
</div>
<div class="footer" id="footer">等待数据…</div>
</div>
<script>
let chart=null;const METRICS_URL='/metrics';
async function fetchData(){try{const r=await fetch(METRICS_URL);const d=await r.json();update(d)}catch(e){document.getElementById('statusText').textContent='离线';document.getElementById('statusText').style.color='#ef4444';document.getElementById('statusDot').className='dot dot-gray'}}
function update(data){const pts=data.filter(d=>!d.event);const start=data.find(d=>d.event==='start');const end=data.find(d=>d.event==='end')
if(end){document.getElementById('statusText').textContent='已完成';document.getElementById('statusText').style.color='#22c55e';document.getElementById('statusDot').className='dot dot-green'}else if(start){document.getElementById('statusText').textContent='训练中';document.getElementById('statusText').style.color='#22c55e';document.getElementById('statusDot').className='dot dot-green'}else{document.getElementById('statusText').textContent='等待中';document.getElementById('statusText').style.color='#fbbf24';document.getElementById('statusDot').className='dot dot-gray'}
if(!pts.length)return
const last=pts[pts.length-1]
if(last.epoch!==undefined)document.getElementById('epochText').textContent=last.epoch
if(last.elapsed!==undefined){const s=Math.floor(last.elapsed);const m=Math.floor(s/60);const sec=s%60;document.getElementById('elapsedText').textContent=m>0?m+'m '+sec+'s':sec+'s'}
document.getElementById('pointsText').textContent=pts.length
const keys=Object.keys(last).filter(k=>!['time','elapsed','event','epoch'].includes(k)&&typeof last[k]==='number')
rebuildChart(pts,keys);rebuildTable(pts,keys)
document.getElementById('footer').textContent='更新于: '+new Date().toLocaleTimeString()}
function rebuildChart(pts,keys){const labels=pts.map((_,i)=>i);const colors=['#38bdf8','#f472b6','#34d399','#fbbf24','#a78bfa','#fb923c','#2dd4bf','#f87171','#818cf8','#f97316']
if(chart)chart.destroy()
const datasets=keys.map((k,i)=>({label:k,data:pts.map(e=>e[k]),borderColor:colors[i%colors.length],backgroundColor:colors[i%colors.length]+'30',tension:.3,fill:true,pointRadius:2}))
const ctx=document.getElementById('metricsChart').getContext('2d')
chart=new Chart(ctx,{type:'line',data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,animation:{duration:300},plugins:{legend:{labels:{color:'#94a3b8',boxWidth:12,font:{size:11}}}},scales:{x:{ticks:{color:'#64748b',font:{size:10}},grid:{color:'#334155'}},y:{ticks:{color:'#64748b',font:{size:10}},grid:{color:'#334155'}}}}})}
function rebuildTable(pts,keys){const hdr=document.getElementById('metricHeaders');hdr.innerHTML=keys.map(k=>'<th>'+k+'</th>').join('')
const bd=document.getElementById('tableBody')
bd.innerHTML=pts.slice(-100).reverse().map((e,i)=>{
const idx=pts.length-i;const t=e.time?e.time.split('T')[1].split('.')[0]:'-';const ep=e.epoch!==undefined?e.epoch:'-'
const vals=keys.map(k=>e[k]!==undefined?(typeof e[k]==='number'?e[k].toFixed(6):e[k]):'-')
return '<tr><td>'+idx+'</td><td>'+t+'</td><td>'+ep+'</td>'+vals.map(v=>'<td>'+v+'</td>').join('')+'</tr>'
}).join('')}
fetchData();setInterval(fetchData,2000)
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    """HTTP handler serving the dashboard HTML and /metrics JSON."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/metrics":
            self._serve_metrics()
        elif parsed.path == "/":
            self._serve_dashboard()
        else:
            self.send_error(404)

    def _serve_metrics(self):
        metrics_file: Path = self.server.metrics_file
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if metrics_file.exists():
            self.wfile.write(metrics_file.read_bytes())
        else:
            self.wfile.write(b"[]")

    def _serve_dashboard(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

    def log_message(self, fmt, *args):
        if args and args[0] != "200":
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="Training Monitor Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--metrics",
        default="training_metrics.json",
        help="Path to the metrics JSON file",
    )
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.metrics_file = Path(args.metrics)

    print("")
    print("  [Training Monitor Dashboard]")
    print("  -----------------------------")
    print(f"  URL:  http://localhost:{args.port}")
    print(f"  Logs: {server.metrics_file}")
    print("")
    print("  Press Ctrl+C to stop.")
    print("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
