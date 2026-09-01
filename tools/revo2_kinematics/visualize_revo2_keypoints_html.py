#!/usr/bin/env python3
"""Build a standalone interactive Revo2 keypoint/mimic FK visualizer.

The generated HTML has no server or JavaScript-package dependency.  It embeds
the same zero-pose transforms, keypoint offsets, joint limits, and mimic
relations used by :class:`Revo2Kinematics`, then evaluates FK in the browser.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from .revo2_kinematics import Revo2Kinematics, _ZERO_POSES
except ImportError:  # Direct execution: python tools/.../this_file.py
    from revo2_kinematics import Revo2Kinematics, _ZERO_POSES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/visualizations/revo2_kinematics"
    / "revo2_keypoints_mimic_interactive.html"
)

HAND_CHAINS = (
    ("index", (0, 1, 2, 3, 17), "#4fd1ff"),
    ("middle", (0, 4, 5, 6, 18), "#6ee7b7"),
    ("ring", (0, 10, 11, 12, 19), "#fbbf24"),
    ("little", (0, 7, 8, 9, 20), "#fb7185"),
    ("thumb", (0, 13, 14, 15, 16), "#c084fc"),
)

ACTIVE_LABELS = (
    "Thumb metacarpal",
    "Thumb proximal",
    "Index proximal",
    "Middle proximal",
    "Ring proximal",
    "Pinky proximal",
)


def _serialized_model(fk: Revo2Kinematics) -> dict:
    lower, upper = fk.get_joint_limits()
    zero_poses = {
        name: {"r": rotation.tolist(), "t": translation.tolist()}
        for name, (rotation, translation) in _ZERO_POSES.items()
    }

    # Use all 64 joint-limit corners to choose one fixed cube.  The camera does
    # not jump when a slider is moved to a new configuration.
    samples = []
    for corner in itertools.product((0, 1), repeat=6):
        q = np.where(np.asarray(corner, dtype=bool), upper, lower)
        samples.append(fk.get_keypoints(q))
    samples_array = np.concatenate(samples, axis=0)
    xyz_min = samples_array.min(axis=0)
    xyz_max = samples_array.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    span = float(max((xyz_max - xyz_min).max(), 0.12))

    return {
        "title": "Revo2 semantic keypoints + mimic FK",
        "activeJointNames": list(fk.joint_names),
        "activeLabels": list(ACTIVE_LABELS),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mimicJointNames": list(fk.mimic_joint_names),
        "mimicLeaderIndices": list(fk.mimic_leader_indices),
        "mimicMultipliers": list(fk.mimic_multipliers),
        "keypointNames": list(fk.keypoint_names),
        "parentLinks": list(fk._parent_links),
        "localXyz": fk._local_xyz.tolist(),
        "zeroPoses": zero_poses,
        "chains": [
            {"name": name, "indices": list(indices), "color": color}
            for name, indices, color in HAND_CHAINS
        ],
        "viewCenter": center.tolist(),
        "viewSpan": span * 1.25,
    }


def _evaluate_serialized_model(model: dict, q: np.ndarray) -> np.ndarray:
    """Python mirror of the embedded browser FK, used before writing HTML."""

    def compose(parent, child):
        parent_r, parent_t = parent
        child_r, child_t = child
        return parent_r @ child_r, parent_r @ child_t + parent_t

    def axis_rotation(axis: str, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        if axis == "X":
            return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)))
        if axis == "Y":
            return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)))
        return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    def moving(name: str, axis: str, angle: float):
        zero = model["zeroPoses"][name]
        return (
            np.asarray(zero["r"]) @ axis_rotation(axis, angle),
            np.asarray(zero["t"]),
        )

    def fixed(name: str):
        zero = model["zeroPoses"][name]
        return np.asarray(zero["r"]), np.asarray(zero["t"])

    mimic = q[np.asarray(model["mimicLeaderIndices"])] * np.asarray(
        model["mimicMultipliers"]
    )
    links = {"right_hand_base_link": (np.eye(3), np.zeros(3))}
    links["right_thumb_metacarpal_link"] = compose(
        links["right_hand_base_link"],
        moving("right_thumb_metacarpal_link", "Z", -q[0]),
    )
    links["right_thumb_proximal_link"] = compose(
        links["right_thumb_metacarpal_link"],
        moving("right_thumb_proximal_link", "X", q[1]),
    )
    links["right_thumb_distal_link"] = compose(
        links["right_thumb_proximal_link"],
        moving("right_thumb_distal_link", "X", mimic[0]),
    )
    links["right_thumb_touch_link"] = compose(
        links["right_thumb_distal_link"], fixed("right_thumb_touch_link")
    )
    for offset, finger in enumerate(("index", "middle", "ring", "pinky")):
        proximal = f"right_{finger}_proximal_link"
        distal = f"right_{finger}_distal_link"
        touch = f"right_{finger}_touch_link"
        links[proximal] = compose(
            links["right_hand_base_link"], moving(proximal, "Y", q[offset + 2])
        )
        links[distal] = compose(
            links[proximal], moving(distal, "Y", mimic[offset + 1])
        )
        links[touch] = compose(links[distal], fixed(touch))

    points = []
    for parent, local in zip(model["parentLinks"], model["localXyz"]):
        rotation, translation = links[parent]
        points.append(rotation @ np.asarray(local) + translation)
    return np.asarray(points)


def _validate_serialization(fk: Revo2Kinematics, model: dict) -> float:
    lower, upper = fk.get_joint_limits()
    rng = np.random.default_rng(20260901)
    max_error = 0.0
    samples = [lower, upper, 0.5 * (lower + upper)]
    samples.extend(rng.uniform(lower, upper) for _ in range(32))
    for q in samples:
        reference = fk.get_keypoints(q)
        serialized = _evaluate_serialized_model(model, q)
        max_error = max(max_error, float(np.max(np.abs(reference - serialized))))
    if max_error > 1.0e-12:
        raise RuntimeError(
            "browser FK serialization does not match Revo2Kinematics: "
            f"max absolute error={max_error:.3e} m"
        )
    return max_error


HTML_TEMPLATE = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Revo2 keypoint &amp; mimic FK</title>
  <style>
    :root { color-scheme: dark; --bg:#07111f; --panel:#0e1b2d; --line:#22344c; --text:#e7eef8; --muted:#8fa5bd; --accent:#38bdf8; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 65% 20%,#102a45 0,#07111f 44%,#050b14 100%); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; overflow:hidden; }
    .app { display:grid; grid-template-columns:minmax(330px,390px) 1fr; width:100vw; height:100vh; }
    aside { overflow:auto; padding:22px; border-right:1px solid var(--line); background:rgba(7,17,31,.91); backdrop-filter:blur(14px); }
    h1 { font-size:21px; margin:0 0 6px; letter-spacing:-.02em; }
    .subtitle { color:var(--muted); font-size:12px; line-height:1.55; margin-bottom:18px; }
    .badge { display:inline-flex; gap:7px; align-items:center; border:1px solid #1d4b68; background:#0c263a; color:#7dd3fc; border-radius:999px; padding:5px 9px; font-size:11px; font-weight:700; margin:0 5px 5px 0; }
    .dot { width:7px; height:7px; border-radius:50%; background:#34d399; box-shadow:0 0 9px #34d399; }
    .section { margin-top:18px; }
    .section-title { font-size:11px; color:#9db1c8; text-transform:uppercase; letter-spacing:.11em; font-weight:800; margin:0 0 10px; }
    .slider-card { padding:10px 11px 9px; margin:8px 0; border:1px solid var(--line); border-radius:10px; background:rgba(14,27,45,.72); }
    .slider-head { display:flex; justify-content:space-between; gap:10px; align-items:baseline; }
    .slider-name { font-size:12px; font-weight:700; }
    .slider-value { font:11px ui-monospace,SFMono-Regular,Menlo,monospace; color:#7dd3fc; white-space:nowrap; }
    .joint-id { display:block; color:#6f849d; font:9px ui-monospace,SFMono-Regular,Menlo,monospace; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
    input[type=range] { width:100%; margin:9px 0 1px; accent-color:var(--accent); }
    .range-row { display:flex; justify-content:space-between; color:#5f7690; font-size:9px; }
    .buttons { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }
    button { border:1px solid #29405b; background:#11233a; color:#dce8f6; border-radius:8px; padding:8px 5px; font-size:11px; font-weight:700; cursor:pointer; }
    button:hover { background:#17314f; border-color:#3b82a8; }
    table { border-collapse:collapse; width:100%; font-size:10px; }
    th,td { padding:6px 5px; border-bottom:1px solid #1c2c41; text-align:right; }
    th:first-child,td:first-child { text-align:left; }
    th { color:#7890aa; font-weight:600; }
    td { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    .mimic-value { color:#f0abfc; }
    .checks { display:flex; flex-wrap:wrap; gap:12px; font-size:11px; color:#a8bad0; }
    .checks label { cursor:pointer; }
    .checks input { accent-color:#38bdf8; vertical-align:-2px; }
    main { min-width:0; min-height:0; position:relative; }
    canvas { width:100%; height:100%; display:block; cursor:grab; }
    canvas.dragging { cursor:grabbing; }
    .hud { position:absolute; pointer-events:none; top:18px; left:18px; display:flex; flex-wrap:wrap; gap:7px; }
    .hud span { padding:6px 9px; border-radius:7px; border:1px solid rgba(83,117,151,.42); background:rgba(4,12,22,.72); color:#a9bdd2; font:10px ui-monospace,SFMono-Regular,Menlo,monospace; backdrop-filter:blur(5px); }
    .legend { position:absolute; right:18px; top:18px; border:1px solid rgba(83,117,151,.42); background:rgba(4,12,22,.72); border-radius:9px; padding:9px 11px; font-size:10px; color:#a9bdd2; pointer-events:none; }
    .legend-row { display:flex; align-items:center; gap:7px; margin:4px 0; }
    .swatch { width:16px; height:3px; border-radius:3px; }
    .help { position:absolute; bottom:16px; right:18px; color:#6f879f; font-size:10px; pointer-events:none; }
    @media(max-width:800px){ body{overflow:auto}.app{grid-template-columns:1fr;height:auto}aside{height:auto;border-right:0;border-bottom:1px solid var(--line)}main{height:72vh}.slider-grid{display:grid;grid-template-columns:1fr 1fr;gap:0 8px} }
  </style>
</head>
<body>
<div class="app">
  <aside>
    <h1>Revo2 keypoint &amp; mimic FK</h1>
    <div class="subtitle">6개 active joint를 각각 움직이면 동일한 FK로 21개 semantic keypoint와 dependent distal joint가 실시간 갱신됩니다. 단위는 rad / m입니다.</div>
    <div><span class="badge"><span class="dot"></span>21 keypoints finite</span><span class="badge">6 active + 5 mimic</span></div>
    <div class="section">
      <div class="section-title">Active joints</div>
      <div id="sliders" class="slider-grid"></div>
    </div>
    <div class="section buttons">
      <button id="openBtn">Open</button><button id="midBtn">Mid</button><button id="closeBtn">Close</button><button id="randomBtn">Random</button>
    </div>
    <div class="section">
      <div class="section-title">Mimic joints — 자동 추종</div>
      <table><thead><tr><th>Follower</th><th>Rule</th><th>rad</th></tr></thead><tbody id="mimicBody"></tbody></table>
    </div>
    <div class="section">
      <div class="section-title">View</div>
      <div class="checks"><label><input id="labelsToggle" type="checkbox" checked> kp labels</label><label><input id="originsToggle" type="checkbox" checked> link origins</label><label><input id="gridToggle" type="checkbox" checked> ground grid</label></div>
    </div>
  </aside>
  <main>
    <canvas id="view"></canvas>
    <div class="hud"><span id="shapeHud">keypoints (21, 3)</span><span id="finiteHud">finite: true</span><span id="selectedHud">preset: open</span></div>
    <div id="legend" class="legend"></div>
    <div class="help">drag: orbit · wheel: zoom · shift+drag: pan</div>
  </main>
</div>
<script>
const MODEL = __MODEL_JSON__;

const eye3=()=>[[1,0,0],[0,1,0],[0,0,1]];
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const scale=(a,s)=>a.map(v=>v*s);
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>Math.sqrt(dot(a,a));
const normalize=a=>scale(a,1/Math.max(norm(a),1e-12));
const matVec=(a,v)=>a.map(r=>dot(r,v));
const matMul=(a,b)=>a.map((r,i)=>b[0].map((_,j)=>r[0]*b[0][j]+r[1]*b[1][j]+r[2]*b[2][j]));
function rot(axis,a){const c=Math.cos(a),s=Math.sin(a);if(axis==='X')return [[1,0,0],[0,c,-s],[0,s,c]];if(axis==='Y')return [[c,0,s],[0,1,0],[-s,0,c]];return [[c,-s,0],[s,c,0],[0,0,1]];}
function compose(p,c){return {r:matMul(p.r,c.r),t:add(matVec(p.r,c.t),p.t)};}
function fixed(name){const z=MODEL.zeroPoses[name];return {r:z.r,t:z.t};}
function moving(name,axis,angle){const z=MODEL.zeroPoses[name];return {r:matMul(z.r,rot(axis,angle)),t:z.t};}
function forward(q){
  const mimic=MODEL.mimicLeaderIndices.map((leader,i)=>q[leader]*MODEL.mimicMultipliers[i]);
  const links={right_hand_base_link:{r:eye3(),t:[0,0,0]}};
  links.right_thumb_metacarpal_link=compose(links.right_hand_base_link,moving('right_thumb_metacarpal_link','Z',-q[0]));
  links.right_thumb_proximal_link=compose(links.right_thumb_metacarpal_link,moving('right_thumb_proximal_link','X',q[1]));
  links.right_thumb_distal_link=compose(links.right_thumb_proximal_link,moving('right_thumb_distal_link','X',mimic[0]));
  links.right_thumb_touch_link=compose(links.right_thumb_distal_link,fixed('right_thumb_touch_link'));
  ['index','middle','ring','pinky'].forEach((finger,i)=>{
    const p=`right_${finger}_proximal_link`,d=`right_${finger}_distal_link`,t=`right_${finger}_touch_link`;
    links[p]=compose(links.right_hand_base_link,moving(p,'Y',q[i+2]));
    links[d]=compose(links[p],moving(d,'Y',mimic[i+1]));
    links[t]=compose(links[d],fixed(t));
  });
  const keypoints=MODEL.localXyz.map((local,i)=>{const pose=links[MODEL.parentLinks[i]];return add(matVec(pose.r,local),pose.t);});
  return {links,keypoints,mimic};
}

const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
let q=MODEL.lower.slice(), yaw=-0.72, pitch=0.42, distance=MODEL.viewSpan*2.8;
let target=MODEL.viewCenter.slice(), showLabels=true, showOrigins=true, showGrid=true;
let dragging=false,lastX=0,lastY=0,panMode=false,preset='open';
const sliders=[];
function shortName(name){return name.replace(/^right_/,'').replace(/_joint$/,'').replaceAll('_',' ');}
function createControls(){
  const root=document.getElementById('sliders');
  MODEL.activeJointNames.forEach((name,i)=>{
    const card=document.createElement('div');card.className='slider-card';
    card.innerHTML=`<div class="slider-head"><span class="slider-name">${i+1}. ${MODEL.activeLabels[i]}</span><span id="value${i}" class="slider-value"></span></div><span class="joint-id">${name}</span><input id="slider${i}" type="range" min="${MODEL.lower[i]}" max="${MODEL.upper[i]}" step="0.001" value="${q[i]}"><div class="range-row"><span>${MODEL.lower[i].toFixed(3)}</span><span>${MODEL.upper[i].toFixed(3)}</span></div>`;
    root.appendChild(card);const slider=card.querySelector('input');sliders.push(slider);
    slider.addEventListener('input',()=>{q[i]=Number(slider.value);preset='custom';update();});
  });
  MODEL.mimicJointNames.forEach((name,i)=>{
    const leader=MODEL.activeJointNames[MODEL.mimicLeaderIndices[i]];
    const tr=document.createElement('tr');tr.innerHTML=`<td title="${name}">${shortName(name)}</td><td>${MODEL.mimicMultipliers[i].toFixed(3)} × ${shortName(leader)}</td><td id="mimic${i}" class="mimic-value">0</td>`;document.getElementById('mimicBody').appendChild(tr);
  });
  document.getElementById('legend').innerHTML=MODEL.chains.map(c=>`<div class="legend-row"><span class="swatch" style="background:${c.color}"></span>${c.name}</div>`).join('')+'<div class="legend-row"><span class="swatch" style="height:7px;width:7px;border-radius:50%;background:#fff"></span>semantic kp</div>';
}
function setQ(values,name){q=values.slice();preset=name;sliders.forEach((s,i)=>s.value=q[i]);update();}
document.getElementById('openBtn').onclick=()=>setQ(MODEL.lower,'open');
document.getElementById('midBtn').onclick=()=>setQ(MODEL.lower.map((v,i)=>(v+MODEL.upper[i])/2),'mid');
document.getElementById('closeBtn').onclick=()=>setQ(MODEL.upper,'close');
document.getElementById('randomBtn').onclick=()=>setQ(MODEL.lower.map((v,i)=>v+Math.random()*(MODEL.upper[i]-v)),'random');
document.getElementById('labelsToggle').onchange=e=>{showLabels=e.target.checked;draw();};
document.getElementById('originsToggle').onchange=e=>{showOrigins=e.target.checked;draw();};
document.getElementById('gridToggle').onchange=e=>{showGrid=e.target.checked;draw();};

function resize(){const dpr=Math.min(devicePixelRatio||1,2);const rect=canvas.getBoundingClientRect();canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
function cameraBasis(){const cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw);const camera=add(target,scale([cp*cy,cp*sy,sp],distance));const forward=normalize(sub(target,camera));const right=normalize(cross(forward,[0,0,1]));const up=normalize(cross(right,forward));return {camera,forward,right,up};}
function project(point){const b=cameraBasis(),rel=sub(point,b.camera),depth=dot(rel,b.forward);const rect=canvas.getBoundingClientRect(),f=Math.min(rect.width,rect.height)*1.12;return {x:rect.width/2+f*dot(rel,b.right)/depth,y:rect.height/2-f*dot(rel,b.up)/depth,depth};}
function line3(a,b,color,width=1,dash=[]){const pa=project(a),pb=project(b);if(pa.depth<=0||pb.depth<=0)return;ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([]);}
function point3(p,r,fill,stroke='#07111f'){const s=project(p);if(s.depth<=0)return;ctx.beginPath();ctx.arc(s.x,s.y,r,0,Math.PI*2);ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();return s;}
function drawGrid(){if(!showGrid)return;const step=.025,extent=.15;for(let i=-6;i<=6;i++){const c=i===0?'rgba(115,145,174,.35)':'rgba(85,112,140,.14)';line3([i*step,-extent,0],[i*step,extent,0],c);line3([-extent,i*step,0],[extent,i*step,0],c);}line3([0,0,0],[.06,0,0],'#fb7185',2);line3([0,0,0],[0,.06,0],'#6ee7b7',2);line3([0,0,0],[0,0,.06],'#60a5fa',2);[['X',[.064,0,0],'#fb7185'],['Y',[0,.064,0],'#6ee7b7'],['Z',[0,0,.064],'#60a5fa']].forEach(([label,p,c])=>{const s=project(p);ctx.fillStyle=c;ctx.font='bold 11px ui-monospace';ctx.fillText(label,s.x,s.y);});}
function draw(){
  const rect=canvas.getBoundingClientRect();ctx.clearRect(0,0,rect.width,rect.height);drawGrid();const state=forward(q);
  MODEL.chains.forEach(chain=>{for(let i=1;i<chain.indices.length;i++)line3(state.keypoints[chain.indices[i-1]],state.keypoints[chain.indices[i]],chain.color,4);});
  if(showOrigins){Object.entries(state.links).forEach(([name,pose])=>{const s=project(pose.t);ctx.save();ctx.translate(s.x,s.y);ctx.rotate(Math.PI/4);ctx.fillStyle=name.includes('distal')?'#f0abfc':'#7894af';ctx.fillRect(-3,-3,6,6);ctx.restore();});}
  const order=state.keypoints.map((p,i)=>({p,i,s:project(p)})).sort((a,b)=>b.s.depth-a.s.depth);
  order.forEach(({p,i,s})=>{point3(p,i===0?6:4.5,i===0?'#f97316':'#f8fafc');if(showLabels){ctx.fillStyle='#c5d5e6';ctx.font='9px ui-monospace';ctx.fillText(MODEL.keypointNames[i],s.x+7,s.y-6);}});
}
function update(){
  const state=forward(q);sliders.forEach((_,i)=>{document.getElementById(`value${i}`).textContent=`${q[i].toFixed(3)} rad · ${(q[i]*180/Math.PI).toFixed(1)}°`;});
  state.mimic.forEach((v,i)=>document.getElementById(`mimic${i}`).textContent=v.toFixed(4));
  const finite=state.keypoints.flat().every(Number.isFinite);document.getElementById('finiteHud').textContent=`finite: ${finite}`;document.getElementById('finiteHud').style.color=finite?'#6ee7b7':'#fb7185';document.getElementById('selectedHud').textContent=`preset: ${preset}`;draw();
}
canvas.addEventListener('pointerdown',e=>{dragging=true;panMode=e.shiftKey;lastX=e.clientX;lastY=e.clientY;canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;if(panMode){const b=cameraBasis(),amount=distance/700;target=add(target,add(scale(b.right,-dx*amount),scale(b.up,dy*amount)));}else{yaw-=dx*.008;pitch=Math.max(-1.35,Math.min(1.35,pitch+dy*.008));}draw();});
canvas.addEventListener('pointerup',()=>{dragging=false;canvas.classList.remove('dragging');});
canvas.addEventListener('wheel',e=>{e.preventDefault();distance=Math.max(MODEL.viewSpan*.75,Math.min(MODEL.viewSpan*8,distance*Math.exp(e.deltaY*.001)));draw();},{passive:false});
window.addEventListener('resize',resize);
createControls();resize();update();
</script>
</body>
</html>
'''


def build_html(output_path: str | Path = DEFAULT_OUTPUT) -> tuple[Path, float]:
    """Generate the standalone HTML and return its path and FK check error."""
    output = Path(output_path).expanduser().resolve()
    fk = Revo2Kinematics()
    model = _serialized_model(fk)
    max_error = _validate_serialization(fk, model)
    html = HTML_TEMPLATE.replace(
        "__MODEL_JSON__",
        json.dumps(model, ensure_ascii=False, separators=(",", ":")),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output, max_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone interactive Revo2 keypoint/mimic viewer."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output HTML (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, max_error = build_html(args.output)
    print(f"HTML: {output}")
    print("active joints: 6")
    print("mimic joints: 5")
    print("semantic keypoints: 21")
    print(f"serialized FK max error: {max_error:.3e} m")
    print("standalone: yes (no external JavaScript or asset files)")


if __name__ == "__main__":
    main()
