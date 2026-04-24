"""
Graph visualization — 3D interactive HTML, force-directed.
All packets float freely — no anchors.
Filter buttons to highlight by type.
"""

import json
import os
import numpy as np

os.makedirs("data/graphs", exist_ok=True)


def _build_graph_data(graph):
    phi          = graph.get_phi()
    max_phi      = max(phi.values(), default=1.0)
    max_pressure = max((p.get_pressure() for p in graph.packets), default=1.0)
    weights      = graph.get_connection_weights()
    max_w        = max((w for _, _, w in weights), default=1.0)

    joint_set  = set(graph.joint_entry_idxs)
    vision_set = set(graph.vision_entry_idxs)
    output_set = set(graph.action_output_idxs)

    joint_names = [
        "FR hip","FR thigh","FR calf",
        "FL hip","FL thigh","FL calf",
        "RR hip","RR thigh","RR calf",
        "RL hip","RL thigh","RL calf",
    ]

    rng = np.random.default_rng(42)
    n   = len(graph.packets)

    nodes = []
    for idx in range(n):
        # spread wider so nodes dont start piled up
        x = float(rng.uniform(-18, 18))
        y = float(rng.uniform(-18, 18))
        z = float(rng.uniform(-14, 14))

        pressure = graph.packets[idx].get_pressure()
        norm_p   = pressure / max(max_pressure, 1e-6)
        size     = 0.10 + norm_p * 0.28
        frozen   = graph.packets[idx].frozen
        phi_norm = phi.get(idx, 0.0) / max(max_phi, 1e-6)

        if idx in joint_set:
            rank  = graph.joint_entry_idxs.index(idx)
            role  = "joint"
            base  = (0.27, 0.53, 1.0)
            label = joint_names[rank] if rank < len(joint_names) else f"j{rank}"
        elif idx in vision_set:
            i     = graph.vision_entry_idxs.index(idx)
            role  = "vision"
            base  = (0.8, 0.67, 0.0)
            label = f"px{i}"
        elif idx in output_set:
            rank  = graph.action_output_idxs.index(idx)
            role  = "actuator"
            base  = (1.0, 0.27, 0.27)
            label = joint_names[rank] if rank < len(joint_names) else f"out{rank}"
        else:
            role  = "internal"
            base  = (0.53, 0.53, 0.53)
            label = f"p{idx}"

        if frozen:
            label += " ❄"

        # base color blended toward cyan by phi
        cyan  = (0.0, 1.0, 1.0)
        r = base[0] * (1 - phi_norm) + cyan[0] * phi_norm
        g = base[1] * (1 - phi_norm) + cyan[1] * phi_norm
        b = base[2] * (1 - phi_norm) + cyan[2] * phi_norm
        color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

        # vision pixels excluded from force sim — too many, collapse the sim
        static = (role == 'vision')
        nodes.append({
            "id": idx, "x": x, "y": y, "z": z,
            "size": size, "color": color, "label": label,
            "frozen": frozen, "role": role,
            "pressure": round(norm_p, 3),
            "phi": round(phi_norm, 3),
            "static": static,
        })

    edges = []
    for src, tgt, w in weights:
        if src >= n or tgt >= n:
            continue
        norm_w = w / max(max_w, 1e-6)
        edges.append({"src": src, "tgt": tgt, "weight": round(norm_w, 3)})

    return nodes, edges


def _html(nodes, edges, episode, n_packets, n_connections):
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Packet Graph — Episode {episode}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a0a; overflow:hidden; font-family:monospace; color:white; position:relative; }}
#info {{
  position:absolute; top:12px; left:50%; transform:translateX(-50%);
  font-size:13px; text-align:center;
  background:rgba(0,0,0,0.5); padding:6px 16px; border-radius:6px;
  pointer-events:none;
}}
#filters {{
  position:absolute; top:52px; left:50%; transform:translateX(-50%);
  display:flex; gap:6px; flex-wrap:wrap; justify-content:center;
}}
#filters button {{
  background:rgba(255,255,255,0.07); color:#ccc;
  border:1px solid #333; border-radius:4px;
  padding:4px 10px; font-size:11px; cursor:pointer;
  font-family:monospace; transition:all 0.15s;
}}
#filters button:hover {{ background:rgba(255,255,255,0.15); color:white; }}
#filters button.active {{ border-color:#00ffff; color:#00ffff; background:rgba(0,255,255,0.1); }}
#tooltip {{
  position:absolute; display:none;
  background:rgba(0,0,0,0.85); color:white;
  font-size:11px; padding:6px 10px; border-radius:4px;
  pointer-events:none; border:1px solid #333;
}}
#controls {{
  position:absolute; top:12px; right:16px;
  color:#aaa; font-size:10px; text-align:right;
  background:rgba(0,0,0,0.5); padding:6px 12px; border-radius:6px;
}}
</style>
</head>
<body>
<div id="info">
  Episode {episode} &nbsp;|&nbsp; Packets: {n_packets} &nbsp;|&nbsp; Connections: {n_connections}
</div>
<div id="filters">
  <button class="active" onclick="setFilter('all')">All</button>
  <button onclick="setFilter('frozen')">❄ Frozen</button>
  <button onclick="setFilter('phi')">Φ Consciousness</button>
  <button onclick="setFilter('pressure')">Pressure</button>
  <button onclick="setFilter('boundary')">World Boundary</button>
  <button onclick="setFilter('joint')">Joint Sensors</button>
  <button onclick="setFilter('vision')">Vision</button>
  <button onclick="setFilter('actuator')">Actuators</button>
  <button onclick="setFilter('internal')">Internal</button>
</div>
<div id="tooltip"></div>
<div id="controls">
  Left drag — rotate<br>
  Right drag — pan<br>
  Scroll — zoom<br>
  Hover — inspect<br>
  Space — toggle sim
</div>
<canvas id="c"></canvas>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const NODES = {nodes_json};
const EDGES = {edges_json};

// ── scene ─────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true}});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0a0a0a);
const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 1000);
camera.position.set(0, 0, 40);
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(5,10,10); scene.add(dl);

// ── positions & velocities — all free ────────────────────
const pos = {{}}, vel = {{}}, isStatic = {{}};
NODES.forEach(n => {{
  pos[n.id]      = {{x:n.x, y:n.y, z:n.z}};
  vel[n.id]      = {{x:0,   y:0,   z:0  }};
  isStatic[n.id] = n.static || false;
}});

// ── meshes ────────────────────────────────────────────────
const nodeMap = {{}}, nodeMesh = [];
const baseColors = {{}};   // store original color per node

NODES.forEach(n => {{
  const geo = new THREE.SphereGeometry(n.size, 10, 10);
  const mat = new THREE.MeshPhongMaterial({{
    color: n.color, emissive: n.color,
    emissiveIntensity: n.frozen ? 0.1 : 0.4,
    transparent: true, opacity: 0.92
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(n.x, n.y, n.z);
  mesh.userData = n;
  scene.add(mesh);
  nodeMap[n.id] = mesh;
  nodeMesh.push(mesh);
  baseColors[n.id] = n.color;
}});

// ── edges ─────────────────────────────────────────────────
const edgeObjs = [];
EDGES.forEach(e => {{
  const geo = new THREE.BufferGeometry();
  const arr = new Float32Array(6);
  geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  const mat = new THREE.LineBasicMaterial({{
    color: new THREE.Color(e.weight*0.4, e.weight*0.3, 1.0),
    transparent: true, opacity: 0.05 + e.weight*0.55
  }});
  const line = new THREE.Line(geo, mat);
  scene.add(line);
  edgeObjs.push({{line, src:e.src, tgt:e.tgt, w:e.weight}});
}});

// ── force simulation — all free ───────────────────────────
const REP=9.0, ATT=0.10, DAMP=0.82;

function forceStep() {{
  const ids = Object.keys(pos).map(Number).filter(id => !isStatic[id]);
  for (let i=0;i<ids.length;i++) {{
    const a=ids[i];
    for (let j=i+1;j<ids.length;j++) {{
      const b=ids[j];
      const dx=pos[a].x-pos[b].x, dy=pos[a].y-pos[b].y, dz=pos[a].z-pos[b].z;
      const d2=dx*dx+dy*dy+dz*dz+0.1;
      const f=REP/d2;
      vel[a].x+=dx*f; vel[a].y+=dy*f; vel[a].z+=dz*f;
      vel[b].x-=dx*f; vel[b].y-=dy*f; vel[b].z-=dz*f;
    }}
  }}
  edgeObjs.forEach(e => {{
    const a=e.src, b=e.tgt;
    if (!pos[a]||!pos[b]) return;
    if (isStatic[a] && isStatic[b]) return;
    const dx=pos[b].x-pos[a].x, dy=pos[b].y-pos[a].y, dz=pos[b].z-pos[a].z;
    const d=Math.sqrt(dx*dx+dy*dy+dz*dz)+0.01;
    const f=ATT*e.w*d;
    if (!isStatic[a]){{vel[a].x+=dx/d*f; vel[a].y+=dy/d*f; vel[a].z+=dz/d*f;}}
    if (!isStatic[b]){{vel[b].x-=dx/d*f; vel[b].y-=dy/d*f; vel[b].z-=dz/d*f;}}
  }});
  ids.forEach(id => {{
    vel[id].x*=DAMP; vel[id].y*=DAMP; vel[id].z*=DAMP;
    pos[id].x+=vel[id].x; pos[id].y+=vel[id].y; pos[id].z+=vel[id].z;
    pos[id].x=Math.max(-20,Math.min(20,pos[id].x));
    pos[id].y=Math.max(-20,Math.min(20,pos[id].y));
    pos[id].z=Math.max(-16,Math.min(16,pos[id].z));
  }});
}}

for (let i=0;i<30;i++) forceStep();
let simRunning=false;

// ── filter system ─────────────────────────────────────────
let currentFilter='all';

function setFilter(f) {{
  currentFilter=f;
  document.querySelectorAll('#filters button').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');

  NODES.forEach(n => {{
    const mesh = nodeMap[n.id];
    let show = false;
    switch(f) {{
      case 'all':      show=true; break;
      case 'frozen':   show=n.frozen; break;
      case 'phi':      show=n.phi>0.3; break;
      case 'pressure': show=n.pressure>0.3; break;
      case 'boundary': show=(n.role==='joint'||n.role==='vision'||n.role==='actuator'); break;
      case 'joint':    show=n.role==='joint'; break;
      case 'vision':   show=n.role==='vision'; break;
      case 'actuator': show=n.role==='actuator'; break;
      case 'internal': show=n.role==='internal'; break;
    }}
    mesh.material.opacity   = show ? 0.92 : 0.04;
    mesh.material.emissiveIntensity = show ? 0.4 : 0.0;
  }});

  edgeObjs.forEach(e => {{
    const sa=NODES.find(n=>n.id===e.src), ta=NODES.find(n=>n.id===e.tgt);
    if (!sa||!ta) return;
    let showSrc=false, showTgt=false;
    [sa,ta].forEach((n,i) => {{
      let s=false;
      switch(f){{
        case 'all': s=true; break;
        case 'frozen': s=n.frozen; break;
        case 'phi': s=n.phi>0.3; break;
        case 'pressure': s=n.pressure>0.3; break;
        case 'boundary': s=(n.role==='joint'||n.role==='vision'||n.role==='actuator'); break;
        case 'joint': s=n.role==='joint'; break;
        case 'vision': s=n.role==='vision'; break;
        case 'actuator': s=n.role==='actuator'; break;
        case 'internal': s=n.role==='internal'; break;
      }}
      if (i===0) showSrc=s; else showTgt=s;
    }});
    e.line.material.opacity=(showSrc||showTgt) ? 0.05+e.w*0.55 : 0.01;
  }});
}}

// ── orbit controls ────────────────────────────────────────
let dragging=false,rightDrag=false,lx=0,ly=0;
const tgt=new THREE.Vector3();
let sph=new THREE.Spherical().setFromVector3(camera.position.clone().sub(tgt));

document.addEventListener('mousedown',e=>{{dragging=true;rightDrag=e.button===2;lx=e.clientX;ly=e.clientY;}});
document.addEventListener('mouseup',()=>dragging=false);
document.addEventListener('contextmenu',e=>e.preventDefault());
document.addEventListener('mousemove',e=>{{
  if(!dragging)return;
  const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
  if(rightDrag){{
    const r=new THREE.Vector3(),u=new THREE.Vector3();
    camera.matrix.extractBasis(r,u,new THREE.Vector3());
    tgt.addScaledVector(r,-dx*0.02);tgt.addScaledVector(u,dy*0.02);
  }}else{{
    sph.theta-=dx*0.005;sph.phi-=dy*0.005;
    sph.phi=Math.max(0.1,Math.min(Math.PI-0.1,sph.phi));
  }}
  camera.position.copy(tgt).add(new THREE.Vector3().setFromSpherical(sph));
  camera.lookAt(tgt);
}});
document.addEventListener('wheel',e=>{{
  sph.radius*=(1+e.deltaY*0.001);
  sph.radius=Math.max(2,Math.min(120,sph.radius));
  camera.position.copy(tgt).add(new THREE.Vector3().setFromSpherical(sph));
  camera.lookAt(tgt);
}});
document.addEventListener('keydown',e=>{{
  if(e.code==='Space'){{simRunning=!simRunning;e.preventDefault();}}
}});

// ── tooltip ───────────────────────────────────────────────
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
const tt=document.getElementById('tooltip');
document.addEventListener('mousemove',e=>{{
  mouse.x=(e.clientX/window.innerWidth)*2-1;
  mouse.y=-(e.clientY/window.innerHeight)*2+1;
  ray.setFromCamera(mouse,camera);
  const hits=ray.intersectObjects(nodeMesh);
  if(hits.length>0){{
    const d=hits[0].object.userData;
    tt.style.display='block';
    tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY-10)+'px';
    tt.innerHTML=`<b>${{d.label}}</b><br>role: ${{d.role}}<br>pressure: ${{d.pressure}}<br>Φ: ${{d.phi}}<br>frozen: ${{d.frozen}}`;
  }}else tt.style.display='none';
}});

window.addEventListener('resize',()=>{{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
}});

// ── render loop ───────────────────────────────────────────
function animate(){{
  requestAnimationFrame(animate);
  if(simRunning) forceStep();
  NODES.forEach(n=>{{const p=pos[n.id];if(p)nodeMap[n.id].position.set(p.x,p.y,p.z);}});
  edgeObjs.forEach(e=>{{
    const pa=pos[e.src],pb=pos[e.tgt];if(!pa||!pb)return;
    const arr=e.line.geometry.attributes.position.array;
    arr[0]=pa.x;arr[1]=pa.y;arr[2]=pa.z;
    arr[3]=pb.x;arr[4]=pb.y;arr[5]=pb.z;
    e.line.geometry.attributes.position.needsUpdate=true;
  }});
  renderer.render(scene,camera);
}}
animate();
</script>
</body>
</html>"""


class GraphViz:

    def __init__(self):
        self._episode = 0

    def update(self, graph):
        self._episode += 1
        nodes, edges = _build_graph_data(graph)

        html = _html(
            nodes, edges,
            episode       = self._episode,
            n_packets     = len(graph.packets),
            n_connections = len(graph.connections)
        )

        ep_path     = f"data/graphs/graph_ep{self._episode:04d}.html"
        latest_path = "data/graph_latest.html"

        with open(ep_path,     "w") as f: f.write(html)
        with open(latest_path, "w") as f: f.write(html)

        print(f"  viz → {latest_path}")

    def save_final(self, path="data/final_graph.html"):
        import shutil
        latest = "data/graph_latest.html"
        if os.path.exists(latest):
            shutil.copy(latest, path)