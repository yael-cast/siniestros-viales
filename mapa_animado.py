"""
mapa_animado.py

"""
import json

import pandas as pd

PLANTILLA = r"""
<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<link href="https://cdn.jsdelivr.net/npm/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<script src="https://cdn.jsdelivr.net/npm/deck.gl@8.9.35/dist.min.js"></script>
<style>
  html,body{margin:0;height:100%;background:#0b0d14;font-family:Segoe UI,Arial,sans-serif;color:#e8eaf0}
  #map{position:absolute;inset:0}
  .panel{position:absolute;left:14px;top:14px;width:250px;padding:14px;border-radius:10px;
         background:rgba(11,13,20,.78);backdrop-filter:blur(4px);z-index:5;font-size:13px}
  .panel h1{font-size:19px;margin:0 0 4px}
  .sub{color:#9aa3b5;font-size:12px;margin-bottom:10px}
  .reloj{font-size:44px;font-weight:300;line-height:1}
  .cifras{display:flex;gap:18px;margin:8px 0 10px}
  .cifras b{font-size:20px;display:block}
  .cifras span{font-size:10px;color:#9aa3b5;letter-spacing:.06em}
  label{display:flex;align-items:center;gap:8px;margin:5px 0;cursor:pointer}
  .dot{width:11px;height:11px;border-radius:50%;display:inline-block}
  select{background:#1b1f2e;color:#e8eaf0;border:1px solid #333a50;border-radius:5px;padding:2px 4px}
  .barra{position:absolute;left:0;right:0;bottom:0;height:120px;z-index:5;
         background:linear-gradient(transparent,rgba(11,13,20,.92) 45%);display:flex;align-items:flex-end;padding:0 14px 8px}
  #play{width:38px;height:38px;border-radius:50%;border:1px solid #444c66;background:#1b1f2e;color:#fff;
        font-size:15px;cursor:pointer;margin-right:12px;flex:none}
  #vel{margin-right:14px;flex:none}
  #hist{flex:1;display:flex;align-items:flex-end;gap:3px;height:80px;position:relative}
  .col{flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%;cursor:pointer;position:relative}
  .col .g{background:#f0a028}.col .f{background:#dc2626}
  .col.act{outline:1px solid #fff}
  .col i{position:absolute;bottom:-14px;left:0;right:0;text-align:center;font-style:normal;font-size:9px;color:#8b93a7}
  .deck-tooltip{font-size:12px!important;border-radius:6px!important}
</style></head><body>
<div id="map"></div>
<div class="panel">
  <h1>Un día de siniestros graves</h1>
  <div class="sub">CABA, todos los años (__DESDE__-__HASTA__) superpuestos según la hora del día.</div>
  <div class="reloj" id="reloj">00:00</div>
  <div class="cifras">
    <div><b id="n-hora">0</b><span>EN ESTA HORA</span></div>
    <div><b id="n-acum">0</b><span>ACUMULADOS</span></div>
  </div>
  <label><input type="checkbox" id="c-f" checked><span class="dot" style="background:#dc2626"></span>Fatales <span id="t-f" style="margin-left:auto;color:#9aa3b5"></span></label>
  <label><input type="checkbox" id="c-g" checked><span class="dot" style="background:#f0a028"></span>Graves <span id="t-g" style="margin-left:auto;color:#9aa3b5"></span></label>
  <label><input type="checkbox" id="c-hex" checked>Hexágonos de densidad</label>
  <label><input type="checkbox" id="c-acum" checked>Acumular durante el día</label>
  <label>Modo: <select id="s-modo"></select></label>
</div>
<div class="barra">
  <button id="play" title="Reproducir / pausar">&#10074;&#10074;</button>
  <select id="vel"><option value="1">1 h/s</option><option value="2" selected>2 h/s</option><option value="4">4 h/s</option></select>
  <div id="hist"></div>
</div>
<script>
// [lon, lat, hora, esFatal, indiceModo, direccion, fecha]
const DATA = __DATOS__;
const MODOS = __MODOS__;
let t = 7, playing = true, vel = 2, ultimaHora = -1;
const $ = id => document.getElementById(id);

// Selector de modo
$("s-modo").innerHTML = '<option value="-1">Todos</option>' + MODOS.map((m,i)=>`<option value="${i}">${m}</option>`).join("");

function visibles(){
  const f=$("c-f").checked, g=$("c-g").checked, m=+$("s-modo").value;
  return DATA.filter(d => (d[3]?f:g) && (m<0 || d[4]===m));
}

// Histograma por hora (barras apiladas grave/fatal)
function dibujarHist(){
  const base = visibles(), g=Array(24).fill(0), f=Array(24).fill(0);
  base.forEach(d => (d[3]?f:g)[d[2]]++);
  const max = Math.max(1, ...g.map((v,i)=>v+f[i]));
  $("hist").innerHTML = g.map((v,i)=>
    `<div class="col" data-h="${i}"><div class="f" style="height:${f[i]/max*100}%"></div>
     <div class="g" style="height:${v/max*100}%"></div><i>${i%3===0?i+"h":""}</i></div>`).join("");
  document.querySelectorAll(".col").forEach(c => c.onclick = () => { t = +c.dataset.h; ultimaHora=-1; actualizar(); });
  $("t-f").textContent = f.reduce((a,b)=>a+b,0).toLocaleString("es-AR");
  $("t-g").textContent = g.reduce((a,b)=>a+b,0).toLocaleString("es-AR");
}

const deckgl = new deck.DeckGL({
  container: "map",
  mapStyle: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  initialViewState: {latitude:-34.615, longitude:-58.44, zoom:10.8, pitch:0},
  controller: true,
  getTooltip: ({object}) => {
    if(!object) return null;
    if(object.points) return {html:`<b>${object.points.length}</b> siniestros en esta zona`, style:{background:"#111",color:"#fff"}};
    return {html:`<b>${object[3]?"Fatal":"Grave"}</b><br>${object[5]}<br>${object[6]} &middot; ${MODOS[object[4]]}`,
            style:{background:"#111",color:"#fff"}};
  },
  layers: []
});

function actualizar(){
  const h = Math.floor(t);
  $("reloj").textContent = String(h).padStart(2,"0")+":"+String(Math.floor((t-h)*60)).padStart(2,"0");
  document.querySelectorAll(".col").forEach(c => c.classList.toggle("act", +c.dataset.h===h));
  if(h === ultimaHora) return;           // solo se redibuja al cambiar de hora
  ultimaHora = h;
  const acum = $("c-acum").checked, base = visibles();
  const hasta = base.filter(d => d[2] <= h), ahora = base.filter(d => d[2] === h);
  $("n-hora").textContent = ahora.length.toLocaleString("es-AR");
  $("n-acum").textContent = hasta.length.toLocaleString("es-AR");
  const datos = acum ? hasta : ahora;
  const color = d => d[3] ? [220,38,38,230] : [240,160,40,200];
  deckgl.setProps({layers: [
    new deck.HexagonLayer({id:"hex", data:datos, visible:$("c-hex").checked, getPosition:d=>[d[0],d[1]],
      radius:450, coverage:0.92, extruded:false, opacity:0.55, pickable:true,
      colorRange:[[90,20,70],[140,30,90],[190,50,110],[230,90,130],[255,150,170],[255,220,225]]}),
    new deck.ScatterplotLayer({id:"pts", data:datos, getPosition:d=>[d[0],d[1]], getFillColor:color,
      getRadius:d => d[2]===h ? 160 : 60, radiusMinPixels:2.5, radiusMaxPixels:14, pickable:true,
      updateTriggers:{getRadius:[h]}})
  ]});
}

// Animación
let previo = performance.now();
function bucle(ahora){
  const dt = (ahora - previo)/1000; previo = ahora;
  if(playing){ t += dt*vel; if(t >= 24) t -= 24; actualizar(); }
  requestAnimationFrame(bucle);
}
$("play").onclick = () => { playing = !playing; $("play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;"; };
$("vel").onchange = e => vel = +e.target.value;
["c-f","c-g","c-hex","c-acum","s-modo"].forEach(id => $(id).onchange = () => { dibujarHist(); ultimaHora=-1; actualizar(); });

dibujarHist(); actualizar(); requestAnimationFrame(bucle);
</script></body></html>
"""


def html_mapa_animado(df: pd.DataFrame) -> str:
    """
    df necesita las columnas: lat, lon, hora (0-23), gravedad ('GRAVE'/'MORTAL'),
    modo, direccion y fecha (datetime). Devuelve el HTML listo para incrustar.
    """
    d = df.dropna(subset=["lat", "lon", "hora"]).copy()
    d["hora"] = d["hora"].astype(int).clip(0, 23)
    d["modo"] = d["modo"].fillna("Sin dato")
    d["direccion"] = d["direccion"].fillna("Sin dato")
    d["fecha_txt"] = pd.to_datetime(d["fecha"]).dt.strftime("%d/%m/%Y").fillna("Sin dato")
    modos = sorted(d["modo"].unique())
    idx = {m: i for i, m in enumerate(modos)}
    filas = [[round(r.lon, 5), round(r.lat, 5), int(r.hora), int(r.gravedad == "MORTAL"),
              idx[r.modo], r.direccion, r.fecha_txt] for r in d.itertuples()]
    anios = pd.to_datetime(d["fecha"]).dt.year
    return (PLANTILLA.replace("__DATOS__", json.dumps(filas, ensure_ascii=False))
            .replace("__MODOS__", json.dumps(modos, ensure_ascii=False))
            .replace("__DESDE__", str(int(anios.min())))
            .replace("__HASTA__", str(int(anios.max()))))
