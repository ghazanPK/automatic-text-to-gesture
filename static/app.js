import {createStage} from './avatar.js';
import {Speech} from './speech.js';
const stage=createStage(document.querySelector('#stage'));stage.camera.position.set(0,1.5,3.3);stage.camera.lookAt(0,.95,0);
const speech=new Speech(stage);
const $=s=>document.querySelector(s);
const mode=document.body.dataset.mode;
if(mode==='wild'||mode==='multilingual'){$('#threshold-wrap').hidden=true;$('#seed-wrap').hidden=false;}
if(mode==='multilingual'){$('#language-wrap').hidden=false;$('#query').value='결과를 보여 주세요';}
if(mode==='ridge'){$('#threshold').value='0.72';$('#threshold-value').textContent='0.72';$('#query').value='open both hands now';}
$('#threshold').addEventListener('input',()=>$('#threshold-value').textContent=$('#threshold').value);
let data=null,frames=[],playing=true,frame=0,last=performance.now();
const edgeNames=[['Hips','Neck'],['Neck','Head'],['Neck','LeftShoulder'],['LeftShoulder','LeftArm'],['LeftArm','LeftForeArm'],['LeftForeArm','LeftHand'],['Neck','RightShoulder'],['RightShoulder','RightArm'],['RightArm','RightForeArm'],['RightForeArm','RightHand']];
function allFrames(){return frames;}
function draw(){
  const frames=allFrames();if(!frames.length)return;frame=Math.max(0,Math.min(frames.length-1,frame));
  const current=frames[frame],names=data.joint_order;
  const neck=names.indexOf('Neck');const origin=current.joints[Math.max(0,neck)]||[0,0,0];
  let centered=current.joints.map(p=>[p[0]-origin[0],p[1]-origin[1],p[2]-origin[2]]);
  const scale=data.motionScale||1;
  centered=centered.map(p=>[p[0]*scale,1.35+p[1]*scale,p[2]*scale]);
  const edges=edgeNames.filter(([a,b])=>names.includes(a)&&names.includes(b)).map(([a,b])=>[names.indexOf(a),names.indexOf(b)]);
  if (!stage.setPosePositions?.(centered,names)) stage.setSkeleton(centered,edges);
  $('#scrub').value=frame;$('#frame-label').textContent=`Frame ${frame+1}/${frames.length} · ${current.slot.gesture_id} · ${current.slot.text||''}`;
}
let lastRig=null;function tick(now){if(data&&!playing&&stage.avatar.rig!==lastRig){lastRig=stage.avatar.rig;draw();}if(playing&&data){const fps=data.fps||15;const delta=Math.floor((now-last)/(1000/fps));if(delta){frame=(frame+delta)%allFrames().length;last=now;draw();}}requestAnimationFrame(tick);}requestAnimationFrame(tick);
$('#play').onclick=()=>{playing=!playing;$('#play').textContent=playing?'Pause':'Play';last=performance.now();};
$('#scrub').oninput=()=>{frame=Number($('#scrub').value);draw();};
$('#speak').onclick=async()=>{try{await speech.speak($('#query').value,{backend:$('#speech-backend').value});}catch(e){$('#status').textContent=e.message;}};
$('#audio-input').onchange=async e=>{const file=e.target.files?.[0];if(!file)return;try{const result=await speech.transcribe(file);$('#query').value=result.text;$('#status').textContent=`Transcribed with ${result.backend}.`;}catch(error){$('#status').textContent=error.message;}};
$('#form').onsubmit=async e=>{
  e.preventDefault();$('#status').textContent='Retrieving…';
  const params=new URLSearchParams({text:$('#query').value,threshold:$('#threshold').value,seed:$('#seed').value,language:$('#language').value});
  try{
    const response=await fetch('/api/query?'+params);const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Retrieval failed');
    data=result;frames=result.slots.flatMap((slot,index)=>slot.frames.map((joints,i)=>({joints,slot,index,i})));frame=0;last=performance.now();
    const neckIndex=Math.max(0,result.joint_order.indexOf('Neck'));
    const extent=Math.max(.001,...allFrames().flatMap(({joints})=>{
      const origin=joints[neckIndex]||[0,0,0];
      return joints.flatMap(point=>point.map((value,axis)=>Math.abs(value-origin[axis])));
    }));
    data.motionScale=Math.min(1.5,1.5/extent);
    $('#scrub').max=Math.max(0,allFrames().length-1);draw();
    const pieces=[result.algorithm,`${result.slots.length} clip(s)`,result.data_label];
    if(result.rule_count!==undefined)pieces.push(`rules: ${result.rule_count}`);
    if(result.fallback_count!==undefined)pieces.push(`fallback: ${result.fallback_count}`);
    if(result.english_text)pieces.push(`English: ${result.english_text}`);
    $('#summary').textContent=pieces.filter(Boolean).join(' · ');
    const table=document.createElement('table'),head=document.createElement('tr');
    ['Text','Gesture','Route / cluster','Score'].forEach(t=>{const th=document.createElement('th');th.textContent=t;head.append(th);});table.append(head);
    result.trace.forEach(row=>{const tr=document.createElement('tr');[row.text||row.english_text,row.gesture_id,row.source||(`cluster ${row.cluster_id??'—'}`),Number(row.similarity).toFixed(3)].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});table.append(tr);});
    $('#trace').replaceChildren(table);$('#status').textContent='Motion loaded.';
  }catch(error){$('#status').textContent=error.message;}
};
