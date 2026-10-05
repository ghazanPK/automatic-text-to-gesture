import {createStage} from './avatar.js?v=20261006-paper1';
import {Speech} from './speech.js?v=20261006-paper1';
import {MotionSequence,loadGestureLibrary} from './gesture-library.js?v=20261006-paper1';
const stage=createStage(document.querySelector('#stage'));stage.camera.position.set(0,1.5,3.3);stage.camera.lookAt(0,.95,0);
const speech=new Speech(stage);
const $=s=>document.querySelector(s);
const mode=document.body.dataset.mode;
if(mode==='wild'||mode==='multilingual'){$('#threshold-wrap').hidden=true;$('#seed-wrap').hidden=false;}
if(mode==='multilingual'){$('#language-wrap').hidden=false;$('#query').value='결과를 보여 주세요';}
if(mode==='ridge'){$('#threshold').value='0.72';$('#threshold-value').textContent='0.72';$('#query').value='open both hands now';}
$('#threshold').addEventListener('input',()=>$('#threshold-value').textContent=$('#threshold').value);
let data=null,frames=[],playing=false,frame=0,last=performance.now(),muted=false;
const motion=new MotionSequence(stage);
const libraryPanel=document.createElement('section');
libraryPanel.setAttribute('aria-label','Prepared gesture library');
libraryPanel.style.cssText='margin:1rem 0;padding:.8rem;border:1px solid #405c77;border-radius:10px';
$('#form').after(libraryPanel);
function formatValue(value){
  if(typeof value==='number')return Number.isInteger(value)?String(value):value.toFixed(3);
  if(typeof value==='string')return value;
  if(value&&typeof value==='object'&&!Array.isArray(value)){
    const parts=Object.entries(value).filter(([,v])=>typeof v==='number'||typeof v==='string').map(([k,v])=>`${k}=${formatValue(v)}`);
    return parts.length?parts.join(', '):null;
  }
  return null;
}
function formatMetrics(metrics){
  return Object.entries(metrics||{}).map(([key,value])=>[key,formatValue(value)]).filter(([,value])=>value!==null).map(([key,value])=>`${key.replaceAll('_',' ')}: ${value}`);
}
function renderLibrary(library){
  libraryPanel.replaceChildren();
  const title=document.createElement('strong');title.textContent='Prepared BEAT gesture bank';libraryPanel.append(title);
  if(!library){const note=document.createElement('p');note.className='small';note.textContent='Local BEAT bank unavailable. The bundled authored starter remains available.';libraryPanel.append(note);return;}
  const count=document.createElement('p');count.className='small';
  const metrics=formatMetrics(library.metrics);
  count.textContent=`${library.clips.length} prepared clips${metrics.length?' · '+metrics.join(' · '):''} · Select an utterance to see its sequence and retrieval route.`;libraryPanel.append(count);
  if(mode==='automatic'&&Number.isFinite(Number(library.default_threshold))&&library.default_threshold!==null){
    $('#threshold').value=String(library.default_threshold);$('#threshold-value').textContent=Number(library.default_threshold).toFixed(2);
    if(library.threshold_rule){const note=document.createElement('p');note.className='small';note.textContent=`Default threshold: ${library.threshold_rule}.`;libraryPanel.append(note);}
  }
  const examples=document.createElement('div');examples.className='controls';examples.style.cssText='max-height:18rem;overflow:auto';
  for(const item of library.suggested_queries||[]){
    const query=typeof item==='string'?item:item.text||item.query;
    if(!query)continue;
    const button=document.createElement('button');button.type='button';button.className='secondary';button.textContent=query;
    button.onclick=()=>{$('#query').value=query;if(mode==='multilingual')$('#language').value=/[\uac00-\ud7af]/.test(query)?'ko':'en';$('#form').requestSubmit();};examples.append(button);
  }
  libraryPanel.append(examples);
  const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='View source clip transcripts';details.append(summary);
  const list=document.createElement('ol');for(const clip of library.clips){const row=document.createElement('li');row.textContent=`${clip.id}: ${clip.text||'Aligned transcript unavailable'}`;list.append(row);}details.append(list);libraryPanel.append(details);
}
const edgeNames=[['Hips','Neck'],['Neck','Head'],['Neck','LeftShoulder'],['LeftShoulder','LeftArm'],['LeftArm','LeftForeArm'],['LeftForeArm','LeftHand'],['Neck','RightShoulder'],['RightShoulder','RightArm'],['RightArm','RightForeArm'],['RightForeArm','RightHand']];
function allFrames(){return frames;}
function draw(){
  const frames=allFrames();if(!frames.length)return;frame=Math.max(0,Math.min(frames.length-1,frame));
  const current=frames[frame],names=data.joint_order;
  // Scrubbing uses the source frame directly; speech playback follows text spans.
  const neck=names.indexOf('Neck');const origin=current.joints[Math.max(0,neck)]||[0,0,0];
  let centered=current.joints.map(p=>[p[0]-origin[0],p[1]-origin[1],p[2]-origin[2]]);
  const scale=data.motionScale||1;
  centered=centered.map(p=>[p[0]*scale,1.35+p[1]*scale,p[2]*scale]);
  const edges=edgeNames.filter(([a,b])=>names.includes(a)&&names.includes(b)).map(([a,b])=>[names.indexOf(a),names.indexOf(b)]);
  if (stage.setPosePositions?.(centered,names,stage.avatar,{axisSigns:data.axisSigns||[1,1,1]})) stage.showAvatar();else stage.setSkeleton(centered,edges);
  $('#scrub').value=frame;$('#frame-label').textContent=`Frame ${frame+1}/${frames.length} · ${current.slot.gesture_id} · ${current.slot.text||''}`;
}
let lastRig=null;function tick(now){if(data&&stage.avatar.rig!==lastRig){lastRig=stage.avatar.rig;draw();}if(playing&&data&&muted){const elapsed=(now-last)/1000;const duration=Math.max(.1,motion.sourceDuration);const sample=motion.draw(Math.min(1,elapsed/duration),{elapsed,duration});updateFrame(sample);if(elapsed>=duration)stop();}requestAnimationFrame(tick);}requestAnimationFrame(tick);
function stop(){speech.cancel();motion.onEnd();playing=false;muted=false;$('#play').textContent='Play speech + gesture';}
function updateFrame(sample){if(!sample)return;let offset=0;for(let i=0;i<sample.index;i++)offset+=motion.slots[i].frames.length;frame=Math.min(frames.length-1,offset+Math.floor(Math.min(sample.localTime*(data.fps||30),motion.slots[sample.index].frames.length-1)));$('#scrub').value=frame;$('#frame-label').textContent=`Clip ${sample.index+1}/${motion.slots.length} · ${sample.slot.gesture_id} · ${sample.slot.text||''}`;}
async function play(){
  if(playing){stop();$('#status').textContent='Stopped. Play restarts the synchronized sequence.';return;}
  if(!data)return;await stage.ready;playing=true;frame=0;draw();$('#play').textContent='Stop';$('#status').textContent='Preparing speech…';
  const backend=$('#speech-backend').value,language=mode==='multilingual'?$('#language').value:'en-US';
  const translated=Boolean(data.english_text)&&(backend==='kokoro'||!window.speechSynthesis?.getVoices().some(voice=>voice.lang.toLowerCase().startsWith(language.toLowerCase())));
  const text=translated?data.english_text:$('#query').value;
  try{await speech.speak(text,{backend,language:translated?'en-US':language,
    onStart:()=>{motion.onStart();$('#status').textContent=translated?'Playing translated English speech and gesture together.':'Playing speech and gesture together.';},
    onProgress:clock=>updateFrame(motion.onProgress(clock)),
    onEnd:({reason})=>{motion.onEnd();playing=false;if(reason==='ended'){updateFrame(motion.draw(1));}$('#play').textContent='Play speech + gesture';$('#status').textContent=reason==='ended'?'Playback complete.':'Playback stopped.';}
  });}catch(error){playing=true;muted=true;last=performance.now();$('#play').textContent='Stop';$('#status').textContent=`${error.message}. Playing motion without speech.`;}
}
$('#play').onclick=play;
$('#scrub').oninput=()=>{stop();frame=Number($('#scrub').value);draw();};
$('#speak').hidden=true;
$('#audio-input').onchange=async e=>{const file=e.target.files?.[0];if(!file)return;try{const result=await speech.transcribe(file);$('#query').value=result.text;$('#status').textContent=`Transcribed with ${result.backend}.`;}catch(error){$('#status').textContent=error.message;}};
$('#form').onsubmit=async e=>{
  e.preventDefault();$('#status').textContent='Retrieving…';
  stop();
  // Send only the controls this mode shows, so hidden defaults never override the prepared index.
  const params=new URLSearchParams({text:$('#query').value,language:$('#language').value});
  if(mode==='automatic')params.set('threshold',$('#threshold').value);
  if(mode==='ridge')params.set('strong_rule_threshold',$('#threshold').value);
  if(mode==='wild'||mode==='multilingual')params.set('seed',$('#seed').value);
  try{
    const response=await fetch('/api/query?'+params);const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Retrieval failed');
    data=result;frames=result.slots.flatMap((slot,index)=>slot.frames.map((joints,i)=>({joints,slot,index,i})));frame=0;last=performance.now();
    const neckIndex=Math.max(0,result.joint_order.indexOf('Neck'));
    let extent=.001;
    for(const {joints} of allFrames()){
      const origin=joints[neckIndex]||[0,0,0];
      for(const point of joints)for(let axis=0;axis<3;axis++)extent=Math.max(extent,Math.abs(point[axis]-origin[axis]));
    }
    data.motionScale=Math.min(1.5,1.5/extent);
    motion.load(data,result.english_text||$('#query').value);
    $('#scrub').max=Math.max(0,allFrames().length-1);draw();
    const pieces=[result.algorithm,`${result.slots.length} clip(s)`,result.data_label];
    if(result.rule_count!==undefined)pieces.push(`rules: ${result.rule_count}`);
    if(result.fallback_count!==undefined)pieces.push(`fallback: ${result.fallback_count}`);
    if(result.english_text)pieces.push(`English: ${result.english_text}`);
    if(result.text_encoder)pieces.push(`text matching: ${result.text_encoder}`);
    const routeCounts=formatValue(result.metrics?.route_counts);if(routeCounts)pieces.push(`routes: ${routeCounts}`);
    for(const key of ['heldout_top1','heldout_cross_view_top1']){if(Number.isFinite(result.metrics?.[key]))pieces.push(`${key.replaceAll('_',' ')}: ${result.metrics[key].toFixed(3)} (chance ${Number(result.metrics.heldout_chance??0).toFixed(3)})`);}
    $('#summary').textContent=pieces.filter(Boolean).join(' · ');
    const table=document.createElement('table'),head=document.createElement('tr');
    ['Text','Gesture','Route / cluster','Score'].forEach(t=>{const th=document.createElement('th');th.textContent=t;head.append(th);});table.append(head);
    const routes=Array.isArray(result.trace)?result.trace:result.slots;
    routes.forEach((row,index)=>{const tr=document.createElement('tr');const route=row.route||row.source?.route||row.source||(`cluster ${row.cluster_id??'—'}`);const routeLabel=typeof route==='object'?route.name||route.type||JSON.stringify(route):route;const score=row.confidence??row.similarity;[`${index+1}. ${row.text||row.english_text||row.matched_text||''}`,row.gesture_id||row.id,routeLabel,Number.isFinite(Number(score))?Number(score).toFixed(3):'—'].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});table.append(tr);});
    $('#trace').replaceChildren(table);
    $('#status').textContent=result.no_match?'No rule passed the similarity floor; the avatar holds an idle pose.':(result.slots.some(slot=>slot.route==='idle_no_match')?'Motion loaded; unmatched spans hold an idle pose.':'Motion loaded.');
  }catch(error){$('#status').textContent=error.message;}
};

// Bundled starter query: load a clip without requiring an upload.
$('#play').textContent='Play speech + gesture';
if(mode==='multilingual')$('#language').value='ko';
loadGestureLibrary().then(library=>{renderLibrary(library);const first=library?.suggested_queries?.[0];const query=typeof first==='string'?first:first?.text||first?.query;if(query){$('#query').value=query;if(mode==='multilingual')$('#language').value=/[\uac00-\ud7af]/.test(query)?'ko':'en';}return stage.ready;}).then(()=>$('#form').requestSubmit());
