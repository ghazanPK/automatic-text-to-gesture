import {MotionSequence} from './gesture-library.js?v=20261006-paper2';

// A repository-local bridge: the server selects the fixed retrieval mode and
// supplies only clips prepared in this repository's ignored outputs directory.
export async function prepareApplicationMotion(stage,text,{actor=null,mode='application'}={}){
  const response=await fetch('/api/beat-query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,mode})});
  const data=await response.json().catch(()=>null);
  if(data?.ready&&Array.isArray(data.slots)&&!data.slots.length)throw new Error(mode==='wearable'?'No exact seed rule matched this utterance; speech will play without co-speech motion.':'No local recorded gesture matched this utterance; speech will play without co-speech motion.');
  if(!response.ok||!data?.ready||!Array.isArray(data.slots))
    throw new Error(data?.error||data?.reason||'Prepare the local BEAT gesture library to enable recorded co-speech motion.');
  const motion=new MotionSequence(stage,{actor}).load(data,text);
  return {motion,data};
}

export function gestureSummary(data){
  const ids=(data?.slots||[]).map(slot=>slot.gesture_id||slot.id).filter(Boolean);
  const route=data?.trace?.route||data?.trace?.routes?.join(' → ')||data?.route||'local recorded motion';
  const note=data?.trace?.translation_note?` (${data.trace.translation_note})`:'';
  return `${route}: ${ids.join(' → ')||'selected clip'}${note}`;
}
