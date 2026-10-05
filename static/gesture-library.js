// A playback adapter shared by retrieval viewers and application demos.
// Text spans locate clips in speech; positions advance at their recorded frame rate.
const clamp=(value,low=0,high=1)=>Math.max(low,Math.min(high,value));
const words=value=>(String(value||'').trim().match(/\S+/g)||[]).length;
const finite=(value,fallback)=>value!==null&&value!==undefined&&value!==''&&Number.isFinite(Number(value))?Number(value):fallback;

export async function loadGestureLibrary(url='/api/beat-library'){
  try{
    const response=await fetch(url);
    if(!response.ok)return null;
    const data=await response.json();
    return data?.ready&&Array.isArray(data.clips)?data:null;
  }catch{return null;}
}

function mix(a,b,weight){
  return a.map((joint,index)=>joint.map((value,axis)=>value+((b[index]?.[axis]??value)-value)*weight));
}

export class MotionSequence{
  constructor(stage=null,{blendSeconds=.16,actor=null}={}){this.stage=stage;this.actor=actor;this.blendSeconds=blendSeconds;this.load(null);}
  load(data,text=''){
    this.data=data;this.fps=Math.max(1,finite(data?.fps,30));this.slots=(data?.slots||[]).filter(slot=>Array.isArray(slot.frames)&&slot.frames.length);
    this.totalFrames=this.slots.reduce((sum,slot)=>sum+slot.frames.length,0);
    this.weights=this.slots.map(slot=>Math.max(1,words(slot.text||slot.english_text)));
    const queryWords=words(text||data?.english_text||data?.text);
    const matchedWords=this.weights.reduce((sum,value)=>sum+value,0);
    // Unmatched words receive a little time at each boundary; no false word-level alignment is claimed.
    const extra=Math.max(0,queryWords-matchedWords)/Math.max(1,this.slots.length);
    this.weights=this.weights.map(value=>value+extra);
    const total=this.weights.reduce((sum,value)=>sum+value,0)||1;
    let offset=0;
    this.ranges=this.weights.map(value=>{const range=[offset/total,(offset+value)/total];offset+=value;return range;});
    this.sourceDuration=this.slots.reduce((sum,slot)=>sum+this.duration(slot),0);
    this.active=false;this.startedAt=0;this.last=null;
    return this;
  }
  duration(slot){return Math.max(1/this.fps,finite(slot.duration_seconds,slot.frames.length/this.fps));}
  frameAt(index){
    let cursor=0;
    for(const slot of this.slots){if(index<cursor+slot.frames.length)return slot.frames[index-cursor];cursor+=slot.frames.length;}
    return this.slots.at(-1)?.frames.at(-1)||null;
  }
  sampleSlot(index,seconds){
    const slot=this.slots[index],frames=slot.frames;
    const position=clamp(seconds*this.fps,0,frames.length-1),left=Math.floor(position),right=Math.min(frames.length-1,left+1);
    return left===right?frames[left]:mix(frames[left],frames[right],position-left);
  }
  sample(progress,{elapsed=0,duration=null}={}){
    if(!this.slots.length)return null;
    progress=clamp(finite(progress,0));
    let index=this.ranges.findIndex(([,end])=>progress<end);
    if(index<0)index=this.slots.length-1;
    const [begin,end]=this.ranges[index],relative=clamp((progress-begin)/Math.max(.0001,end-begin));
    const total=Number.isFinite(duration)&&duration>0?duration:
      progress>.03&&elapsed>0?Math.max(elapsed/progress,this.sourceDuration):this.sourceDuration;
    const windowDuration=(end-begin)*total;
    const localTime=relative*windowDuration;
    let positions=this.sampleSlot(index,localTime);
    // At a join, blend toward the next clip over a short section of speech time.
    if(index<this.slots.length-1&&windowDuration>0){
      const transition=Math.min(this.blendSeconds,windowDuration*.35);
      const blend=clamp((relative*windowDuration-(windowDuration-transition))/transition);
      if(blend>0)positions=mix(positions,this.sampleSlot(index+1,0),blend);
    }
    this.last={index,slot:this.slots[index],positions,localTime,progress};
    return this.last;
  }
  draw(progress,clock={}){
    const sample=this.sample(progress,clock);if(!sample||!this.stage)return sample;
    const names=this.data.joint_order||[],neck=Math.max(0,names.indexOf('Neck'));
    const origin=sample.positions[neck]||[0,0,0],scale=this.data.motionScale||1;
    const centered=sample.positions.map(point=>[(point[0]-origin[0])*scale,1.35+(point[1]-origin[1])*scale,(point[2]-origin[2])*scale]);
    if(this.stage.setPosePositions?.(centered,names,this.actor||this.stage.avatar,{axisSigns:this.data.axisSigns||[1,1,1]})){
      if(this.actor)this.actor.root.visible=true;
      else this.stage.showAvatar();
    }
    return sample;
  }
  onStart(){this.active=true;this.startedAt=performance.now();}
  onProgress(clock){if(!this.active)return null;return this.draw(clock.progress,clock);}
  playSilent(){
    this.onEnd();this.onStart();const start=performance.now(),duration=Math.max(.1,this.sourceDuration);
    const tick=now=>{if(!this.active)return;const elapsed=(now-start)/1000;
      this.draw(Math.min(1,elapsed/duration),{elapsed,duration});
      if(elapsed>=duration){this.onEnd();return;}
      this.frameRequest=requestAnimationFrame(tick);
    };this.frameRequest=requestAnimationFrame(tick);
  }
  onEnd(){this.active=false;if(this.frameRequest){cancelAnimationFrame(this.frameRequest);this.frameRequest=null;}}
}
