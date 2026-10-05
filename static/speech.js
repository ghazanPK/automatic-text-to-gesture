// Playback lifetime and actual start/end callbacks own mouth and motion timing.
export class Speech {
  constructor(stage=null,base='/api'){this.stage=stage;this.base=base;this.audio=null;this.generation=0;this.finish=null;this.context=null;}
  cancel(){this.generation++;this.audio?.pause();window.speechSynthesis?.cancel();this.finish?.('cancelled');this.finish=null;this.audio=null;}
  async speak(text,{backend='browser',language='en-US',actor=this.stage?.avatar,onStart=()=>{},onProgress=()=>{},onEnd=()=>{}}={}){
    this.cancel();const generation=this.generation;let url=null,audio=null;
    if(backend==='kokoro'){
      const r=await fetch(this.base+'/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
      if(!r.ok)throw new Error((await r.json()).error||'Local TTS unavailable');
      url=URL.createObjectURL(await r.blob());if(generation!==this.generation){URL.revokeObjectURL(url);return;}
      audio=this.audio=new Audio(url);
    }else if(!window.speechSynthesis)throw new Error('Browser speech is unavailable; configure local TTS.');
    return new Promise((resolve,reject)=>{
      let started=false,ended=false,raf=0,start=0,anchorTime=0,anchorChar=0,analyser=null,bins=null;
      const estimate=Math.max(.8,text.trim().split(/\s+/).length/2.6),length=Math.max(1,text.length);
      const finish=(reason='ended',error=null)=>{
        if(ended)return;ended=true;cancelAnimationFrame(raf);
        if(reason==='ended'&&!started)error=new Error('The selected voice did not start playback');
        this.stage?.setSpeech(false,actor);this.stage?.setSpeechLevel?.(0,actor);
        if(url)URL.revokeObjectURL(url);if(this.finish===finish)this.finish=null;
        onEnd({reason});error?reject(error):resolve({reason});
      };
      this.finish=finish;
      const tick=()=>{
        if(ended||generation!==this.generation)return;
        const elapsed=audio?audio.currentTime:(performance.now()-start)/1000,duration=audio?.duration;
        const char=audio&&Number.isFinite(duration)?elapsed/duration*length:Math.min(length-1,anchorChar+(elapsed-anchorTime)*length/estimate);
        let level=0;
        if(analyser){analyser.getByteTimeDomainData(bins);const rms=Math.sqrt(bins.reduce((sum,v)=>sum+((v-128)/128)**2,0)/bins.length);level=Math.min(.85,rms*5);}
        else {const c=text[Math.floor(char)]||' ';level=/[aeiouy가-힣]/i.test(c)?.48:/[\s.,!?;:]/.test(c)?0:.13;}
        this.stage?.setSpeechLevel?.(level,actor);
        onProgress({elapsed,progress:Math.min(.99,char/length),duration:Number.isFinite(duration)?duration:null});
        raf=requestAnimationFrame(tick);
      };
      const begin=()=>{if(started||ended||generation!==this.generation)return;started=true;start=performance.now();this.stage?.setSpeech(true,actor);onStart();tick();};
      if(audio){
        try{const AudioContext=window.AudioContext||window.webkitAudioContext;if(AudioContext){this.context ||= new AudioContext();const source=this.context.createMediaElementSource(audio);analyser=this.context.createAnalyser();analyser.fftSize=512;bins=new Uint8Array(analyser.fftSize);source.connect(analyser);analyser.connect(this.context.destination);this.context.resume();}}catch{analyser=null;}
        audio.onplaying=begin;audio.onended=()=>finish();audio.onerror=()=>finish('error',new Error('Audio playback failed'));
        audio.play().catch(error=>finish('error',error));
      }else{
        const utterance=new SpeechSynthesisUtterance(text);utterance.lang=language;utterance.onstart=begin;
        utterance.onboundary=event=>{if(event.name==='word'){anchorChar=event.charIndex;anchorTime=(performance.now()-start)/1000;}};
        utterance.onend=()=>finish();utterance.onerror=event=>finish('error',new Error(`Browser speech: ${event.error}`));
        window.speechSynthesis.speak(utterance);
      }
    });
  }
  async transcribe(file){const r=await fetch(this.base+'/asr',{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file});const data=await r.json();if(!r.ok)throw new Error(data.error||'ASR unavailable');return data;}
}
