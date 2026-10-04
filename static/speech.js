// Replaceable local speech service with an explicit browser fallback.
export class Speech {
  constructor(stage=null,base='/api'){this.stage=stage;this.base=base;this.audio=null;this.generation=0;}
  cancel(){this.generation++;this.audio?.pause();if(this.audio?.src?.startsWith('blob:'))URL.revokeObjectURL(this.audio.src);this.audio=null;window.speechSynthesis?.cancel();this.stage?.setSpeech(false);}
  async speak(text,{backend='browser'}={}){
    this.cancel();const generation=this.generation;
    if(backend==='kokoro'){
      const r=await fetch(this.base+'/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
      if(!r.ok)throw new Error((await r.json()).error||'Local TTS unavailable');
      const url=URL.createObjectURL(await r.blob());if(generation!==this.generation){URL.revokeObjectURL(url);return;}
      this.audio=new Audio(url);this.stage?.setSpeech(true);const finish=()=>{this.stage?.setSpeech(false);URL.revokeObjectURL(url);};this.audio.onended=this.audio.onerror=finish;try{await this.audio.play();}catch(error){finish();throw error;}return;
    }
    if(!window.speechSynthesis)throw new Error('Browser speech is unavailable; configure local TTS.');
    const utterance=new SpeechSynthesisUtterance(text);utterance.onstart=()=>this.stage?.setSpeech(true);utterance.onend=utterance.onerror=()=>this.stage?.setSpeech(false);window.speechSynthesis.speak(utterance);
  }
  async transcribe(file){const r=await fetch(this.base+'/asr',{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file});const data=await r.json();if(!r.ok)throw new Error(data.error||'ASR unavailable');return data;}
}
