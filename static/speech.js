// Playback lifetime and actual start/end callbacks own mouth and motion timing.

// ---------------------------------------------------------------------------
// Text → phoneme → viseme. A compact rule-based English grapheme-to-phoneme
// pass and a Hangul jamo mapping produce the 15 Oculus/ARKit-style viseme_*
// classes with relative durations. It approximates mouth shapes for demos; it
// is not a pronunciation dictionary or forced alignment.
export const VISEMES=Object.freeze(['sil','PP','FF','TH','DD','kk','CH','SS','nn','RR','aa','E','I','O','U']);
const VOWEL=new Set(['aa','E','I','O','U']);
const PEAK={sil:0,PP:1,FF:.85,TH:.7,DD:.6,kk:.55,CH:.75,SS:.65,nn:.6,RR:.65,aa:.9,E:.75,I:.65,O:.85,U:.75};
const JAW={aa:.26,E:.15,I:.09,O:.2,U:.08,DD:.05,kk:.07,CH:.04,SS:.02,nn:.05,RR:.07,TH:.06,FF:0,PP:0,sil:0};
const DURATION={PP:.7,FF:.8,TH:.8,DD:.55,kk:.6,CH:.9,SS:.9,nn:.65,RR:.7,aa:1,E:1,I:.9,O:1,U:.9};
const EXCEPTIONS={the:['TH','aa'],a:['aa'],to:['DD','U'],you:['I','U'],one:['U','aa','nn'],two:['DD','U'],are:['aa','RR'],was:['U','aa','SS'],of:['aa','FF'],
  is:['I','SS'],what:['U','aa','DD'],do:['DD','U'],does:['DD','aa','SS'],i:['aa','I'],have:['aa','FF'],said:['SS','E','DD'],says:['SS','E','SS'],there:['TH','E','RR'],
  their:['TH','E','RR'],they:['TH','E','I'],our:['aa','U','RR'],your:['I','O','RR'],who:['U'],come:['kk','aa','PP'],some:['SS','aa','PP'],done:['DD','aa','nn'],
  eye:['aa','I'],once:['U','aa','nn','SS'],people:['PP','I','PP','nn'],could:['kk','U','DD'],would:['U','U','DD'],should:['CH','U','DD'],through:['TH','RR','U'],
  hello:['E','nn','O'],okay:['O','kk','E','I'],ok:['O','kk','E','I']};
const DIGITS=['zero','one','two','three','four','five','six','seven','eight','nine'];
// Ordered longest-first; `start`/`end` restrict a rule to a word boundary.
const EN_RULES=[
  ['tion',['CH','aa','nn']],['sion',['CH','aa','nn']],['ture',['CH','RR']],['ough',['O']],['augh',['O']],['eigh',['E','I']],['igh',['aa','I']],
  ['tch',['CH']],['sch',['SS','kk']],['kn',['nn'],'start'],['wr',['RR'],'start'],['mb',['PP'],'end'],
  ['sh',['CH']],['ch',['CH']],['th',['TH']],['ph',['FF']],['wh',['U']],['ck',['kk']],['ng',['nn']],['nk',['nn','kk']],['qu',['kk','U']],['gh',[]],
  ['ee',['I']],['ea',['I']],['ie',['I']],['ei',['E','I']],['oo',['U']],['ou',['aa','U']],['ow',['O','U']],['ai',['E','I']],['ay',['E','I']],['oa',['O']],
  ['oe',['O']],['oi',['O','I']],['oy',['O','I']],['au',['O']],['aw',['O']],['ew',['I','U']],['ue',['U']],['ui',['U']],
  ['ar',['aa','RR']],['er',['RR']],['ir',['RR']],['ur',['RR']],['or',['O','RR']]
];
const EN_LETTER={a:['aa'],e:['E'],i:['I'],o:['O'],u:['U'],b:['PP'],p:['PP'],m:['PP'],f:['FF'],v:['FF'],t:['DD'],d:['DD'],n:['nn'],l:['nn'],
  k:['kk'],q:['kk'],x:['kk','SS'],s:['SS'],z:['SS'],j:['CH'],r:['RR'],w:['U'],h:[]};
const isVowelLetter=c=>/[aeiou]/.test(c);
export function englishVisemes(word){
  let w=String(word||'').toLowerCase().replace(/[^a-z]/g,'');
  if(!w)return [];
  if(EXCEPTIONS[w])return EXCEPTIONS[w].slice();
  // Silent final e (make, time) unless it forms a syllabic -le (table).
  if(w.length>2&&w.endsWith('e')&&!isVowelLetter(w.at(-2))&&!(w.endsWith('le')&&!isVowelLetter(w.at(-3)||'a')))w=w.slice(0,-1);
  const out=[];
  for(let i=0;i<w.length;){
    const rule=EN_RULES.find(([pattern,,where])=>w.startsWith(pattern,i)&&(where!=='start'||i===0)&&(where!=='end'||i+pattern.length===w.length));
    if(rule){out.push(...rule[1]);i+=rule[0].length;continue;}
    const c=w[i],next=w[i+1]||'';
    if(c===w[i-1]&&!isVowelLetter(c)){i++;continue;}// doubled consonant
    if(c==='c')out.push(/[eiy]/.test(next)?'SS':'kk');
    else if(c==='g')out.push(/[eiy]/.test(next)&&i>0?'CH':'kk');
    else if(c==='y')out.push('I');
    else out.push(...(EN_LETTER[c]||[]));
    i++;
  }
  return out;
}
// Hangul syllable = initial (19) × medial (21) × final (28).
const KO_INITIAL=['kk','kk','nn','DD','DD','RR','PP','PP','PP','SS','SS',null,'CH','CH','CH','kk','DD','PP',null];
const KO_MEDIAL=[['aa'],['E'],['I','aa'],['I','E'],['aa'],['E'],['I','aa'],['I','E'],['O'],['U','aa'],['U','E'],['U','E'],['I','O'],['U'],['U','aa'],['U','E'],['U','I'],['I','U'],['I'],['I'],['I']];
const KO_FINAL=[null,'kk','kk','kk','nn','nn','nn','DD','nn','kk','PP','nn','nn','nn','PP','nn','PP','PP','PP','DD','DD','kk','DD','DD','kk','DD','PP',null];
export function hangulVisemes(syllable){
  const code=String(syllable).codePointAt(0)-0xac00;
  if(!(code>=0&&code<11172))return [];
  const initial=Math.floor(code/588),medial=Math.floor(code%588/28),final=code%28;
  return [KO_INITIAL[initial],...KO_MEDIAL[medial],KO_FINAL[final]].filter(Boolean);
}
// Build a phoneme track whose units are relative durations. Each text segment
// maps a character span to a unit span so speech progress (boundary events,
// estimated character position or audio time) can be converted to mouth shape.
export function buildVisemeTrack(text){
  const source=String(text||''),phonemes=[],segments=[];let units=0,voiced=0;
  const push=(viseme,duration)=>{phonemes.push({viseme,start:units,end:units+duration,center:units+duration/2});units+=duration;if(viseme!=='sil')voiced++;};
  const word=(list,charStart,charEnd,peak=1)=>{
    const unitStart=units;
    // Adjacent vowels form a diphthong-like glide and share time.
    list.forEach((v,i)=>{const glide=VOWEL.has(v)&&(VOWEL.has(list[i+1])||VOWEL.has(list[i-1]));push(v,(DURATION[v]||.7)*(glide?.7:1));if(peak!==1)phonemes.at(-1).peak=peak;});
    segments.push({charStart,charEnd,unitStart,unitEnd:units});
  };
  for(const match of source.matchAll(/[A-Za-z']+|[0-9]+|[가-힣]|[.!?]+|[,;:—–]+|\n+|\s+|./gsu)){
    const token=match[0],start=match.index,end=start+token.length;
    if(/^[A-Za-z']+$/.test(token))word(englishVisemes(token),start,end);
    else if(/^[0-9]+$/.test(token))word([...token].flatMap(d=>englishVisemes(DIGITS[d])),start,end);
    else if(/^[가-힣]$/.test(token))word(hangulVisemes(token),start,end);
    else if(/^[.!?]+$/.test(token)){const s=units;push('sil',2.6);segments.push({charStart:start,charEnd:end,unitStart:s,unitEnd:units});}
    else if(/^[,;:—–]+$/.test(token)||/^\n+$/.test(token)){const s=units;push('sil',1.6);segments.push({charStart:start,charEnd:end,unitStart:s,unitEnd:units});}
    else if(/^\s+$/.test(token))segments.push({charStart:start,charEnd:end,unitStart:units,unitEnd:units});
    else if(/\p{L}/u.test(token))word(['aa'],start,end,.5);// unsupported script: generic open/close
    else segments.push({charStart:start,charEnd:end,unitStart:units,unitEnd:units});
  }
  if(!voiced)return null;
  return {text:source,units,phonemes,segments};
}
function findIndex(list,value,startKey,endKey){
  let low=0,high=list.length-1;
  while(low<=high){const mid=(low+high)>>1,item=list[mid];if(value<item[startKey])high=mid-1;else if(value>=item[endKey])low=mid+1;else return mid;}
  return -1;
}
export function charToUnit(track,char){
  if(!track)return 0;
  if(!(char>0))return 0;
  const i=findIndex(track.segments,char,'charStart','charEnd');
  if(i<0)return char>=track.text.length?track.units:0;
  const s=track.segments[i],span=Math.max(1,s.charEnd-s.charStart);
  return s.unitStart+(s.unitEnd-s.unitStart)*Math.min(1,(char-s.charStart)/span);
}
// Triangular coarticulation: each phoneme peaks at its centre and fades over
// one duration, so neighbours cross-fade at the shared boundary.
export function sampleVisemeTrack(track,unit){
  const out={};
  if(!track||!(unit>=0)||unit>=track.units)return out;
  let i=findIndex(track.phonemes,unit,'start','end');if(i<0)return out;
  let jaw=0;
  for(let j=Math.max(0,i-1);j<=Math.min(track.phonemes.length-1,i+1);j++){
    const p=track.phonemes[j],w=Math.max(0,1-Math.abs(unit-p.center)/(p.end-p.start));
    if(!w||p.viseme==='sil')continue;
    const peak=(p.peak??1)*PEAK[p.viseme],name='viseme_'+p.viseme;
    out[name]=Math.min(1,(out[name]||0)+w*peak);jaw+=w*JAW[p.viseme]*(p.peak??1);
  }
  if(jaw>0)out.jawOpen=Math.min(1,jaw);
  return out;
}
function scaleWeights(weights,gain){const out={};for(const [k,v] of Object.entries(weights))out[k]=v*gain;return out;}

// Choose a browser voice: exact name or voiceURI, then a name substring, then
// the best match for the language (exact tag, then primary subtag). Returns
// null when nothing fits so the browser keeps its default for utterance.lang.
const langTag=value=>String(value||'').replace(/_/g,'-').toLowerCase();
export function selectVoice(voices,{voice=null,lang=null}={}){
  const list=Array.isArray(voices)?voices:[];
  if(voice){
    const wanted=String(voice),lower=wanted.toLowerCase();
    const found=list.find(v=>v.name===wanted||v.voiceURI===wanted)||list.find(v=>String(v.name||'').toLowerCase().includes(lower));
    if(found)return found;
  }
  if(!lang)return null;
  const tag=langTag(lang),primary=tag.split('-')[0],rank=v=>(v.default?2:0)+(v.localService?1:0);
  const exact=list.filter(v=>langTag(v.lang)===tag).sort((a,b)=>rank(b)-rank(a));
  if(exact.length)return exact[0];
  const near=list.filter(v=>langTag(v.lang).split('-')[0]===primary).sort((a,b)=>rank(b)-rank(a));
  return near[0]||null;
}
const between=(value,low,high)=>Number.isFinite(Number(value))&&value!==null&&value!==''?Math.max(low,Math.min(high,Number(value))):null;

export class Speech {
  constructor(stage=null,base='/api'){this.stage=stage;this.base=base;this.audio=null;this.generation=0;this.finish=null;this.context=null;}
  cancel(){this.generation++;this.audio?.pause();window.speechSynthesis?.cancel();this.finish?.('cancelled');this.finish=null;this.audio=null;}
  // Voice options: voice (browser voice name or URI), lang (alias of language),
  // pitch (0–2), rate (0.1–10), kokoroVoice (local TTS voice id) and ttsUrl (a
  // local TTS route such as /api/voice/{voice}/tts; it selects the local path).
  async speak(text,{backend='browser',language='en-US',lang=null,voice=null,pitch=null,rate=null,kokoroVoice=null,ttsUrl=null,actor=this.stage?.avatar,lipSync=true,onStart=()=>{},onProgress=()=>{},onEnd=()=>{}}={}){
    this.cancel();const generation=this.generation;let url=null,audio=null;
    const speechLang=lang||language,speechRate=between(rate,.1,10);
    if(backend==='kokoro'||backend==='local'||ttsUrl){
      // Without a voice the body stays {"text"} for existing local servers.
      const body=kokoroVoice?{text,voice:kokoroVoice}:{text};
      const r=await fetch(ttsUrl||this.base+'/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      if(!r.ok)throw new Error((await r.json()).error||'Local TTS unavailable');
      url=URL.createObjectURL(await r.blob());if(generation!==this.generation){URL.revokeObjectURL(url);return;}
      audio=this.audio=new Audio(url);
    }else if(!window.speechSynthesis)throw new Error('Browser speech is unavailable; configure local TTS.');
    // Without phonemes (unsupported script) the stage keeps its amplitude envelope.
    const track=lipSync&&this.stage?.setVisemes?buildVisemeTrack(text):null;
    return new Promise((resolve,reject)=>{
      let started=false,ended=false,raf=0,start=0,anchorTime=0,anchorChar=0,analyser=null,bins=null;
      // Browser voices without boundary events follow this estimate; it scales with rate.
      const estimate=Math.max(.8,text.trim().split(/\s+/).length/2.6/(audio?1:speechRate||1)),length=Math.max(1,text.length);
      const finish=(reason='ended',error=null)=>{
        if(ended)return;ended=true;cancelAnimationFrame(raf);
        if(reason==='ended'&&!started)error=new Error('The selected voice did not start playback');
        this.stage?.setVisemes?.(null,actor);this.stage?.setSpeech(false,actor);this.stage?.setSpeechLevel?.(0,actor);
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
        if(track){
          const unit=audio&&Number.isFinite(duration)&&duration>0?elapsed/duration*track.units:charToUnit(track,char);
          let weights=sampleVisemeTrack(track,unit);
          // Real audio gates the shapes: silence in the waveform closes the mouth.
          if(analyser)weights=scaleWeights(weights,Math.max(0,Math.min(1,(level-.03)*5)));
          this.stage.setVisemes(weights,actor);
        }
        onProgress({elapsed,progress:Math.min(.99,char/length),duration:Number.isFinite(duration)?duration:null});
        raf=requestAnimationFrame(tick);
      };
      const begin=()=>{if(started||ended||generation!==this.generation)return;started=true;start=performance.now();this.stage?.setSpeech(true,actor);onStart();tick();};
      if(audio){
        try{const AudioContext=window.AudioContext||window.webkitAudioContext;if(AudioContext){this.context ||= new AudioContext();const source=this.context.createMediaElementSource(audio);analyser=this.context.createAnalyser();analyser.fftSize=512;bins=new Uint8Array(analyser.fftSize);source.connect(analyser);analyser.connect(this.context.destination);this.context.resume();}}catch{analyser=null;}
        audio.onplaying=begin;audio.onended=()=>finish();audio.onerror=()=>finish('error',new Error('Audio playback failed'));
        audio.play().catch(error=>finish('error',error));
      }else{
        const utterance=new SpeechSynthesisUtterance(text);utterance.lang=speechLang;
        const chosen=selectVoice(window.speechSynthesis.getVoices?.()||[],{voice,lang:speechLang});
        if(chosen)utterance.voice=chosen;
        const p=between(pitch,0,2);if(p!==null)utterance.pitch=p;if(speechRate!==null)utterance.rate=speechRate;
        utterance.onstart=begin;
        utterance.onboundary=event=>{if(event.name==='word'){anchorChar=event.charIndex;anchorTime=(performance.now()-start)/1000;}};
        utterance.onend=()=>finish();utterance.onerror=event=>finish('error',new Error(`Browser speech: ${event.error}`));
        window.speechSynthesis.speak(utterance);
      }
    });
  }
  async transcribe(file){const r=await fetch(this.base+'/asr',{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file});const data=await r.json();if(!r.ok)throw new Error(data.error||'ASR unavailable');return data;}
}
