"""Lazy local Kokoro TTS / faster-whisper ASR adapters; weights stay user-owned."""
import io,json,os,tempfile,wave
from pathlib import Path
from threading import Lock

class SpeechBackend:
    def __init__(self):self.tts=None;self.asr=None;self.lock=Lock()
    def status(self):return {'tts':'kokoro' if os.environ.get('KOKORO_MODEL_DIR') else 'browser','asr':bool(os.environ.get('WHISPER_MODEL_DIR')),'models_bundled':False}
    def synthesize(self,text):
        if not isinstance(text,str) or not text.strip() or len(text)>4000:raise ValueError('Supply 1–4000 characters of text.')
        folder=Path(os.environ.get('KOKORO_MODEL_DIR',''))
        paths=[folder/'config.json',folder/'kokoro-v1_0.pth',folder/'voices/af_heart.pt']
        if not os.environ.get('KOKORO_MODEL_DIR') or not all(p.is_file() for p in paths):raise ValueError('Set KOKORO_MODEL_DIR to a local Kokoro folder containing config.json, kokoro-v1_0.pth and voices/af_heart.pt; browser speech remains available.')
        with self.lock:
            if self.tts is None:
                from kokoro import KModel,KPipeline
                model=KModel(repo_id='hexgrad/Kokoro-82M',config=str(paths[0]),model=str(paths[1])).to('cpu').eval()
                self.tts=KPipeline(lang_code='a',repo_id='hexgrad/Kokoro-82M',model=model,device='cpu')
            import numpy as np
            pieces=[audio.detach().cpu().numpy() for _,_,audio in self.tts(text,voice=str(paths[2])) if audio is not None]
            if not pieces:raise ValueError('No audio generated; check the local phonemizer installation.')
            pcm=(np.clip(np.concatenate(pieces),-1,1)*32767).astype('<i2');out=io.BytesIO()
            with wave.open(out,'wb') as wav:wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm.tobytes())
            return out.getvalue()
    def transcribe(self,audio):
        if not audio or len(audio)>20*1024*1024:raise ValueError('Audio must be nonempty and at most 20 MB.')
        folder=os.environ.get('WHISPER_MODEL_DIR')
        if not folder or not (Path(folder)/'model.bin').is_file():raise ValueError('Set WHISPER_MODEL_DIR to a local faster-whisper small model directory. Typed input remains available.')
        with self.lock:
            if self.asr is None:
                from faster_whisper import WhisperModel
                self.asr=WhisperModel(folder,device='cpu',compute_type='int8',local_files_only=True)
            fd,name=tempfile.mkstemp(suffix='.audio');os.close(fd)
            try:
                Path(name).write_bytes(audio);segments,info=self.asr.transcribe(name,word_timestamps=True,vad_filter=True)
                rows=[{'start':s.start,'end':s.end,'text':s.text,'words':[{'word':w.word,'start':w.start,'end':w.end,'probability':w.probability} for w in s.words or []]} for s in segments]
                return {'text':' '.join(s['text'].strip() for s in rows),'language':info.language,'segments':rows,'backend':'faster-whisper'}
            finally:Path(name).unlink(missing_ok=True)

def speech_route(handler,backend):
    """Optional integration for stdlib HTTP handlers. Return True when handled."""
    path=handler.path.split('?',1)[0]
    if path not in ('/api/tts','/api/asr'):return False
    try:
        length=int(handler.headers.get('Content-Length','0'))
        if length<=0 or length>20*1024*1024:raise ValueError('Invalid request length.')
        body=handler.rfile.read(length)
        if path.endswith('/tts'):result=backend.synthesize(json.loads(body)['text']);kind='audio/wav'
        else:result=json.dumps(backend.transcribe(body)).encode();kind='application/json'
        handler.send_response(200);handler.send_header('Content-Type',kind);handler.send_header('Content-Length',str(len(result)));handler.end_headers();handler.wfile.write(result)
    except (ValueError,KeyError,json.JSONDecodeError,ImportError,RuntimeError,OSError) as exc:
        result=json.dumps({'error':str(exc)}).encode();handler.send_response(503 if isinstance(exc,ImportError) else 400);handler.send_header('Content-Type','application/json');handler.end_headers();handler.wfile.write(result)
    return True
