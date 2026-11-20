"""JSON persistence for NPC profiles, dialogue history and saved snippets."""
import json, os
from pathlib import Path
from threading import RLock

class Store:
    def __init__(self, path=None):
        self.path=Path(path or os.environ.get("DATA_PATH","data/app.json"))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if not self.path.exists(): self.path.write_text("{}",encoding="utf-8")
        self._lock=RLock()
    def _read(self):
        try: return json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError: return {}
    def _write(self,data):
        tmp=self.path.with_suffix(self.path.suffix+".tmp")
        tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
        tmp.replace(self.path)
    def put(self,key,value):
        with self._lock:
            data=self._read(); data[key]=value; self._write(data)
    def get(self,key,default=None): return self._read().get(key,default)
    def all(self): return dict(sorted(self._read().items()))
    def append(self,key,value):
        with self._lock:
            data=self._read(); data.setdefault(key,[]).append(value); self._write(data)
            return data[key]
    def delete(self,key):
        with self._lock:
            data=self._read(); data.pop(key,None); self._write(data)
