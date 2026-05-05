from pydantic import BaseModel
from typing import Optional

class LogEntry(BaseModel):
    id:          str
    type:        str
    label:       str
    description: str
    confidence:  float
    timestamp:   str
    camera:      str
    known:       Optional[bool] = None
    name:        Optional[str]  = None

class Settings(BaseModel):
    face:    bool = True
    vehicle: bool = True
    qr:      bool = True
    object:  bool = True
