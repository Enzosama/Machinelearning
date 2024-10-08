from _typeshed import Unused
from typing import Final

from Xlib.display import Display
from Xlib.protocol import rq
from Xlib.xobject import resource

extname: Final = "XTEST"
CurrentCursor: Final = 1

class GetVersion(rq.ReplyRequest): ...

def get_version(self: Display | resource.Resource, major: int, minor: int) -> GetVersion: ...

class CompareCursor(rq.ReplyRequest): ...

def compare_cursor(self: Display | resource.Resource, cursor: int) -> int: ...

class FakeInput(rq.Request): ...

def fake_input(
    self: Display | resource.Resource, event_type: int, detail: int = 0, time: int = 0, root: int = 0, x: int = 0, y: int = 0
) -> None: ...

class GrabControl(rq.Request): ...

def grab_control(self: Display | resource.Resource, impervious: bool) -> None: ...
def init(disp: Display, info: Unused) -> None: ...
