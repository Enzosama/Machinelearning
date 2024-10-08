from _typeshed import StrPath
from collections.abc import Collection

from xdg.IniFile import IniFile

class IconTheme(IniFile):
    def __init__(self) -> None: ...
    dir: str
    name: str
    comment: str
    directories: list[str]
    type: str
    def parse(self, file: StrPath) -> None: ...  # type: ignore[override]
    def getDir(self) -> str: ...
    def getName(self) -> str: ...
    def getComment(self) -> str: ...
    def getInherits(self) -> list[str]: ...
    def getDirectories(self) -> list[str]: ...
    def getScaledDirectories(self) -> list[str]: ...
    def getHidden(self) -> bool: ...
    def getExample(self) -> str: ...
    def getSize(self, directory: StrPath) -> int: ...
    def getContext(self, directory: StrPath) -> str: ...
    def getType(self, directory: StrPath) -> str: ...
    def getMaxSize(self, directory: StrPath) -> int: ...
    def getMinSize(self, directory: StrPath) -> int: ...
    def getThreshold(self, directory: StrPath) -> int: ...
    def getScale(self, directory: StrPath) -> int: ...
    def checkExtras(self) -> None: ...
    def checkGroup(self, group: str) -> None: ...
    def checkKey(self, key: str, value: str, group: str) -> None: ...

class IconData(IniFile):
    def __init__(self) -> None: ...
    def parse(self, file: StrPath) -> None: ...  # type: ignore[override]
    def getDisplayName(self) -> str: ...
    def getEmbeddedTextRectangle(self) -> list[int]: ...
    def getAttachPoints(self) -> list[tuple[int, int]]: ...
    def checkExtras(self) -> None: ...
    def checkGroup(self, group: str) -> None: ...
    def checkKey(self, key: str, value: str, group: str) -> None: ...

icondirs: list[str]
themes: list[IconTheme]
theme_cache: dict[str, IconTheme]
dir_cache: dict[str, tuple[str, float, float]]
icon_cache: dict[tuple[str, int, str, tuple[str, ...]], tuple[float, str]]

def getIconPath(
    iconname: str, size: int | None = None, theme: str | None = None, extensions: Collection[str] = ["png", "svg", "xpm"]
) -> str: ...
def getIconData(path: str) -> IconData: ...
def LookupIcon(iconname: str, size: int, theme: str, extensions: Collection[str]) -> str: ...
def DirectoryMatchesSize(subdir: str, iconsize: int, theme: str) -> bool: ...
def DirectorySizeDistance(subdir: str, iconsize: int, theme: str) -> int: ...
