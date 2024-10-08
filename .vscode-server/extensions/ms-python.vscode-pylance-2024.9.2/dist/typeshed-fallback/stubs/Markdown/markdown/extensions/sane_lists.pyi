from markdown import blockparser
from markdown.blockprocessors import OListProcessor, UListProcessor
from markdown.extensions import Extension

class SaneOListProcessor(OListProcessor):
    def __init__(self, parser: blockparser.BlockParser) -> None: ...

class SaneUListProcessor(UListProcessor):
    def __init__(self, parser: blockparser.BlockParser) -> None: ...

class SaneListExtension(Extension): ...

def makeExtension(**kwargs) -> SaneListExtension: ...
