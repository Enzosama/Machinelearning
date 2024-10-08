from html.parser import HTMLParser

def diagnose(data) -> None: ...
def lxml_trace(data, html: bool = True, **kwargs) -> None: ...

class AnnouncingParser(HTMLParser):
    def handle_starttag(self, name, attrs) -> None: ...
    def handle_endtag(self, name) -> None: ...
    def handle_data(self, data) -> None: ...
    def handle_charref(self, name) -> None: ...
    def handle_entityref(self, name) -> None: ...
    def handle_comment(self, data) -> None: ...
    def handle_decl(self, data) -> None: ...
    def unknown_decl(self, data) -> None: ...
    def handle_pi(self, data) -> None: ...

def htmlparser_trace(data) -> None: ...
def rword(length: int = 5): ...
def rsentence(length: int = 4): ...
def rdoc(num_elements: int = 1000): ...
def benchmark_parsers(num_elements: int = 100000) -> None: ...
def profile(num_elements: int = 100000, parser: str = "lxml") -> None: ...
