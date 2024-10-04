from _typeshed import Incomplete, Unused
from collections.abc import Callable
from logging import Logger
from re import Pattern
from typing import Literal, NamedTuple, overload

from ._fonttools_shims import BasePen, _TTGlyphSet
from .drawing import ClippingPath, PaintedPath
from .fpdf import FPDF
from .image_datastructures import ImageCache

LOGGER: Logger

__pdoc__: dict[str, bool]

def force_nodocument(item): ...

NUMBER_SPLIT: Pattern[str]
TRANSFORM_GETTER: Pattern[str]

class Percent(float): ...

unit_splitter: Pattern[str]
relative_length_units: set[str]
absolute_length_units: dict[str, int]
angle_units: dict[str, float]

def resolve_length(length_str, default_unit: str = "pt"): ...
def resolve_angle(angle_str, default_unit: str = "deg"): ...
def xmlns(space, name): ...
def xmlns_lookup(space, *names): ...
def without_ns(qualified_tag: str) -> str: ...

shape_tags: Incomplete

def svgcolor(colorstr): ...
def convert_stroke_width(incoming): ...
def convert_miterlimit(incoming): ...
def clamp_float(min_val, max_val): ...
def inheritable(value, converter=...): ...
def optional(value, converter=...): ...

svg_attr_map: dict[str, Callable[[Incomplete], tuple[str, Incomplete]]]

def apply_styles(stylable, svg_element) -> None: ...

class ShapeBuilder:
    @overload
    @staticmethod
    def new_path(tag, clipping_path: Literal[True]) -> ClippingPath: ...
    @overload
    @staticmethod
    def new_path(tag, clipping_path: Literal[False] = False) -> PaintedPath: ...
    @overload
    @classmethod
    def rect(cls, tag, clipping_path: Literal[True]) -> ClippingPath: ...
    @overload
    @classmethod
    def rect(cls, tag, clipping_path: Literal[False] = False) -> PaintedPath: ...
    @overload
    @classmethod
    def circle(cls, tag, clipping_path: Literal[True]) -> ClippingPath: ...
    @overload
    @classmethod
    def circle(cls, tag, clipping_path: Literal[False] = False) -> PaintedPath: ...
    @overload
    @classmethod
    def ellipse(cls, tag, clipping_path: Literal[True]) -> ClippingPath: ...
    @overload
    @classmethod
    def ellipse(cls, tag, clipping_path: Literal[False] = False) -> PaintedPath: ...
    @classmethod
    def line(cls, tag) -> PaintedPath: ...
    @classmethod
    def polyline(cls, tag) -> PaintedPath: ...
    @overload
    @classmethod
    def polygon(cls, tag, clipping_path: Literal[True]) -> ClippingPath: ...
    @overload
    @classmethod
    def polygon(cls, tag, clipping_path: Literal[False] = False) -> PaintedPath: ...

def convert_transforms(tfstr): ...

class PathPen(BasePen):
    pdf_path: PaintedPath
    last_was_line_to: bool
    first_is_move: bool | None
    def __init__(self, pdf_path: PaintedPath, glyphSet: _TTGlyphSet | None = ...): ...
    def arcTo(self, rx, ry, rotation, arc, sweep, end) -> None: ...

def svg_path_converter(pdf_path: PaintedPath, svg_path: str) -> None: ...

class SVGObject:
    image_cache: ImageCache | None

    @classmethod
    def from_file(cls, filename, *args, encoding: str = "utf-8", **kwargs): ...
    cross_references: Incomplete
    def __init__(self, svg_text, image_cache: ImageCache | None = None) -> None: ...
    preserve_ar: Incomplete
    width: Incomplete
    height: Incomplete
    viewbox: Incomplete
    def update_xref(self, key: str | None, referenced) -> None: ...
    def extract_shape_info(self, root_tag) -> None: ...
    base_group: Incomplete
    def convert_graphics(self, root_tag) -> None: ...
    def transform_to_page_viewport(self, pdf, align_viewbox: bool = True): ...
    def transform_to_rect_viewport(
        self, scale, width, height, align_viewbox: bool = True, ignore_svg_top_attrs: bool = False
    ): ...
    def draw_to_page(
        self, pdf: FPDF, x: Incomplete | None = None, y: Incomplete | None = None, debug_stream: Incomplete | None = None
    ) -> None: ...
    def handle_defs(self, defs) -> None: ...
    def build_xref(self, xref): ...
    def build_group(self, group, pdf_group: Incomplete | None = None): ...
    def build_path(self, path): ...
    def build_shape(self, shape): ...
    def build_clipping_path(self, shape, clip_id): ...
    def apply_clipping_path(self, stylable, svg_element) -> None: ...
    def build_image(self, image) -> SVGImage: ...

class SVGImage(NamedTuple):
    href: str
    x: float
    y: float
    width: float
    height: float
    svg_obj: SVGObject

    def __deepcopy__(self, _memo: Unused) -> SVGImage: ...
    def render(
        self, _gsd_registry: Unused, _style: Unused, last_item, initial_point
    ) -> tuple[Incomplete, Incomplete, Incomplete]: ...
    def render_debug(
        self, gsd_registry: Unused, style: Unused, last_item, initial_point, debug_stream, _pfx: Unused
    ) -> tuple[Incomplete, Incomplete, Incomplete]: ...
