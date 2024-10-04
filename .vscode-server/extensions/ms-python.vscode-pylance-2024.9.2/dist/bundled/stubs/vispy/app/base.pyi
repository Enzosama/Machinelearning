# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import time
from timeit import default_timer

from ..util import SimpleBunch

class BaseApplicationBackend(object):
    def _vispy_get_backend_name(self): ...
    def _vispy_process_events(self): ...
    def _vispy_run(self): ...
    def _vispy_reuse(self): ...
    def _vispy_quit(self): ...
    def _vispy_get_native_app(self): ...

    # is called by inputhook.py for pauses
    # to remove CPU stress
    # this is virtual so that some backends which have specialize
    # functionality to deal with user input / latency can use those methods
    def _vispy_sleep(self, duration_sec): ...

class BaseCanvasBackend(object):
    def __init__(self, vispy_canvas): ...
    def _process_backend_kwargs(self, kwargs): ...
    def _vispy_set_current(self): ...
    def _vispy_swap_buffers(self): ...
    def _vispy_set_title(self, title): ...
    def _vispy_set_size(self, w, h): ...
    def _vispy_set_position(self, x, y): ...
    def _vispy_set_visible(self, visible): ...
    def _vispy_set_fullscreen(self, fullscreen): ...
    def _vispy_update(self): ...
    def _vispy_close(self): ...
    def _vispy_get_size(self): ...
    def _vispy_get_physical_size(self): ...
    def _vispy_get_position(self): ...
    def _vispy_get_fullscreen(self): ...
    def _vispy_get_geometry(self): ...
    def _vispy_get_native_canvas(self): ...
    def _vispy_get_fb_bind_location(self): ...
    def _vispy_mouse_press(self, **kwargs): ...
    def _vispy_mouse_move(self, **kwargs): ...
    def _vispy_mouse_release(self, **kwargs): ...
    def _vispy_mouse_double_click(self, **kwargs): ...
    def _vispy_detect_double_click(self, ev, **kwargs): ...

class BaseTimerBackend(object):
    def __init__(self, vispy_timer): ...
    def _vispy_start(self, interval): ...
    def _vispy_stop(self): ...
    def _vispy_get_native_timer(self): ...
