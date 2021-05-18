# Adapted from https://github.com/openai/multiagent-particle-envs.
# TODO (dries): Try using this class directly from PettingZoo and delete this file.

"""
2D rendering framework
"""
from __future__ import division

import os
import sys
from typing import Any, Dict, List, Tuple

import six

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym import error

try:
    import pyglet  # type: ignore
except ImportError:
    raise ImportError(
        "HINT: you can install pyglet directly via 'pip install "
        "pyglet'. But if you really just want to install all "
        "Gym dependencies and not have to think about it, 'pip "
        "install -e .[all]' or 'pip install gym[all]' will do it."
    )

try:
    from pyglet.canvas.base import Display  # type: ignore
    from pyglet.gl import (  # type: ignore
        GL_BLEND,
        GL_LINE_LOOP,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_LINE_STRIP,
        GL_LINES,
        GL_NICEST,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_POLYGON,
        GL_QUADS,
        GL_SRC_ALPHA,
        GL_TRIANGLES,
        glBegin,
        glBlendFunc,
        glClearColor,
        glColor4f,
        glEnable,
        glEnd,
        glHint,
        glLineWidth,
        glPopMatrix,
        glPushMatrix,
        glRotatef,
        glScalef,
        glTranslatef,
        glVertex2f,
        glVertex3f,
    )

except ImportError:
    raise ImportError(
        "Error occured while running `from pyglet.gl import *`. "
        "HINT: make sure you have OpenGL install. On Ubuntu, "
        "you can run 'apt-get install python-opengl'. If you're "
        "running on a server, you may need a virtual frame buffer; "
        "something like this should work: '"
        'xvfb-run -s "-screen 0 1400x900x24" python <your_script.py>\'',
    )

import math
from typing import Union

import numpy as np

RAD2DEG = 57.29577951308232


def get_display(
    spec: Union[None, str],
) -> Union[None, Display]:
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. "
            "(Must be a string like :0 or None.)".format(spec)
        )


class Attr(object):
    def enable(self) -> None:
        raise NotImplementedError

    def disable(self) -> None:
        pass


class Color(Attr):
    def __init__(self, vec4: Tuple[float, float, float, float]) -> None:
        self.vec4 = vec4

    def enable(self) -> None:
        glColor4f(*self.vec4)


class Geom(object):
    def __init__(self) -> None:
        self._color: Color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self) -> None:
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self) -> None:
        raise NotImplementedError

    def add_attr(self, attr: Any) -> None:
        self.attrs.append(attr)

    def set_color(self, r: float, g: float, b: float, alpha: float = 1) -> None:
        self._color.vec4 = (r, g, b, alpha)


def _add_attrs(geom: Geom, attrs: Dict[str, Any]) -> None:
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    # if "linewidth" in attrs:
    #     geom.set_linewidth(attrs["linewidth"])


class Transform(Attr):
    def __init__(
        self,
        translation: Tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        scale: Tuple[float, float] = (1, 1),
    ) -> None:
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self) -> None:
        glPushMatrix()
        glTranslatef(
            self.translation[0], self.translation[1], 0
        )  # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self) -> None:
        glPopMatrix()

    def set_translation(self, newx: float, newy: float) -> None:
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new: float) -> None:
        self.rotation = float(new)

    def set_scale(self, newx: float, newy: float) -> None:
        self.scale = (float(newx), float(newy))


class LineWidth(Attr):
    def __init__(self, stroke: int) -> None:
        self.stroke = stroke

    def enable(self) -> None:
        glLineWidth(self.stroke)


class Point(Geom):
    def __init__(self) -> None:
        Geom.__init__(self)

    def render1(self) -> None:
        glBegin(GL_POINTS)  # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class FilledPolygon(Geom):
    def __init__(self, v: List) -> None:
        Geom.__init__(self)
        self.v = v

    def render1(self) -> None:
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

        color = (
            self._color.vec4[0] * 0.5,
            self._color.vec4[1] * 0.5,
            self._color.vec4[2] * 0.5,
            self._color.vec4[3] * 0.5,
        )
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()


class PolyLine(Geom):
    def __init__(self, v: List, close: bool) -> None:
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self) -> None:
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

    # def set_linewidth(self, x):
    #     self.linewidth.stroke = x


def make_circle(
    radius: float = 10, res: int = 30, filled: bool = True
) -> Union[FilledPolygon, PolyLine]:
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(v: List, filled: bool = True) -> Union[FilledPolygon, PolyLine]:
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v: List) -> PolyLine:
    return PolyLine(v, False)


def make_capsule(length: int, width: int) -> Geom:
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs: List) -> None:
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self) -> None:
        for g in self.gs:
            g.render()


class Line(Geom):
    def __init__(
        self,
        start: Tuple[float, float] = (0.0, 0.0),
        end: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self) -> None:
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


class Viewer(object):
    def __init__(
        self, width: float, height: float, display: Union[None, str] = None
    ) -> None:
        display = get_display(display)

        self.width = width
        self.height = height

        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.geoms: List[Geom] = []
        self.onetime_geoms: List[Geom] = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self) -> None:
        self.window.close()

    def window_closed_by_user(self) -> None:
        self.close()

    def set_bounds(self, left: float, right: float, bottom: float, top: float) -> None:
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def add_geom(self, geom: Geom) -> None:
        self.geoms.append(geom)

    def add_onetime(self, geom: Geom) -> None:
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array: bool = False) -> np.array:
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            if hasattr(image_data, "data"):
                image_data = image_data.data
                arr = np.fromstring(image_data, dtype=np.uint8, sep="")
            elif hasattr(image_data, "get_data"):
                arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
            else:
                return
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape((buffer.height, buffer.width, 4))
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr

    # Convenience
    def draw_circle(
        self,
        radius: float = 10,
        res: int = 30,
        filled: bool = True,
        **attrs: Dict[str, Any],
    ) -> Geom:
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(
        self, v: List, filled: bool = True, **attrs: Dict[str, Any]
    ) -> Geom:
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v: List, **attrs: Dict[str, Any]) -> Geom:
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        **attrs: Dict[str, Any],
    ) -> Geom:
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self) -> np.array:
        self.window.flip()
        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep="")
        arr = arr.reshape((self.height, self.width, 4))
        return arr[::-1, :, 0:3]


class Image(Geom):
    def __init__(self, fname: str, width: int, height: int) -> None:
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self) -> None:
        self.img.blit(
            -self.width / 2, -self.height / 2, width=self.width, height=self.height
        )


# ================================================================


class SimpleImageViewer(object):
    def __init__(self, display: str = None) -> None:
        self.window: Union[None, pyglet.window.Window] = None
        self.isopen = False
        self.display = display

    def imshow(self, arr: np.array) -> None:
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display
            )
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (
            self.height,
            self.width,
            3,
        ), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            self.width, self.height, "RGB", arr.tobytes(), pitch=self.width * -3
        )
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self) -> None:
        if self.window is not None:
            if self.isopen:
                self.window.close()
                self.isopen = False

    def __del__(self) -> None:
        self.close()
