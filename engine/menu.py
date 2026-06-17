"""
A keyboard-navigated text menu.

Game-agnostic UI: it knows nothing about levels or the game loop. The owner
builds it with a title and a list of `(label, action)` pairs, calls
`update()` once per frame with the GLFW window handle, and acts on the
returned action string (or None when nothing was activated this frame).

Visuals: a large title near the top and the items stacked below, all on a
plain (transparent) background — the owner is expected to clear to a solid
colour first. The selected item is shown purely by a larger font, with no
marker. Each item is backed by two pre-built `TextOverlay`s (normal and
selected size); `draw()` picks one per item, so nothing is allocated per
frame and the selection costs nothing to change.
"""

import glfw

from engine.text_overlay import TextOverlay


class Menu:
    def __init__(self, title, items, window_width, window_height,
                 title_font_size=72, item_font_size=34, selected_font_size=48,
                 font_name="segoeui.ttf", title_font_name="segoeuib.ttf",
                 color=(235, 235, 245, 255)):
        self._actions = [action for _, action in items]
        self._selected = 0

        # Title: centered, shifted up toward the top, no background box.
        self._title = TextOverlay(
            title, window_width, window_height,
            font_size=title_font_size, color=color,
            padding=6, corner="center", background=(0, 0, 0, 0),
            offset_px=(0, int(window_height * 0.30)),
            font_name=title_font_name)

        # Items: stacked and vertically centered as a block, sitting a touch
        # below the screen centre. Each gets a normal-size and a selected-size
        # overlay at the same vertical slot, so growing the selection stays
        # centered on its slot.
        n = len(items)
        spacing = selected_font_size + 30
        block_center_y = -int(window_height * 0.05)

        self._normal = []
        self._selected_overlays = []
        for i, (label, _) in enumerate(items):
            slot_y = block_center_y + int(((n - 1) / 2.0 - i) * spacing)
            self._normal.append(TextOverlay(
                label, window_width, window_height,
                font_size=item_font_size, color=color,
                padding=6, corner="center", background=(0, 0, 0, 0),
                offset_px=(0, slot_y), font_name=font_name))
            self._selected_overlays.append(TextOverlay(
                label, window_width, window_height,
                font_size=selected_font_size, color=color,
                padding=6, corner="center", background=(0, 0, 0, 0),
                offset_px=(0, slot_y), font_name=font_name))

        # Edge-trigger state so a held key moves the selection / activates once.
        self._up_was = False
        self._down_was = False
        self._activate_was = False

    def reset(self):
        """Put the cursor back on the first item and swallow any key that is
        currently held, so re-opening the menu doesn't immediately move or
        activate. Called whenever the menu is (re-)shown."""
        self._selected = 0
        self._up_was = self._down_was = self._activate_was = True

    def update(self, window):
        """Poll navigation keys (edge-triggered) and return the activated
        action, or None if nothing was activated this frame."""
        up = (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
              or glfw.get_key(window, glfw.KEY_W) == glfw.PRESS)
        down = (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_S) == glfw.PRESS)
        activate = (glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS
                    or glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS)

        if up and not self._up_was:
            self._selected = (self._selected - 1) % len(self._actions)
        if down and not self._down_was:
            self._selected = (self._selected + 1) % len(self._actions)
        self._up_was = up
        self._down_was = down

        action = None
        if activate and not self._activate_was:
            action = self._actions[self._selected]
        self._activate_was = activate

        return action

    def draw(self):
        self._title.draw()
        for i in range(len(self._actions)):
            overlay = (self._selected_overlays[i] if i == self._selected
                       else self._normal[i])
            overlay.draw()
