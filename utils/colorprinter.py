import random

from colored import fg, attr, bg


class ColorPrint:
    def __init__(self, quiet=False):
        self.i = 0
        self.quiet = quiet
        self.bright_colors = [1, 3, 5, 8, 32, 44,
                              45, 46, 55, 73, 77, 80, 82, 88, 107, 125,
                              155, 160, 167, 169, 186, 197, 203, 222, 226]

    def _print(self, data):
        if self.quiet:
            pass
        else:
            print(data)

    def purple(self, data):
        self._print(f'{fg(207)}{data}{attr(0)}')

    def green(self, data):
        self._print(f'{fg(46)}{data}{attr(0)}')

    def yellow(self, data):
        self._print(f'{fg(226)}{data}{attr(0)}')

    def red(self, data):
        self._print(f'{fg(9)}{data}{attr(0)}')

    def blue(self, data):
        self._print(f'{fg(39)}{data}{attr(0)}')

    def navy(self, data):
        self._print(f'{fg(27)}{data}{attr(0)}')

    def white(self, data):
        self._print(f'{fg(231)}{data}{attr(0)}')

    def green_black(self, data):
        self._print(f'{fg(47)}{bg(0)}{attr(4)}{data}{attr(0)}')

    def blue_black(self, data):
        self._print(f'{fg(4)}{bg(0)}{data}{attr(0)}')

    def white_black(self, data):
        self._print(f'{fg(253)}{bg(0)}{data}{attr(0)}')

    def dark(self, data):
        self._print(f'{fg(57)}{bg(0)}{data}{attr(0)}')

    def alert(self, data):
        self._print(f'{fg(196)}{bg(0)}{attr(1)}{data}{attr(0)}')

    def random_color(self, data, static_set=None):
        if static_set is None:
            color = random.randint(0, 255)
        elif static_set == 'bright':
            color = random.choice(self.bright_colors)
        else:
            color = random.choice(static_set)
        self._print(f'{fg(color)}{data}{attr(0)}')

    def debug(self, data):
        if data is not None:
            self._print(f'[debug] {fg(253)}{bg(0)}{data}{attr(0)}')