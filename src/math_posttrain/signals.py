"""Persistent, non-overlapping GRPO reward-signal windows."""
class SignalWindows:
    def __init__(self, state=None, window_groups=20):
        self.state = state or dict(count=0, zero=0, streak=0, windows=0, stop=False)
        self.window_groups = window_groups

    def add(self, groups):
        for group in groups:
            self.state['count'] += 1
            self.state['zero'] += int(group['zero_variance'])
            if self.state['count'] == self.window_groups:
                ratio = self.state['zero'] / self.window_groups
                self.state['streak'] = self.state['streak'] + 1 if ratio > .8 else 0
                self.state['windows'] += 1
                self.state['last_zero_fraction'] = ratio
                self.state['stop'] |= self.state['streak'] >= 3
                self.state['count'] = self.state['zero'] = 0
        return self.state
