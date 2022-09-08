class TimePeriods:
    supported = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    custom = []
    modes = ['trend', 'precise', 'scalp', 'alerts', 'insane', 'standard', 'custom']

    def __append__(self, tf):
        if self.supported.__contains__(tf) and not self.custom.__contains__(tf):
            print(f'Appending {tf} to custom time frame series. Current selection: {self.custom}')
            self.custom.append(tf)
            return True
        else:
            print('Unsupported time period.')
            return False

    def choose_mode(self, mode='alerts'):
        trend = ['5m', '15m', '30m', '1h', '2h', '4h']
        standard = ['3m', '15m', '30m', '4h']
        precise = ['15m', '30m', '1h', '4h', '6h', '12h']
        scalp = ['1m', '5m', '15m', '30m']
        alerts = ['1h', '2h', '4h', '12h']
        insane = ['3m', '5m', '15m', '30m', '2h', '4h', '8h']
        custom = self.custom

        if mode == 'trend':
            return 'trend', trend
        elif mode == 'precise':
            return 'precise', precise
        elif mode == 'scalp':
            return 'scalp', scalp
        elif mode == 'alerts':
            return 'alerts', alerts
        elif mode == 'custom':
            return 'custom', self.custom
        elif mode == 'standard':
            return 'standard', standard
        elif mode == 'insane':
            return 'insane', insane
        else:
            return False

    def supported_modes(self):
        ret = "Supported Modes & Time Periods:\n"
        for i in self.modes:
            tfs = self.choose_mode(i)
            ret += f'{i}: {tfs}\n'
        return ret

    def clear(self):
        self.custom = []


