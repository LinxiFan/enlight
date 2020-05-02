"""
Debugging purposes
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from .termcolor import colored


class Tracer:
    def __init__(
            self,
            trace_file: str,
            mode: str,
            verbose: bool = True,
            atol: float = 1e-6,
            rtol: float = 1e-4,
    ):
        assert mode in ['save', 'check', 'check-raise', 'off']
        self.mode = mode
        self.verbose = verbose
        if self.disabled: return
        self.trace_file = os.path.expanduser(trace_file)
        self.trace = []
        self._pointer = 0
        self.tol = (atol, rtol)
        if self.is_check():
            assert os.path.exists(self.trace_file), \
                f'in check mode, trace_file must exist: {self.trace_file}'
            with open(self.trace_file, 'rb') as fp:
                self.trace = pickle.load(fp)
        if self.enabled:
            self._info('TRACER INIT MODE: ' + mode.upper())

    @property
    def disabled(self):
        return self.mode == 'off'

    @property
    def enabled(self):
        return self.mode != 'off'

    def is_check(self):
        return self.mode.startswith('check')

    def _to_float(self, value):
        if isinstance(value, nn.Module):
            return [float(m.abs().mean().item()) for m in value.parameters()]
        elif torch.is_tensor(value):
            return float(value.abs().mean().item())
        elif isinstance(value, np.ndarray):
            return float(np.mean(np.abs(value)))
        else:
            return float(value)

    def _to_shape(self, value):
        if isinstance(value, nn.Module):
            return [tuple(m.size()) for m in value.parameters()]
        elif torch.is_tensor(value):
            return tuple(value.size())
        elif isinstance(value, np.ndarray):
            return tuple(value.shape)
        else:
            return tuple()

    def _info(self, title, *args):
        ctext = colored(f'[{title}]', color='green', attrs=['bold'])
        print(ctext, *args)

    def _error(self, title, *args):
        ctext = colored(f'[{title}]', color='white', on_color='red', attrs=['bold'])
        print(ctext, *args)

    def _value_to_str(self, value):
        if isinstance(value, list):
            s = ', '.join([f'{v:.4g}' for v in value])
            return f'[{s}]'
        else:
            return f'{value:.4g}'

    def _check_diff(self, name, v, tr_v):
        atol, rtol = self.tol
        progress = (self._pointer + 1) / len(self.trace) * 100
        if abs(v - tr_v) > atol + rtol * abs(tr_v):
            self._error(f'VALUE DIFF {progress:>4.1f}%',
                        f'{name}: actual {v} != saved {tr_v}')
            if self.mode == 'check-raise':
                raise ValueError('TRACE FAILED')
        else:
            if self.verbose:
                self._info(f'TRACE PASS {progress:>4.1f}%', name)

    def _save(self, name, value):
        if self.disabled: return
        assert self.mode == 'save'
        shape = self._to_shape(value)
        value = self._to_float(value)
        self.trace.append((name, shape, value))
        if self.verbose:
            self._info('SAVE TRACE', name,
                       'shape=', shape,
                       'value=', self._value_to_str(value))

    def _check(self, name, value):
        tr_name, tr_shape, tr_value = self.trace[self._pointer]
        shape = self._to_shape(value)
        assert name == tr_name, \
            f'event name does not match: {name} != {tr_name}'
        if name != tr_name:
            self._error('NAME MISMATCH', f'actual "{name}" != saved "{tr_name}"')
            raise ValueError
        if shape != tr_shape:
            self._error('SHAPE MISMATCH',
                        f'{name} actual {shape} != saved {tr_shape}')
            raise ValueError

        value = self._to_float(value)
        if isinstance(value, list):
            ## nn.Module
            for i, (v, tr_v) in enumerate(zip(value, tr_value)):
                self._check_diff(f'Module {name}[{i}]', v, tr_v)
        else:
            self._check_diff(f'Var {name}', value, tr_value)
        self._pointer += 1

    def event(self, name, value):
        if self.disabled: return
        if self.mode == 'save':
            self._save(name, value)
        elif self.is_check():
            self._check(name, value)
        else:
            raise NotImplementedError(self.mode)

    def done(self):
        if self.disabled: return
        if self.mode == 'save':
            with open(self.trace_file, 'wb') as fp:
                pickle.dump(self.trace, fp)
            self._info('DONE', f'debug trace saved to {self.trace_file}')
        else:
            self._info('DONE', f'trace check completed')

