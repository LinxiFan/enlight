import io
import os
import sys


class PrintRedirection(object):
    """
    Context manager: temporarily redirects stdout and stderr
    """
    def __init__(self, stdout=None, stderr=None):
        """
        Args:
          stdout: if None, defaults to sys.stdout, unchanged
          stderr: if None, defaults to sys.stderr, unchanged
        """
        if stdout is None:
            stdout = sys.stdout
        if stderr is None:
            stderr = sys.stderr
        self._stdout, self._stderr = stdout, stderr

    def __enter__(self):
        self._old_out, self._old_err = sys.stdout, sys.stderr
        self._old_out.flush()
        self._old_err.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
        # restore the normal stdout and stderr
        sys.stdout, sys.stderr = self._old_out, self._old_err

    def flush(self):
        "Manually flush the replaced stdout/stderr buffers."
        self._stdout.flush()
        self._stderr.flush()


class PrintToFile(PrintRedirection):
    """
    Print to file and save/close the handle at the end.
    """
    def __init__(self, out_file=None, err_file=None):
        """
        Args:
          out_file: file path
          err_file: file path. If the same as out_file, print both stdout
              and stderr to one file in order.
        """
        self.out_file, self.err_file = out_file, err_file
        if out_file:
            out_file = os.path.expanduser(out_file)
            self.out_file = open(out_file, 'w')
        if err_file:
            err_file = os.path.expanduser(out_file)
            if err_file == out_file: # redirect both stdout/err to one file
                self.err_file = self.out_file
            else:
                self.err_file = open(os.path.expanduser(out_file), 'w')
        super().__init__(stdout=self.out_file, stderr=self.err_file)

    def __exit__(self, *args):
        super().__exit__(*args)
        if self.out_file:
            self.out_file.close()
        if self.err_file:
            self.err_file.close()


def PrintSuppress(no_out=True, no_err=False):
    """
    Args:
      no_out: stdout writes to sys.devnull
      no_err: stderr writes to sys.devnull
    """
    out_file = os.devnull if no_out else None
    err_file = os.devnull if no_err else None
    return PrintToFile(out_file=out_file, err_file=err_file)


class PrintString(PrintRedirection):
    """
    Redirect stdout and stderr to strings.
    """
    def __init__(self):
        self.out_stream = io.StringIO()
        self.err_stream = io.StringIO()
        super().__init__(stdout=self.out_stream, stderr=self.err_stream)

    def stdout(self):
        "Returns: stdout as one string."
        return self.out_stream.getvalue()

    def stderr(self):
        "Returns: stderr as one string."
        return self.err_stream.getvalue()

    def stdout_by_line(self):
        "Returns: a list of stdout line by line, ignore trailing blanks"
        return self.stdout().rstrip().split('\n')

    def stderr_by_line(self):
        "Returns: a list of stderr line by line, ignore trailing blanks"
        return self.stderr().rstrip().split('\n')


def print_str(*args, **kwargs):
    """
    Same as print() signature but returns a string
    """
    sstream = io.StringIO()
    kwargs.pop('file', None)
    print(*args, **kwargs, file=sstream)
    return sstream.getvalue()