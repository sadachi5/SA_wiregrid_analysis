"""
Dummy module of pydfmux
"""

print('warning: pydfmux is not installed! Use dummy module.')

class DFMUXError(Exception):
    """A generic Exception class for DFMUX algorithm errors.

    Args:
        message: The Error message to raise (and which will be logged with the
            traceback if caught explicitly with a logging object or implicitly
            with the custom exceptionhook). There is a default message but it
            should really never be used.
            When raising this Error be specific.

        data: Any intermediate data product being produced or handed around in
            the algorithm raising the error that you want dumped to disk before
            the error is raised.

        logger: A standard logging object being used by the raising algorithm
            to be used to announce data dump and also extract output path
            in the save_returns module.
    """
    def __init__(self, message="Failure in DFMUX Code (generic message -- see stacktrace or logs)", data=None, logger=False):

        self.message = message
        self.data    = data
        Exception.__init__(self, message)

class TuberRemoteError(DFMUXError):
    pass
