class FlakyError(Exception):
    pass

def is_flaky(err, *args):
    return issubclass(err[0], FlakyError)
