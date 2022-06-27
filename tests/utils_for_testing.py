class FlakyError(Exception):
    pass

def is_flaky(err, *args):
    return issubclass(erro[0], FlakyError)
