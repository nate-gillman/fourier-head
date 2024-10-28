import os
from contextlib import contextmanager


@contextmanager
def use_770_permissions():
    """Using this context manager ensures we write files with 770 permissions.

    This is helpful when using shared filesystems that need write access from several users in
    the same group - but the permissions are quite lenient.
    """
    original_umask = os.umask(0o007)
    try:
        yield
    finally:
        os.umask(original_umask)
