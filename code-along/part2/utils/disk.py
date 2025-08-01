import gzip

from .config import get_mode

from diskcache import FanoutCache, Disk
from diskcache.core import MODE_BINARY

BytesType = bytes
import io

BytesIO = io.BytesIO


class GzipDisk(Disk):
    # Adds Gzip compression to when storing binary data and decompresses when retrieving it
    def store(self, value, read, key=None):
        """
        Override from base class diskcache.Disk. Compresses value if it is in bytes.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large buffers
          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and
          compression and decompression operations did not properly handle results of
          2 or 4 GiB.

        :param value: value to convert
        :param bool read: True when value is file-like object
        :param key:
        :return: (size, mode, filename, value) tuple for Cache table
        """
        # pylint: disable=unidiomatic-typecheck
        if type(value) is BytesType:
            if read:
                value = value.read()
                read = False

            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

            for offset in range(0, len(value), 2 ** 30):
                gz_file.write(value[offset:offset + 2 ** 30])
            gz_file.close()

            value = str_io.getvalue()

        return super(GzipDisk, self).store(value, read)

    def fetch(self, mode, filename, value, read):
        """
        Override from base class diskcache.Disk. Fetches data from the disk, decompresses it if necessary
        and returns it as a python object.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large buffers
          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and
          compression and decompression operations did not properly handle results of
          2 or 4 GiB.

        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        """
        value = super(GzipDisk, self).fetch(mode, filename, value, read)

        if mode == MODE_BINARY:
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            read_csio = BytesIO()

            while True:
                uncompressed_data = gz_file.read(2 ** 30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break

            value = read_csio.getvalue()

        return value


def get_cache(scope_str):
    return FanoutCache(
        'data/cache/' + scope_str,
        disk=GzipDisk,
        shards=64,
        timeout=1,
        size_limit= 260 * 1024**3 # if get_mode() == 'local' else 80 * 1024**3,
        # disk_min_file_size=2**20,
    )
