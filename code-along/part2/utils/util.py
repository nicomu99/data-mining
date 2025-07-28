import time
import datetime
import collections
import numpy as np

from .logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])


def irc2xyz(coord_irc, origin_xyz, vx_size_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]  # Step 1: Flip coordinates from IRC to CRI
    origin_a = np.array(origin_xyz)
    vx_size_xyz = np.array(vx_size_xyz)

    # Steps 2 - 4: Scale, mult with directions, add offset
    coords_xyz = (direction_a @ (cri_a * vx_size_xyz)) + origin_a
    return XyzTuple(*coords_xyz)


def xyz2irc(coord_xyz, origin_xyz, vx_size_xyz, direction_a):
    # Inverse transformation
    origin_a = np.array(origin_xyz)
    vx_size_xyz = np.array(vx_size_xyz)
    coord_a = np.array(coord_xyz)

    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vx_size_xyz  # Inverse of steps 2 - 4
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))  # Flip coordinates


def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module


def enumerate_with_estimate(iterable, desc_str, start_index=0, print_index=4, backoff=None, iter_len=None):
    if iter_len is None:
        iter_len = len(iterable)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_index < start_index * backoff:
        print_index *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_index, item) in enumerate(iterable):
        yield current_index, item
        if current_index == print_index:
            duration_sec = ((time.time() - start_ts)
                            / (current_index - start_index + 1)
                            * (iter_len - start_index)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_index,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_index *= backoff

        if current_index + 1 == start_index:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))
