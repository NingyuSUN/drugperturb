"""Minimal h5ad readers (h5py only) and a read-only HTTP range file for metadata audits.

Only ``obs`` columns and ``raw/X`` row blocks are read; nothing is loaded into
AnnData. The range file lets a design audit read ``obs`` from the public URL
without transferring expression values.
"""
from __future__ import annotations

import io
import urllib.request

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests with a block cache."""

    def __init__(self, url, block=4 << 20):
        self.url, self.pos, self.block, self.cache, self.requests = url, 0, block, {}, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
            self.size = int(response.headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos

    def _block(self, index):
        if index not in self.cache:
            start = index*self.block
            end = min(self.size, start+self.block)-1
            request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=120) as response:
                self.cache[index] = response.read()
            self.requests += 1
        return self.cache[index]

    def readinto(self, buffer):
        n = min(len(buffer), self.size-self.pos)
        if n <= 0:
            return 0
        out = bytearray()
        while len(out) < n:
            index, offset = divmod(self.pos+len(out), self.block)
            out += self._block(index)[offset:offset+n-len(out)]
        buffer[:n] = out
        self.pos += n
        return n


def _text(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def read_obs(h5, columns=None):
    """obs columns as a DataFrame indexed by obs names (categoricals kept as categoricals)."""
    obs = h5['obs']
    index_name = obs.attrs['_index']
    index = _text(obs[index_name][()])
    names = list(obs.attrs['column-order']) if columns is None else list(columns)
    data = {}
    for name in names:
        node = obs[name]
        if isinstance(node, h5py.Group):
            categories = _text(node['categories'][()])
            data[name] = pd.Categorical.from_codes(node['codes'][()], categories)
        else:
            values = node[()]
            data[name] = _text(values) if values.dtype.kind in 'OS' else values
    return pd.DataFrame(data, index=pd.Index(index, name=index_name))


def read_var_names(h5, group='raw/var'):
    node = h5[group]
    return _text(node[node.attrs['_index']][()])


def csr_shape(h5, path='raw/X'):
    node = h5[path]
    if node.attrs.get('encoding-type') != 'csr_matrix':
        raise ValueError(f'{path} is not a CSR matrix')
    return tuple(int(x) for x in node.attrs['shape'])


def iter_csr_rows(h5, path='raw/X', rows_per_block=20000):
    """Yield (row_start, CSR block) over the rows of an h5ad CSR matrix."""
    node = h5[path]
    n_rows, n_cols = csr_shape(h5, path)
    indptr = node['indptr'][()]
    for start in range(0, n_rows, rows_per_block):
        stop = min(n_rows, start+rows_per_block)
        lo, hi = int(indptr[start]), int(indptr[stop])
        block = sparse.csr_matrix((node['data'][lo:hi], node['indices'][lo:hi], indptr[start:stop+1]-lo),
                                  shape=(stop-start, n_cols))
        yield start, block
