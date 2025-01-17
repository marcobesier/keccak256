#!/usr/bin/env python3
"""
Python implementation of keccak256 (Ethereum-style, not NIST SHA3-256).
Computes keccak256 of the string that's hardcoded in main().
"""

from operator import xor
from copy import deepcopy
from functools import reduce
from math import log

# --------------------------------------------------------------------
#                          Constants & Helpers
# --------------------------------------------------------------------

RoundConstants = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

RotationConstants = [
    [0, 1, 62, 28, 27],
    [36, 44, 6, 55, 20],
    [3, 10, 43, 25, 39],
    [41, 45, 15, 21, 8],
    [18, 2, 61, 56, 14],
]

Masks = [(1 << i) - 1 for i in range(65)]

def bits2bytes(x):
    return (x + 7) // 8

def rol(value, left, bits):
    top = value >> (bits - left)
    bot = (value & Masks[bits - left]) << left
    return bot | top

def multirate_padding(used_bytes, align_bytes):
    padlen = align_bytes - used_bytes
    if padlen == 0:
        padlen = align_bytes
    if padlen == 1:
        return [0x81]
    else:
        return [0x01] + ([0x00] * (padlen - 2)) + [0x80]

# --------------------------------------------------------------------
#                          Keccak Permutation
# --------------------------------------------------------------------

def keccak_f(state):
    def keccak_round(a, rc):
        w, h = state.W, state.H
        rangew, rangeh = state.rangeW, state.rangeH
        lanew = state.lanew
        zero = state.zero

        # Theta
        c = [reduce(xor, a[x]) for x in rangew]
        d = [0] * w
        for x in rangew:
            d[x] = c[(x - 1) % w] ^ rol(c[(x + 1) % w], 1, lanew)
            for y in rangeh:
                a[x][y] ^= d[x]

        # Rho & Pi
        b = zero()
        for x in rangew:
            for y in rangeh:
                b[y % w][(2 * x + 3 * y) % h] = rol(a[x][y], RotationConstants[y][x], lanew)

        # Chi
        for x in rangew:
            for y in rangeh:
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % w][y]) & b[(x + 2) % w][y])

        # Iota
        a[0][0] ^= rc

    nr = 12 + 2 * int(log(state.lanew, 2))
    for ir in range(nr):
        keccak_round(state.s, RoundConstants[ir])

# --------------------------------------------------------------------
#                          Keccak State & Sponge
# --------------------------------------------------------------------

class KeccakState:
    W = 5
    H = 5
    rangeW = range(W)
    rangeH = range(H)

    @staticmethod
    def zero():
        return [[0] * KeccakState.W for _ in KeccakState.rangeH]

    @staticmethod
    def lane2bytes(s, w):
        out = []
        for b in range(0, w, 8):
            out.append((s >> b) & 0xFF)
        return out

    @staticmethod
    def bytes2lane(bb):
        r = 0
        for b in reversed(bb):
            r = (r << 8) | b
        return r

    @staticmethod
    def ilist2bytes(bb):
        return bytes(bb)

    def __init__(self, bitrate, b):
        self.bitrate = bitrate
        self.b = b
        assert self.bitrate % 8 == 0
        self.bitrate_bytes = bits2bytes(self.bitrate)
        self.lanew = self.b // 25
        self.s = KeccakState.zero()

    def absorb(self, block):
        assert len(block) == self.bitrate_bytes
        block += [0] * bits2bytes(self.b - self.bitrate)
        i = 0
        for y in self.rangeH:
            for x in self.rangeW:
                lane_bytes = block[i : i + 8]
                self.s[x][y] ^= KeccakState.bytes2lane(lane_bytes)
                i += 8

    def squeeze(self):
        full = self.get_bytes()
        return full[: self.bitrate_bytes]

    def get_bytes(self):
        out = [0] * bits2bytes(self.b)
        i = 0
        for y in self.rangeH:
            for x in self.rangeW:
                v = KeccakState.lane2bytes(self.s[x][y], self.lanew)
                out[i : i + 8] = v
                i += 8
        return out

    def set_bytes(self, bb):
        i = 0
        for y in self.rangeH:
            for x in self.rangeW:
                self.s[x][y] = KeccakState.bytes2lane(bb[i : i + 8])
                i += 8

class KeccakSponge:
    def __init__(self, bitrate, width, padfn, permfn):
        self.state = KeccakState(bitrate, width)
        self.padfn = padfn
        self.permfn = permfn
        self.buffer = []

    def copy(self):
        return deepcopy(self)

    def absorb_block(self, block_bytes):
        assert len(block_bytes) == self.state.bitrate_bytes
        self.state.absorb(block_bytes)
        self.permfn(self.state)

    def absorb(self, data_bytes):
        self.buffer += data_bytes
        while len(self.buffer) >= self.state.bitrate_bytes:
            self.absorb_block(self.buffer[: self.state.bitrate_bytes])
            self.buffer = self.buffer[self.state.bitrate_bytes :]

    def absorb_final(self):
        padded = self.buffer + self.padfn(len(self.buffer), self.state.bitrate_bytes)
        self.absorb_block(padded)
        self.buffer = []

    def squeeze_once(self):
        out_block = self.state.squeeze()
        self.permfn(self.state)
        return out_block

    def squeeze(self, length):
        out = []
        while len(out) < length:
            out += self.squeeze_once()
        return out[:length]

# --------------------------------------------------------------------
#                          Keccak-256 Class
# --------------------------------------------------------------------

class KeccakHash:
    def __init__(self):
        # For Keccak256: rate=1088, capacity=512 => total 1600 bits
        # Output length = 256 bits
        self.output_bits = 256
        bitrate_bits = 1088
        capacity_bits = 512
        self.sponge = KeccakSponge(
            bitrate_bits,
            bitrate_bits + capacity_bits,
            multirate_padding,
            keccak_f,
        )
        self.digest_size = bits2bytes(self.output_bits)
        self.block_size = bits2bytes(bitrate_bits)

    def update(self, data: bytes):
        self.sponge.absorb(list(data))

    def digest(self) -> bytes:
        final = self.sponge.copy()
        final.absorb_final()
        out_bytes = final.squeeze(self.digest_size)
        return KeccakState.ilist2bytes(out_bytes)

    def hexdigest(self) -> str:
        return self.digest().hex()

def keccak256(data: bytes) -> bytes:
    h = KeccakHash()
    h.update(data)
    return h.digest()

def keccak256_hex(data: bytes) -> str:
    return keccak256(data).hex()

# --------------------------------------------------------------------
#                                Main
# --------------------------------------------------------------------

def main():
    # Hardcoded input: "2K53cuR1tY"
    msg = b"2K53cuR1tY"
    digest_hex = keccak256_hex(msg)
    print(f"keccak256(\"2K53cuR1tY\") = {digest_hex}")

if __name__ == "__main__":
    main()
