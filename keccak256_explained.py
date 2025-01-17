# The comments in this file often make use of ASCII diagrams. Here's how they should be read:
#
# 1. Single Flow (Top to Bottom)
#
# Input
#   ↓
# [Box]
#   ↓
# Output
#
# This means: Input flows into Box, which processes it into Output (sequential steps)
#
# 2. Multiple Inputs to One Box
#
# Input1  Input2
#    ↓      ↓
#    [Process]
#       ↓
#     Output
#
# This means: Both Input1 and Input2 are needed/used by the Process to create the Output (multiple requirements)
#
# 3. Side Arrows
#
# [Box] ← Input
#   ↓
# Output
#
# This means: Input modifies or is applied to Box (modification or application)
#
# 4. Example from the Code
#
# Input Block (136 bytes)
#        ↓
#     ┌─────────┐
#     │ State   │ ← XOR input with state
#     │ Matrix  │
#     └─────────┘
#
# Reading this:
#   - Input Block flows down into State Matrix
#   - "XOR input with state" is an operation applied to the State Matrix (side arrow)
#
#
# For a more complete visualization of the entire process, uncomment the following mermaid code and paste it into a tool like https://mermaid.live:
# sequenceDiagram
#     autonumber

#     participant main as main()
#     participant keccak256_hex as keccak256_hex(data)
#     participant keccak256 as keccak256(data)
#     participant KeccakHash as new KeccakHash()
#     participant Sponge as KeccakSponge
#     participant State as KeccakState
#     participant keccak_f as keccak_f(state)
#     participant RoundConsts as RoundConstants
#     participant RotationConsts as RotationConstants

#     note over main: Hardcoded input b"2K53cuR1tY"
#     main->>keccak256_hex: keccak256_hex(...)
#     keccak256_hex->>keccak256: return keccak256(...)
#     keccak256->>KeccakHash: h = KeccakHash() 
#     note right of KeccakHash: Creates 1600-bit state<br>(1088-bit rate + 512-bit capacity)

#     keccak256->>KeccakHash: h.update(...)
#     KeccakHash->>Sponge: absorb(list(data))
#     Sponge->>Sponge: Check if buffer has 136 bytes?
#     alt If buffer >= 136
#         Sponge->>Sponge: absorb_block(block_bytes)
#         note right of Sponge: XOR input block<br>with state matrix
#         Sponge->>State: state.absorb(...)
#         note right of State: 5×5 lanes, each 64 bits
#         Sponge->>keccak_f: permfn(state)
#         note right of keccak_f: Apply Theta, Rho, Pi,<br>Chi, Iota
#         keccak_f->>RoundConsts: Use round constants
#         keccak_f->>RotationConsts: Use rotation constants
#         keccak_f->>State: Updated lanes
#     end

#     keccak256->>KeccakHash: h.digest()
#     note right of KeccakHash: 1) copy sponge<br>2) absorb_final()<br>3) squeeze(32 bytes)

#     KeccakHash->>Sponge: final = sponge.copy()
#     KeccakHash->>Sponge: final.absorb_final()
#     note right of Sponge: Pad leftover data<br>so final block = 136 bytes
#     Sponge->>Sponge: absorb_block(final_block)
#     Sponge->>keccak_f: permfn(state)

#     KeccakHash->>Sponge: out_bytes = final.squeeze(32)
#     Sponge->>Sponge: squeeze_once() if needed
#     Sponge->>keccak_f: permfn(state) each block
#     Sponge->>KeccakHash: return 32 bytes (hash)

#     keccak256->>keccak256_hex: return raw bytes
#     keccak256_hex->>main: convert to hex<br>print result



#!/usr/bin/env python3
"""
Python implementation of keccak256 (Ethereum-style, not NIST SHA3-256).
Computes keccak256 of the string that's hardcoded in main().
"""

# For bitwise XOR operations
from operator import xor
# Example: xor(0b1010, 0b1100) = 0b0110

# For making deep copies of the state
from copy import deepcopy
# Example: new_state = deepcopy(old_state)  # Fully independent copy

# For reducing lists using XOR
from functools import reduce
# Example: reduce(xor, [1, 2, 3]) = 1 ^ 2 ^ 3

# For calculating the logarithm
from math import log
# Example: log(64, 2) = 6  # log base 2 of 64


# --------------------------------------------------------------------
#                          Constants & Helpers
# --------------------------------------------------------------------
# This is the list of 24 round constants used in the ι (iota) step of each permutation round:
#
# RoundConstants = [
#     0x0000000000000001,  # Round 0
#     0x0000000000008082,  # Round 1
#     0x800000000000808A,  # Round 2
#     # ... and so on for all 24 rounds
# ]
#
# Properties of these constants:
# 1. Each is a 64-bit value (same as lane width)
# 2. They're chosen to:
#    - Break symmetry between rounds
#    - Have simple binary patterns
#    - Provide good diffusion
# 3. They're XORed into the first lane (a[0][0]) in each round
#
# Example of how one is used:
#
# Round 0:
# First lane:   1100 0011 ...
# RC[0]:        0000 0001
#               ↓ XOR
# New value:    1100 0010 ...
#
# (For more details on the usage, see the keccak_f function further down below.)
#
# These constants:
# - Are fixed (same for all keccak operations)
# - Were derived during keccak's design
# - Help ensure each permutation round behaves differently
# - Are part of keccak's security properties
RoundConstants = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

# These are the rotation constants used in the ρ (rho) step of the permutation, defining how many
# bits each lane should be rotated.
#
# They are used in the rho step:
#
# # For each lane at position [x][y]:
# rotated = rol(a[x][y], RotationConstants[y][x], lanew)

# Example for position [1][2]:
# - Value at a[1][2] is rotated left by 10 bits
#
# (For more details on the usage, see the keccak_f function further down below.)
#
# Visual representation of rotation amounts:
#
# State Matrix (rotation amounts for each lane)
# ┌──┬──┬───┬───┬───┐
# │ 0│ 1│62 │28 │27 │
# ├──┼──┼───┼───┼───┤
# │36│44│ 6 │55 │20 │
# ├──┼──┼───┼───┼───┤
# │ 3│10│43 │25 │39 │
# ├──┼──┼───┼───┼───┤
# │41│45│15 │21 │ 8 │
# ├──┼──┼───┼───┼───┤
# │18│ 2│61 │56 │14 │
# └──┴──┴───┴───┴───┘
#
# These constants:
# - Are chosen for optimal bit mixing
# - Create different rotation patterns for each lane
# - Help achieve good diffusion properties
# - Are fixed (same for all keccak operations)
RotationConstants = [
    #x=0  1   2   3   4
    [ 0,  1, 62, 28, 27],  # y=0
    [36, 44,  6, 55, 20],  # y=1
    [ 3, 10, 43, 25, 39],  # y=2
    [41, 45, 15, 21,  8],  # y=3
    [18,  2, 61, 56, 14],  # y=4
]

# This line creates a list of bit masks from 0 to 64 bits wide. Let's break down how each
# mask is generated:
#
# i = 0:  (1 << 0) - 1  = 1 - 1     = 0b0          = 0
# i = 1:  (1 << 1) - 1  = 2 - 1     = 0b1          = 1
# i = 2:  (1 << 2) - 1  = 4 - 1     = 0b11         = 3
# i = 3:  (1 << 3) - 1  = 8 - 1     = 0b111        = 7
# i = 4:  (1 << 4) - 1  = 16 - 1    = 0b1111       = 15
# ...
# i = 64: (1 << 64) - 1 = 2^64 - 1  = 0b11...11    = 2^64-1
#
# The << operator is a left shift (not a left rotation!) - it moves all bits in a number to the
# left by a specified amount, filling with zeros on the right:
#
# 1 << 0  (shift by 0):
# 0000 0001 → 0000 0001  (no change)

# 1 << 1  (shift by 1):
# 0000 0001 → 0000 0010  (1 becomes 2)

# 1 << 2  (shift by 2):
# 0000 0001 → 0000 0100  (1 becomes 4)

# 1 << 3  (shift by 3):
# 0000 0001 → 0000 1000  (1 becomes 8)
#
# It's a way to create powers of 2 and, when combined with subtraction, create masks of consecutive 1 bits.
# Masks is used to implement the rol helper function below.
Masks = [(1 << i) - 1 for i in range(65)]

# This helper function converts a number of bits to the minimum number of bytes needed to store those bits, rounding up.
# Here are some examples:
#
# bits2bytes(8)  = (8 + 7) // 8  = 15 // 8 = 1  byte
# bits2bytes(9)  = (9 + 7) // 8  = 16 // 8 = 2  bytes
# bits2bytes(16) = (16 + 7) // 8 = 23 // 8 = 2  bytes
# bits2bytes(17) = (17 + 7) // 8 = 24 // 8 = 3  bytes
#
# And a visualization:
#
# For 9 bits:
# ┌─────────┬─┐
# │byte 1   │1│ Need 2 bytes total
# └─────────┴─┘
#
# For 16 bits:
# ┌─────────┬─────────┐
# │byte 1   │byte 2   │ Need 2 bytes total
# └─────────┴─────────┘
#
# For 17 bits:
# ┌─────────┬─────────┬─┐
# │byte 1   │byte 2   │1│ Need 3 bytes total
# └─────────┴─────────┴─┘
#
# The +7 and integer division // ensure rounding up:
# - Any partial byte needs a full byte to store it
# - Adding 7 before division forces rounding up
# - Used when allocating space for bit sequences
def bits2bytes(x):
    return (x + 7) // 8

# This "rotate left" function performs a bit rotation to the left.
# Let's break it down with an example of left-rotating an 8-bit number by 3 bits, e.g.:
#
# rol(0b10110011, 3, 8)
#
# value = 10110011

# Step 1: top = value >> (8-3)             # Move first 3 bits to end
# 10110011 >> 5 = 00000101

# Step 2: bot = (value & 0b00011111) << 3  # Move last 5 bits to start
# 10011 << 3 = 10011000

# Step 3: return bot | top                 # Combine with OR
# 10011000
# 00000101
# --------
# 10011101
#
# The rol function is used for rotating bits within lanes in the ρ (rho) step. It:
# - Preserves all bits but changes their positions
# - Performs a circular rotation (no bits are lost)
def rol(value, left, bits):
    # Move 'left' number of bits from left side to right side
    top = value >> (bits - left)
    # Move remaining bits to the left
    bot = (value & Masks[bits - left]) << left
    # Combine the pieces
    return bot | top

# This function implements keccak's padding rule, which ensures the input length is a multiple of the rate.
# IMPORTANT: This is exactly where the Ethereum-style keccak256 differs from the official SHA3-256 that was
# standardized by NIST! (More on that below.)
def multirate_padding(used_bytes, align_bytes):
    # Calculate how many padding bytes needed
    padlen = align_bytes - used_bytes
    # If perfectly aligned, add full block of padding
    if padlen == 0:
        padlen = align_bytes
    # Special case: if only need 1 byte
    if padlen == 1:
        return [0x81]   # Combined start and end marker
    # Normal case: need 2 more bytes
    else:
        # 0x01                  - Start marker
        # [0x00] * (padlen - 2) - Zero padding
        # 0x80                  - End marker
        return [0x01] + ([0x00] * (padlen - 2)) + [0x80]
    # Here are some examples:
    #
    # align_bytes = 8 (example block size)
    #
    # Case 1: used_bytes = 6
    # padlen = 8 - 6 = 2
    # Returns: [0x01, 0x80]
    #
    # Case 2: used_bytes = 7
    # padlen = 8 - 7 = 1
    # Returns: [0x81]
    #
    # Case 3: used_bytes = 8
    # padlen = 8 (full block)
    # Returns: [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]
    #
    # Visual padding pattern:
    #
    # Normal:   [0x01][0x00...0x00][0x80]
    #           ↑     ↑             ↑
    #           start zeros        end
    #
    # Special:  [0x81]  (when only 1 byte needed)
    #           ↑
    #           combined start/end
    #
    # IMPORTANT: This function is exactly where the Ethereum-style keccak256 differs from the official
    # SHA3-256 that was standardized by NIST. To get an implementation of SHA3-256, change this function to:
    #
    # def multirate_padding(used_bytes, align_bytes):
    #     padlen = align_bytes - used_bytes
    #     if padlen == 0:
    #         padlen = align_bytes
    #     if padlen == 1:
    #         return [0x86]  # Changed from 0x81 to 0x86
    #     else:
    #         return [0x06] + ([0x00] * (padlen - 2)) + [0x80]  # Changed from 0x01 to 0x06

# --------------------------------------------------------------------
#                          Keccak Permutation
# --------------------------------------------------------------------

# The keccak_f function implements the core Keccak-f[1600] permutation function, 
# which is the main transformation function in keccak. This function:
# 1. Takes a KeccakState object (5x5 matrix)
# 2. Applies 24 rounds of transformations
# 3. Each round consists of five steps (θ, ρ, π, χ, ι):
#
# Input State
#     ↓
# For 24 rounds:
#     ┌─→ θ (Theta)  - Diffusion
#     │   ↓
#     │   ρ (Rho)    - Rotation
#     │   ↓
#     │   π (Pi)     - Permutation
#     │   ↓
#     │   χ (Chi)    - Nonlinear mixing
#     │   ↓
#     └── ι (Iota)   - Add round constant
#     ↓
# Output State
#
# Visual of state transformation:
#
# Initial State      Final State
# ┌─────┬─────┐     ┌─────┬─────┐
# │ A B │ C D │     │ W X │ Y Z │
# ├─────┼─────┤  →  ├─────┼─────┤
# │ E F │ G H │     │ P Q │ R S │
# └─────┴─────┘     └─────┴─────┘
#
# Calling keccak_f a "permutation" might seem like a misnomer. After all, it **modifies** bits with XOR and
# other, sometimes non-linear, operations, rather than just shuffling bits around in a 5x5x64 cube without
# altering their values. However, it's important to note that all operations performed by keccak_f are
# **invertible operations on a fixed-sized state**, so that keccak_f still qualifies as a permutation in the
# mathematical sense.
#
# Before we move on, a brief note on keccak_round being nested inside keccak_f: This is done for two reasons:
# 1. Scope Access:
#    - keccak_round can directly access state properties (state.W, state.H, etc.)
#    - Avoids passing these parameters repeatedly
# 2. Encapsulation:
#    - keccak_round is only used by keccak_f
#    - Keeps the round implementation private to keccak_f
def keccak_f(state):
    # The keccak_round function defines one round of the Keccak-f permutation function, taking two parameters:
    # 1. a: The state array (5x5 matrix of 64-bit lanes)
    #    ┌────┬────┬────┬────┬────┐
    #    │ a00│ a10│ a20│ a30│ a40│
    #    ├────┼────┼────┼────┼────┤
    #    │ a01│ a11│ a21│ a31│ a41│
    #    ├────┼────┼────┼────┼────┤
    #    │ a02│ a12│ a22│ a32│ a42│
    #    ├────┼────┼────┼────┼────┤
    #    │ a03│ a13│ a23│ a33│ a43│
    #    ├────┼────┼────┼────┼────┤
    #    │ a04│ a14│ a24│ a34│ a44│
    #    └────┴────┴────┴────┴────┘
    #    NOTICE THAT WE WILL ALWAYS USE THE FIRST INDEX TO SPECIFY THE COLUMN AND THE SECOND INDEX TO
    #    SPECIFY THE ROW!   
    #
    # 2. rc: Round constant
    #    - A specific value for each round
    #    - Used in the ι (iota) step
    #    - XORed with the first lane (a[0][0])
    # This function applies all five steps of a keccak round:
    # - θ (theta): Diffusion
    # - ρ (rho): Rotation
    # - π (pi): Permutation
    # - χ (chi): Nonlinear mixing
    # - ι (iota): Add round constant
    def keccak_round(a, rc):
        # The four lines of code below extract frequently used values from the state object
        # for easier access within the round function. In other words, this is just a little
        # optimization to avoid repeated attribute access and make the code more readable.
        #
        # Get dimensions of state matrix
        w, h = state.W, state.H                        # Both are 5 since the state is a 5x5 matrix
        # Get range objects for iteration              # Both are range(0, 5)
        rangew, rangeh = state.rangeW, state.rangeH
        # Get lane width in bits
        lanew = state.lanew                            # 64 bits per lane
        # Get function to create zero matrix
        zero = state.zero                              # Function to create empty 5x5 matrix

        ####################################
        #               THETA              #
        ####################################
        # This is the θ (theta) step of the round function, which provides diffusion. Let's break it down:
        # 1. For each column x, XOR all values in that column together:
        c = [reduce(xor, a[x]) for x in rangew]
        # c[0] = a[0][0] ^ a[0][1] ^ a[0][2] ^ a[0][3] ^ a[0][4]
        # c[1] = a[1][0] ^ a[1][1] ^ a[1][2] ^ a[1][3] ^ a[1][4]
        # etc.
        #
        # 2. Create array for computed differences
        d = [0] * w  # [0, 0, 0, 0, 0]
        #
        # 3. For each column
        for x in rangew:
            # The next line 
            # 
            # d[x] = c[(x - 1) % w] ^ rol(c[(x + 1) % w], 1, lanew)
            #
            # computes a difference using left and right neighbors. There's a lot to unpack in this
            # line so let's jump right in, shall we?
            # The term "difference" here refers to how this operation captures changes between neighboring
            # columns through XOR, which is commonly used to measure differences between bit patterns.
            # So, our line of code computes a "difference" in the sense of XOR differences between neighboring
            # columns (If that doesn't make sense just yet, don't worry. We'll get to it in a bit. (pun intended)):
            # 1. c[(x - 1) % w] - gets the column sum to the left
            # 2. c[(x + 1) % w] - gets the column sum to the right
            # 3. rol(c[(x + 1) % w], 1, lanew) - rotates the right column sum by 1 bit
            # 4. ^ - XORs these values together
            #
            # Here's a visual example for column 2:
            #
            # Column:     1       2       3
            # Sum:       c[1]    c[2]    c[3]
            #             ↓       ↓       ↓
            # ┌─────┐  ┌─────┐ ┌─────┐ ┌─────┐
            # │     │  │Sum  │ │     │ │Sum  │
            # │     │  │left │ │     │ │right│ → rotate right sum by 1
            # └─────┘  └─────┘ └─────┘ └─────┘
            #                     ↓
            #              XOR them together
            #                     ↓
            #                 d[2] value
            #
            # Okay, so far so good, by why do we need "% w" (modulo 5) here? Great question!
            # The "% w" is needed to handle the "wrap-around" at the edges of the state matrix. In other words,
            # it makes the state array circular in the horizontal direction.
            #
            # For x = 0:
            #   left  = (0-1) % 5 = 4  (wraps to last column)
            #   right = (0+1) % 5 = 1
            #
            # For x = 4:
            #   left  = (4-1) % 5 = 3
            #   right = (4+1) % 5 = 0  (wraps to first column)
            #
            # Here's a visual representation of this:
            #
            # State Matrix
            # ┌────┬────┬────┬────┬────┐
            # │ 0  │ 1  │ 2  │ 3  │ 4  │
            # └────┴────┴────┴────┴────┘
            #   ↑                    ↑
            #   └────────────────────┘
            #      wraps around
            #
            # For column 0:           For column 4:
            # 4 ← 0 → 1               3 ← 4 → 0
            #
            # So, without the modulo:
            # - Column 0 would try to access column -1 (invalid)
            # - Column 4 would try to access column 5 (invalid)
            #
            # To sum things up: The modulo ensures every column has both a left and right neighbor by creating
            # a circular relationship between the columns.
            #
            # Phew... you're still here? Good.
            # 
            # Next, let's talk about rol.
            # rol stands for "rotate left" - it performs a circular bit rotation on a value.
            # For example, with an 8-bit value, rol(value, shift_amount, total_bits) works as follows:
            #
            # rol(0b10110011, 1, 8):
            # Before: 1 0 1 1 0 0 1 1
            # After:  0 1 1 0 0 1 1 1
            #         ↑             ↑
            #         leftmost bit moves to rightmost position
            #
            # rol(0b10110011, 2, 8):
            # Before: 1 0 1 1 0 0 1 1
            # After:  1 1 0 0 1 1 1 0
            #         ↑ ↑         ↑ ↑
            #         two bits rotate left
            #
            # So, for the concrete example of our keccak code:
            #
            # rol(c[(x + 1) % w], 1, lanew)
            #
            # - Takes the right neighbor's column sum
            # - Rotates it left by 1 bit
            # - lanew is 64 (bits per lane)
            # - Bits that "fall off" the left end appear on the right
            # 
            # Before: 1011...0011 (64 bits)
            # After:  0110...0111 (64 bits)
            #         ↑         ↑
            #         leftmost bit moves to rightmost position
            # 
            # This rotation helps in bit mixing and diffusion across the state.
            #
            # Okay... In the unlikely case you're still reading this comment, let us clarify 
            # why this expression is to be interpreted as a "difference". For this, we'll work with
            # a simplified example using 4-bit values instead of 64-bit lanes:
            #
            # Consider three adjacent column sums:
            # c[1] = 1010  (left neighbor)
            # c[2] = 1100  (current column)
            # c[3] = 0101  (right neighbor)
            #
            # Computing d[2]:
            # 1. Left value:  c[1]     = 1010
            # 2. Right value: c[3]     = 0101
            # 3. Rotate right: rol(0101, 1, 4) = 1010
            #
            # d[2] = c[1] ^ rol(c[3], 1, 4)
            #      = 1010 ^ 1010
            #      = 0000
            #
            # This shows no "difference" because left and rotated-right are identical!
            #
            # Let's try another:
            #
            # c[1] = 1010  (left neighbor)
            # c[2] = 1100  (current column)
            # c[3] = 0110  (right neighbor)
            #
            # Computing d[2]:
            # 1. Left value:  c[1]     = 1010
            # 2. Right value: c[3]     = 0110
            # 3. Rotate right: rol(0110, 1, 4) = 1100
            #
            # d[2] = c[1] ^ rol(c[3], 1, 4)
            #      = 1010 ^ 1100
            #      = 0110
            #
            # This shows a "difference" because the values weren't identical!
            #
            # To sum things up: In modular arithmetic (mod 2), XOR behaves exactly like subtraction because -1 = 1 mod 2.
            d[x] = c[(x - 1) % w] ^ rol(c[(x + 1) % w], 1, lanew)
            
            # The for loop below XORs the computed difference with every value in the given column.
            #
            # Here's a visual representation:
            #
            # For column x:
            #
            # Before:                After:
            # ┌────┐                ┌─────────┐
            # │ A0 │                │A0 ^ d[x]│
            # ├────┤                ├─────────┤
            # │ A1 │                │A1 ^ d[x]│
            # ├────┤      d[x]      ├─────────┤
            # │ A2 │   ─────────►   │A2 ^ d[x]│
            # ├────┤                ├─────────┤
            # │ A3 │                │A3 ^ d[x]│
            # ├────┤                ├─────────┤
            # │ A4 │                │A4 ^ d[x]│
            # └────┘                └─────────┘
            #
            # The same difference value d[x] is XORed into every value in the column. This:
            # - Propagates the difference information throughout the column
            # - Creates dependencies between columns 
            # - Helps achieve the diffusion property of the hash function
            # Each value in the column is modified based on the differences between neighboring columns.
            for y in rangeh:
                a[x][y] ^= d[x]

        ####################################
        #             RHO & PI             #
        ####################################
        # This is the combined ρ (rho) and π (pi) step, which performs both rotation and rearrangement.
        # Here are the key aspects:
        # 1. Each value is rotated by a specific amount (from RotationConstants)
        # 2. Each value moves to a new position based on the formula
        # 3. The modulo operations (%w, %h) ensure we stay in the 5x5 matrix (for a more in-depth explanation,
        #    see Theta section above)
        # 4. This creates a specific permutation pattern that helps achieving further diffusion
        #
        # Here's a visualization:
        #
        # Original Position (x,y)     New Position (y, 2x+3y)
        # ┌────┬────┬────┬────┬────┐  ┌────┬────┬────┬────┬────┐
        # │    │    │ V  │    │    │  │    │    │    │    │    │
        # ├────┼────┼────┼────┼────┤  ├────┼────┼────┼────┼────┤
        # │    │    │    │    │    │  │    │    │    │    │    │
        # ├────┼────┼────┼────┼────┤  ├────┼────┼────┼────┼────┤
        # │    │    │    │    │    │  │    │    │ V' │    │    │
        # ├────┼────┼────┼────┼────┤  ├────┼────┼────┼────┼────┤
        # │    │    │    │    │    │  │    │    │    │    │    │
        # ├────┼────┼────┼────┼────┤  ├────┼────┼────┼────┼────┤
        # │    │    │    │    │    │  │    │    │    │    │    │
        # └────┴────┴────┴────┴────┘  └────┴────┴────┴────┴────┘
        #    1. Rotate bits in V          2. Move to new position
        #    2. Store as V'
        #
        # Create new empty state matrix
        b = zero()  # 5x5 matrix of zeros
        # For all 64-bit lanes
        for x in rangew:
            for y in rangeh:
                # # 1. Left-rotate the value by a specific constant (rho)
                # rotated = rol(a[x][y], RotationConstants[y][x], lanew)
                #       
                # # 2. Move to new position (pi)
                # new_x = y % w                    # new_x = y
                # new_y = (2 * x + 3 * y) % h      # new_y = 2x + 3y
                #
                # # 3. Store in new position
                # b[new_x][new_y] = rotated  
                b[y % w][(2 * x + 3 * y) % h] = rol(a[x][y], RotationConstants[y][x], lanew)

        ####################################
        #                CHI               #
        ####################################
        # This is the χ (chi) step, which adds non-linearity to the permutation. For each lane, it combines
        # three adjacent lanes in the same row using bitwise operations:
        #
        # for x in rangew:
        #     for y in rangeh:
        #         # For each lane:
        #         current = b[x][y]                   # Current lane
        #         next_one = b[(x + 1) % w][y]        # Next lane in row
        #         next_two = b[(x + 2) % w][y]        # Two lanes over in row
                
        #         # Combine using nonlinear function:
        #         a[x][y] = current ^ ((~next_one) & next_two)
        # 
        # So, for a single row (y fixed), the lanes are combined as follows:
        #
        # Row of 5 lanes (modulo wrapping at ends)
        # ┌────┬────┬────┬────┬────┐
        # │ L0 │ L1 │ L2 │ L3 │ L4 │
        # └────┴────┴────┴────┴────┘
        #
        # For L0: uses L0, L1, L2
        # For L1: uses L1, L2, L3
        # For L2: uses L2, L3, L4
        # For L3: uses L3, L4, L0  (wraps around)
        # For L4: uses L4, L0, L1  (wraps around)
        #
        # In our case, the non-linear operation x ^ (~y & z) is a bitwise function that:
        # - Takes three 64-bit inputs
        # - Produces one 64-bit output
        # - Is non-linear (can't be reduced to just XORs)
        # - Helps prevent the function from being reversed easily
        # This step is crucial for the cryptographic security of keccak as it introduces non-linearity into
        # what would otherwise be a purely linear transformation.
        #
        # To better understand why this step makes keccak much more secure, let's first analyze why x ^ (~y & z)
        # is non-linear and subsequently discuss why this fact makes the operation hard to reverse compared to 
        # purely linear operations.
        #
        # To start, let's take a closer look at the & operator:
        # If & were a linear operation (in a Boolean algebra sense), it would satisfy 
        #
        # (a & b) ^ (c & d) = (a ^ c) & (b ^ d), where a,b,c,d are 1-bit numbers.
        #
        # However, one can easily find a counter example, e.g., a=1, b=1, c=0, d=1. 
        # With this, the left-hand side evaluates to 1 while the right-hand side evaluates to 0.
        # Hence, & is not linear and, therefore, x ^ (~y & z) is not a linear function.
        #
        # Lastly, let's understand why non-linearity is much harder to reverse than a purely linear transformation:
        #
        # Linear Transformation:
        #
        # f(x) = x ^ k  (where k is some constant)
        #
        # To reverse it:
        # - Just XOR with k again
        # - Works because (x ^ k) ^ k = x
        # - Can be expressed as a system of linear equations
        #
        # Non-Linear Transformation:
        #
        # f(x,y,z) = x ^ (~y & z)
        #
        # To reverse it:
        # - Given output o, need to find x,y,z where o = x ^ (~y & z)
        # - No simple algebraic solution
        # - (~y & z) creates "if-then" relationships between bits
        # - Multiple input combinations might give same output
        #
        # Think of it like:
        #
        # Linear:
        # Input → [Simple Math] → Output
        # Output → [Simple Inverse Math] → Input
        #
        # Nonlinear:
        # Input → [Complex Dependencies] → Output
        # Output → [???] → Input(s)                 # Might have multiple solutions!
        #
        # It's like the difference between:
        #
        # 1. Solving: x + 5 = 12 (linear, easy)
        # 2. Solving: x + (if y>3 then 2 else 7) = 12 (non-linear, harder)
        #
        # To sum up: The non-linear relationships create complex dependencies that can't be solved with simple 
        # algebra, making reversal computationally harder.
        for x in rangew:
            for y in rangeh:
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % w][y]) & b[(x + 2) % w][y])

        ####################################
        #               IOTA               #
        ####################################
        # This line represents the ι (iota) step, which XORs a round constant into the first lane of the state.
        # Purpose: 
        # 1. Breaks symmetry between rounds 
        # 2. Each round uses a different round constant (rc)
        # 3. Without this, all rounds would be identical
        # 4. Makes each round unique and distinguishable
        # It's like adding a unique "signature" to each round of the permutation to ensure they're all 
        # different from each other.
        #
        # Here's a visual representation:
        #
        # State Matrix
        # ┌────────┬────┬────┬────┬────┐
        # │a[0][0] │    │    │    │    │
        # │   ^    │    │    │    │    │
        # │   rc   │    │    │    │    │
        # ├────────┼────┼────┼────┼────┤
        # │        │    │    │    │    │
        # ├────────┼────┼────┼────┼────┤
        # │        │    │    │    │    │
        # ├────────┼────┼────┼────┼────┤
        # │        │    │    │    │    │
        # ├────────┼────┼────┼────┼────┤
        # │        │    │    │    │    │
        # └────────┴────┴────┴────┴────┘
        a[0][0] ^= rc  # XOR round constant into first lane (top-left)

    ##############################################################
    #     END OF keccak_round / CONTINUATION OF keccak_f         #
    ##############################################################
    # These last three lines orchestrate the entire Keccak-f permutation by applying multiple rounds:
    # Calculate number of rounds based on lane width
    nr = 12 + 2 * int(log(state.lanew, 2))
    # Apply that many rounds
    for ir in range(nr):
        # Apply one round with its specific round constant
        keccak_round(state.s, RoundConstants[ir])
    #
    # For keccak256 (64-bit lanes), this means:
    #
    # Round  0: keccak_round(state, RoundConstants[0])
    # Round  1: keccak_round(state, RoundConstants[1])
    # Round  2: keccak_round(state, RoundConstants[2])
    # ...
    # Round 23: keccak_round(state, RoundConstants[23])
    #
    # Each round applies:
    #
    # 1. θ (theta)   - column parity mixing
    # 2. ρ (rho)     - rotation of lanes
    # 3. π (pi)      - permutation of lanes
    # 4. χ (chi)     - nonlinear mixing
    # 5. ι (iota)    - addition of round constant
    # 
    # Visual flow:
    #
    # Initial State
    #      ↓
    # Round 0 (with RC[0])
    #      ↓
    # Round 1 (with RC[1])
    #      ↓
    # Round 2 (with RC[2])
    #      ↓
    #     ...
    #      ↓
    # Round 23 (with RC[23])
    #      ↓
    # Final State
    #
    # To sum up: Each round transforms the entire state using a different round constant to ensure unique behavior in each round.

# --------------------------------------------------------------------
#                          Keccak State & Sponge
# --------------------------------------------------------------------

# The KeccakState class represents the core state matrix of the Keccak hash function. Here's a high-level overview:
#
# 1. State Structure:
#
#    5×5 Matrix of 64-bit values:
#    ┌─────┬─────┬─────┬─────┬─────┐
#    │ 0,0 │ 1,0 │ 2,0 │ 3,0 │ 4,0 │
#    ├─────┼─────┼─────┼─────┼─────┤
#    │ 0,1 │ 1,1 │ 2,1 │ 3,1 │ 4,1 │
#    ├─────┼─────┼─────┼─────┼─────┤
#    │ 0,2 │ 1,2 │ 2,2 │ 3,2 │ 4,2 │
#    ├─────┼─────┼─────┼─────┼─────┤
#    │ 0,3 │ 1,3 │ 2,3 │ 3,3 │ 4,3 │
#    ├─────┼─────┼─────┼─────┼─────┤
#    │ 0,4 │ 1,4 │ 2,4 │ 3,4 │ 4,4 │
#    └─────┴─────┴─────┴─────┴─────┘
#
# 2. Main Operations:
#    - absorb: XORs input data into the state
#    - squeeze: Extracts data from the state
#    - get_bytes: Converts state to bytes 
#    - set_bytes: Sets state from bytes 
# 
# 3. State Division:
#
#    Total State (1600 bits)
#    ┌────────────────────┐
#    │     Rate           │ ← 1088 bits for data
#    ├────────────────────┤
#    │    Capacity        │ ← 512 bits for security
#    └────────────────────┘
#
# 4. Helper Methods:
#    - lane2bytes: Converts 64-bit lanes to bytes
#    - bytes2lane: Converts bytes to 64-bit lanes
#    - zero: Creates empty 5x5 state matrix
#
# As a summary, the state is where:
# - Input data gets absorbed
# - Permutations are applied 
# - The output hash is extracted from
class KeccakState:
    # Class-level constants that define the dimensions of the 5x5 keccak state matrix:
    # ┌─────┬─────┬─────┬─────┬─────┐
    # │ 0,0 │ 1,0 │ 2,0 │ 3,0 │ 4,0 │  ← rangeW [0-4]
    # ├─────┼─────┼─────┼─────┼─────┤
    # │ 0,1 │ 1,1 │ 2,1 │ 3,1 │ 4,1 │
    # ├─────┼─────┼─────┼─────┼─────┤
    # │ 0,2 │ 1,2 │ 2,2 │ 3,2 │ 4,2 │  ↑
    # ├─────┼─────┼─────┼─────┼─────┤  rangeH
    # │ 0,3 │ 1,3 │ 2,3 │ 3,3 │ 4,3 │  [0-4]
    # ├─────┼─────┼─────┼─────┼─────┤  ↓
    # │ 0,4 │ 1,4 │ 2,4 │ 3,4 │ 4,4 │
    # └─────┴─────┴─────┴─────┴─────┘
    #
    # These ranges are used in the code for iterating over the matrix, e.g., like this:
    #
    # for x in self.rangeW:      # x goes from 0 to 4
    #     for y in self.rangeH:  # y goes from 0 to 4
    #         # Access element at position (x,y)
    #         value = self.s[x][y]
    #
    # The 5x5 size is a fundamental part of keccak's design, where each position holds a 64-bit value (called a "lane"), making the total size 5x5x64 = 1600 bits.
    W = 5
    H = 5
    rangeW = range(W)
    rangeH = range(H)

    # This static method creates an empty 5x5 matrix filled with zeros. Here's a break down:
    # 1. [0] * KeccakState.W creates a row of 5 zeros: [0, 0, 0, 0, 0]
    # 2. for _ in KeccakState.rangeH ensures this is done 5 times (one time for each row)
    # 3. The result is a 5x5 matrix of zeros:
    #
    # [
    #     [0, 0, 0, 0, 0],  # Row 0
    #     [0, 0, 0, 0, 0],  # Row 1
    #     [0, 0, 0, 0, 0],  # Row 2
    #     [0, 0, 0, 0, 0],  # Row 3
    #     [0, 0, 0, 0, 0]   # Row 4
    # ]
    # 
    # Here's a usage example:
    # 
    # # Create empty state matrix
    # empty_state = KeccakState.zero()
    # # Access elements 
    # empty_state[0][0] (top-left corner)
    # empty_state[4][4] (bottom-right corner)
    #
    # The @staticmethod decorator means:
    # - Method can be called without creating a KeccakState instance
    # - No self parameter needed
    # - Used like: KeccakState.zero()
    # The zero() method is used when initializing a new KeccakState to create its empty state matrix.
    @staticmethod
    def zero():
        return [[0] * KeccakState.W for _ in KeccakState.rangeH]

    # This helper method converts a lane (64-bit value) into bytes:
    # - Extracts bytes from least significant to most significant (little-endian)
    # - & 0xFF masks off all but the lowes 8 bits
    # - Used when converting state matrix to output bytes
    # - @staticmethod means it can be called without creating an instance
    #
    # Here's an example with a 16-bit/2-byte value (for simplicity):
    #
    # Let s = 0x1234 (binary: 0001 0010 0011 0100)
    # w = 16 bits
    #
    # First iteration (b = 0):
    # s >> 0 = 0x1234
    # 0x1234 & 0xFF = 0x34 (first byte)
    # 
    # Second iteration (b = 8):
    # s >> 8 = 0x12
    # 0x12 & 0xFF (second byte)
    # 
    # Result: [0x34, 0x12] # Little-endian byte order
    # 
    # For a 64-bit lane:
    # 
    # 64-bit value: 0x0123456789ABCDEF
    # 
    # Converted to bytes (little-endian):
    # [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]
    @staticmethod
    def lane2bytes(s, w):
        # s: the lane value (e.g., a 64-bit integer)
        # w: width in bits (typically 64)
        out = []
        for b in range(0, w, 8):  # Step by 8 bits (1 byte) at a time
            # Extract each byte using right shift and mask
            out.append((s >> b) & 0xFF)
        return out

    # This helper method does the opposite of lane2bytes - it converts a sequence of bytes back into
    # a lane value (e.g., 64-bit integer). Here are the key points:
    # - Processes bytes in reverse order (from most to least significant)
    # - Uses left shifts (<<) and OR operations ( | ) to build the value
    # - Inverse operation of lane2bytes
    # - Used when converting input bytes to state matrix values
    #
    # Here's an example with a 16-bit/2-byte value (for simplicity):
    #
    # Input bytes: bb = [0x34, 0x12]
    #
    # First iteration (b = 0x12):
    # r = 0
    # r << 8 = 0x00
    # 0x00 | 0x12 = 0x12
    # r = 0x12
    #
    # Second iteration (b = 0x34):
    # r << 8 = 0x1200
    # 0x1200 | 0x34 = 0x1234
    # r = 0x1234
    #
    # Result: 0x1234
    #
    # For a 64-bit lane:
    #
    # Input (little-endian bytes):
    # bb = [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]
    #
    # After processing:
    # Returns: 0x0123456789ABCDEF
    @staticmethod
    def bytes2lane(bb):
        # Initialize result
        r = 0
        # Process bytes from most to least significant
        for b in reversed(bb):
            # Shift left and OR in the next byte
            r = (r << 8) | b
        # Return final result
        return r

    # A very simple helper method that converts a list of integers into a Python bytes object using ASCII encoding.
    # Example usage:
    #
    # # List of integers
    # integers = [65, 66, 67]   # ASCII values for 'A', 'B', 'C'
    #
    # # Convert to bytes
    # result = KeccakState.ilist2bytes(integers)
    # # result = b'ABC'
    #
    # # Another example
    # integers = [0x1A, 0x2B, 0x3C]
    # result = KeccakState.ilist2bytes(integers)
    # # result = b'\x1a+<'
    #
    # This method is used in the final step of hashing to convert the output from a list of integer values to a proper bytes object:
    #
    # def digest(self) -> bytes:
    #     # ... other processing ...
    #     out_bytes = final.squeeze(self.digest_size)  # Returns list of integers
    #     return KeccakState.ilist2bytes(out_bytes)    # Converts to bytes
    # 
    # It's a simple utility method that helps with type conversion, making the output compatible with Python's
    # standard library expectations for hash functions.
    @staticmethod
    def ilist2bytes(bb):
        # Converts list of integers to bytes object
        return bytes(bb)

    # Let's now break down the KeccakState initialization. For keccak256, the values are:
    # 
    # bitrate = 1088       # Rate (r)
    # b = 1600             # Total width
    # # This results in:
    # bitrate_bytes = 136  # 1088/8
    # lanew = 64           # 1600/25
    #
    # Here's a visual representation:
    #
    # Total State (b=1600 bits)
    # ┌────────────────────┐
    # │ Rate (r=1088 bits) │ ← For data input/output
    # ├────────────────────┤
    # │ Capacity (512 bits)│ ← For security
    # └────────────────────┘
    #
    # State Matrix (5x5)
    # Each cell = 64-bit lane
    # ┌────┬────┬────┬────┬────┐
    # │ 64 │ 64 │ 64 │ 64 │ 64 │
    # ├────┼────┼────┼────┼────┤
    # │ 64 │ 64 │ 64 │ 64 │ 64 │
    # ├────┼────┼────┼────┼────┤
    # │ 64 │ 64 │ 64 │ 64 │ 64 │
    # ├────┼────┼────┼────┼────┤
    # │ 64 │ 64 │ 64 │ 64 │ 64 │
    # ├────┼────┼────┼────┼────┤
    # │ 64 │ 64 │ 64 │ 64 │ 64 │
    # └────┴────┴────┴────┴────┘
    def __init__(self, bitrate, b):
        # Rate (r): how many bits for input/output
        self.bitrate = bitrate
        # Total width in bits (1600 for keccak256)
        self.b = b
        # Ensure bitrate is byte-aligned
        assert self.bitrate % 8 == 0
        # Convert bitrate to bytes (1088 bits -> 136 bytes)
        self.bitrate_bytes = bits2bytes(self.bitrate)
        # Calculate lane width (1600/25 = 64 bits per lane)
        self.lanew = self.b // 25   # 25 comes from 5x5 state matrix
        # Initialize empty 5x5 state matrix
        self.s = KeccakState.zero()

    # The absorb method performs the following operations:
    # 1. Verify block size (136 bytes)
    # 2. Pad with zeros to 200 bytes
    # 3. For each positiion in 5x5 matrix:
    #    - Take next 8 bytes
    #    - Convert to 64-bit lane value
    #    - XOR with existing state value
    # 4. Process left-to-right, top-to-bottom
    #
    # Here's an example flow:
    #
    # Input block: [byte0...byte135]  (136 bytes)
    # Add padding: [byte0...byte135, 0...0]  (200 bytes total)
    #
    # Process each lane:
    # ┌─────────┬─────────┬────────┬────────┬────────┐
    # │bytes0-7 │bytes8-15│  ...   │  ...   │  ...   │
    # ├─────────┼─────────┼────────┼────────┼────────┤
    # │  ...    │   ...   │  ...   │  ...   │  ...   │
    # ├─────────┼─────────┼────────┼────────┼────────┤
    # │  ...    │   ...   │  ...   │  ...   │  ...   │
    # ├─────────┼─────────┼────────┼────────┼────────┤
    # │  ...    │   ...   │  ...   │  ...   │  ...   │
    # ├─────────┼─────────┼────────┼────────┼────────┤
    # │  ...    │   ...   │  ...   │  ...   │  ...   │
    # └─────────┴─────────┴────────┴────────┴────────┘
    def absorb(self, block):
        # Verify block is correct size (136 bytes for keccak256)
        assert len(block) == self.bitrate_bytes
        # Pad block with zeros to fill entire state (200 bytes total)
        block += [0] * bits2bytes(self.b - self.bitrate)
        # Process each lane (64 bits / 8 bytes) in the state matrix
        i = 0
        for y in self.rangeH:       # 5 rows
            for x in self.rangeW:   # 5 columns
                # Get next 8 bytes for this lane
                lane_bytes = block[i : i + 8]
                # Convert bytes to 64-bit value and XOR with current state
                self.s[x][y] ^= KeccakState.bytes2lane(lane_bytes)
                i += 8

    # The squeeze method of KeccakState:
    # 1. Gets all bytes from the state
    # 2. Returns only the "rate" portion
    # 3. Keeps the "capacity" portion private (for security)
    # The rate portion is what's used for the hash output, while the capacity portion is never directly exposed.
    #
    # Example:
    #
    # For keccak256:
    # full = all 200 bytes (1600 bits) of state
    # self.bitrate_bytes = 136 (1088 bits)
    #
    # If full = [byte0, byte1, ..., byte199]
    # Returns: [byte0, byte1, ..., byte135]
    #
    #
    # Here's a visual representation:
    #
    # State Matrix (1600 bits / 200 bytes)
    # ┌────────────────────┐
    # │ Rate portion       │ ← Return these bytes (136)
    # │ (1088 bits)        │
    # ├────────────────────┤
    # │ Capacity portion   │ ← Ignore these bytes (64)
    # │ (512 bits)         │
    # └────────────────────┘
    def squeeze(self):
        # Convert entire state to bytes
        full = self.get_bytes()
        # Return only the rate portion (first 136 for keccak256)
        return full[: self.bitrate_bytes]

    # This method converts the entire state to a byte array, which is then used by squeeze() to get the output hash.
    # The process is as follows:
    # 1. Create empty 200-byte array 
    # 2. For each lane in the state matrix:
    #    - Convert 64-bit lane to 8 bytes 
    #    - Place bytes in correct position in output array
    # 3. Process left-to-right, top-to-bottom
    # 
    # Example flow:
    #
    # State Matrix (5x5 of 64-bit lanes)
    # ┌────────┬────────┬────────┬────────┬────────┐
    # │lane0,0 │lane1,0 │lane2,0 │lane3,0 │lane4,0 │
    # ├────────┼────────┼────────┼────────┼────────┤
    # │lane0,1 │lane1,1 │lane2,1 │lane3,1 │lane4,1 │
    # ├────────┼────────┼────────┼────────┼────────┤
    # │  ...   │  ...   │  ...   │  ...   │  ...   │
    # └────────┴────────┴────────┴────────┴────────┘

    # Converts to:
    # [byte0...byte7][byte8...byte15]...[byte192...byte199]
    #
    # Visual of byte placement:
    # Lane 0,0 → bytes 0-7
    # Lane 1,0 → bytes 8-15
    # Lane 2,0 → bytes 16-23
    # ...and so on
    def get_bytes(self):
        # Create array for all bytes (200 bytes for 1600-bit state)
        out = [0] * bits2bytes(self.b)
        i = 0
        # Process each position in 5x5 matrix
        for y in self.rangeH:       # 5 rows
            for x in self.rangeW:   # 5 columns
                # Convert 64-bit lane to 8 bytes
                v = KeccakState.lane2bytes(self.s[x][y], self.lanew)
                # Place bytes in output array 
                out[i : i + 8] = v
                i += 8
        return out

    # This method is essentially the inverse of get_bytes(), converting a byte array back into
    # the state matrix format.
    # The process is as follows:
    # 1. Start with array of 200 bytes
    # 2. For each postion in 5x5 matrix:
    #    - Take next 8 bytes
    #    - Convert to 64-bit lane value
    #    - Store in state matrix
    # 3. Process left-to-right, top-to-bottom
    #
    # Example flow:
    # 
    # Input bytes:
    # [byte0...byte7][byte8...byte15]...[byte192...byte199]
    #
    # Converts to State Matrix:
    # ┌────────┬────────┬────────┬────────┬────────┐
    # │lane0,0 │lane1,0 │lane2,0 │lane3,0 │lane4,0 │
    # ├────────┼────────┼────────┼────────┼────────┤
    # │lane0,1 │lane1,1 │lane2,1 │lane3,1 │lane4,1 │
    # ├────────┼────────┼────────┼────────┼────────┤
    # │  ...   │  ...   │  ...   │  ...   │  ...   │
    # └────────┴────────┴────────┴────────┴────────┘
    #
    # Visual of byte to lane mapping:
    #
    # bytes 0-7   → Lane 0,0
    # bytes 8-15  → Lane 1,0
    # bytes 16-23 → Lane 2,0
    # ...and so on
    def set_bytes(self, bb):
        i = 0
        # Process each position in 5x5 matrix
        for y in self.rangeH:      # 5 rows
            for x in self.rangeW:  # 5 columns
                # Convert next 8 bytes to 64-bit lane value
                self.s[x][y] = KeccakState.bytes2lane(bb[i : i + 8])
                i += 8

# The KeccakSponge class implements the core "sponge construction" of keccak which
# - absorbs input data,
# - transforms the absorbed data through permutations,
# - and squeezes out the hash value.
# 
# The process can be visualized as follows:
#
# 1. Absorption Phase:
#
# Input Data → Buffer → Full Blocks → State
#                    ↓
#                 Padding (for last block)
#
# 2. Transformation:
#
# State Matrix (5x5)
#     ↓
# Keccak-f permutation
#     ↓
# Updated State
#
# 3. Squeezing Phase:
# 
# State → Extract Block → Permute → Extract Block → ...
#          ↓
#          Output Hash
#
# As a summary, the sponge construction is what makes keccak unique and flexible - it can:
# - Accept any input length
# - Produce any output length
# - Process data incrementally
# - Maintain security through the capacity portion of the state
class KeccakSponge:
    # Let's break down the KeccakSponge initialization.
    # When called from KeccakHash, it's initialized like this:
    # sponge = KeccakSponge(
    #     bitrate_bits=1088,        # How many bits to process at once
    #     width=1600,               # Total state size (bitrate + capacity)
    #     padfn=multirate_padding,  # How to pad incomplete blocks
    #     permfn=keccak_f           # Core permutation function
    # )
    # The different components are:
    # 1. State (KeccakState)
    #    - 5x5 matrix of 64-bit values
    #    - Total size = 1600 bits
    #    - Split into:
    #      - Rate region (1088 bits): for data input/output
    #      - Capacity region (512 bits): for security
    # 2. Padding function (multirate_padding)
    #    - Adds padding when input isn't a multiple of block size
    #    - Ensures input can be processed in complete blocks
    # 3. Permutation Function (keccak_f)
    #    - Core cryptographic transformation
    #    - Mixes the state after each block is absorbed
    # 4. Buffer
    #    - Temporary storage for incoming data
    #    - Holds data until we have a full block (136 bytes)
    #    - Then processes full blocks and keeps remainder
    # Visual representation:
    #
    # Input Data → Buffer → Full Blocks → State Matrix
    #                                ┌─────────┐
    #                                │ 5x5     │
    #                                │ state   │ ← Permutation (keccak_f)
    #                                └─────────┘
    def __init__(self, bitrate, width, padfn, permfn):
        # Create the internal state matrix (5x5 array)
        self.state = KeccakState(bitrate, width)
        # Store the padding function (multirate_padding)
        self.padfn = padfn
        # Store the permutation function (keccak_f)
        self.permfn = permfn
        # Intialize empty buffer for incoming data
        self.buffer = []

    # Key points of copy():
    # 1. Uses Python's deepcopy from the copy module
    # 2. Creates a completely independent copy of the sponge, including:
    #    - The state (5x5 matrix)
    #    - The buffer contents
    #    - Function references (padfn, permfn)
    # Why deep copy vs shallow copy?
    # # Shallow copy (copy.copy) would share nested objects
    # shallow = copy.copy(sponge)
    # shallow.state[0][0] = 1 # Would affect the original sponge!
    # 
    # # Deep copy (copy.deepcopy) creates independent copies of everything
    # deep = deepcopy(sponge)
    # deep.state[0][0] = 1    # Original sponge unaffected
    #
    # This method is crucial for digest() to work correctly:
    #
    # def digest(self):
    #     final = self.copy()    # Create independent copy
    #     final.absorb_final()   # Modify the copy
    #     # Original sponge remains unchanged for future updates
    #
    # Without a deep copy, modifying the state during finalization would affect the original sponge, preventing further updates to the hash.
    def copy(self):
        return deepcopy(self)

    # The absorb_block method processes one full block of input data. 
    def absorb_block(self, block_bytes):
        # Verify block is exactly one rate-portion size (For keccak256: 1088 bits = 136 bytes)
        assert len(block_bytes) == self.state.bitrate_bytes # 136
        # XOR the input block into the state's rate portion. More precisely:
        # - Convert bytes to lane values
        # - XOR them into the states rate portion
        # - Leave capacity portion untouched
        self.state.absorb(block_bytes)
        # Apply the Keccak-f[1600] permutation to mix the state:
        # - Theta (θ): diffusion
        # - Rho (ρ): rotation
        # - Pi (π): rearrangement
        # - Chi (χ): nonlinearity
        # - Iota (ι): add constants
        #
        # Visual representation:
        #
        # Input Block (136 bytes)
        #        ↓
        #     ┌─────────┐
        #     │ State   │ ← XOR input with state
        #     │ Matrix  │
        #     └─────────┘
        #        ↓
        # Keccak-f permutation
        #        ↓
        #     ┌─────────┐
        #     │ Updated │
        #     │ State   │
        #     └─────────┘
        self.permfn(self.state)

    # Let's break down the absorb method, which handles incoming data:
    # - Buffer accumulates data until there's enough for a full block
    # - Processes as many full blocks as possible 
    # - Keeps remainder in bufffer for next update
    # - For keccak256, blocks are 136 bytes (1088 bits)
    #
    # Example flow:
    # 
    # # If bitrate_bytes = 136 (keccak256's block size), and we receive 200 bytes, then:
    # 
    # # Initial state:
    # buffer = []
    # data_bytes = [byte1, byte2, ..., byte200]
    # 
    # # After self.buffer += data_types:
    # buffer = [byte1, byte2, ..., byte200] # 200 bytes
    # 
    # # First iteration (process first 136 bytes):
    # absorb_block([byte1, ..., byte136])
    # buffer = [byte137, ..., byte200]      # 64 bytes remain
    # # No more full blocks (64 < 136), so loop ends. Remaining 64 bytes stay in buffer for the next update
    #
    #
    # Visual representation:
    #
    # Input Data
    #     ↓
    # [Buffer] → Full Block (136 bytes) → absorb_block()
    #     ↓
    # Remaining Data (stays in buffer)
    #
    def absorb(self, data_bytes):
        # Add new data to existing buffer
        self.buffer += data_bytes

        # Process full blocks (136 bytes each)
        while len(self.buffer) >= self.state.bitrate_bytes:
            # Take next block (self.buffer[: self.state.bitrate_bytes]) and process it (self.absorb_block(...))
            self.absorb_block(self.buffer[: self.state.bitrate_bytes])
            # Remove processed data from buffer
            self.buffer = self.buffer[self.state.bitrate_bytes :]

    # This method handles the final block of data. The method works in three steps:
    # 1. Create padded block:
    #    - Start with remaining data in buffer
    #    - Add padding using multirate_padding function
    # 2. Process the final padded block
    # 3. Clear the buffer
    # 
    # Here's an example:
    #
    # # If bitrate_bytes = 136 and buffer has 100 bytes:
    #
    # # 1. Calculate padding:
    # buffer_size = 100
    # needed_padding = multirate_padding(100, 136)  # Creates padding bytes
    # # Padding ensures final block is exactly 136 bytes
    # 
    # # 2. Create final block:
    # padded = buffer_bytes + padding_bytes         # Now exactly 136 bytes
    #
    # # 3. Process it:
    # absorb_block(padded)                          # XOR into state and apply permutation
    #
    # # 4. Clear buffer for potential reuse
    # buffer = []
    #
    # Visual representation:
    #
    # Buffer Data (incomplete block)
    #          ↓
    # [Padding Function] → Additional Bytes
    #          ↓
    # Complete Block (136 bytes)
    #          ↓
    #     absorb_block()
    #          ↓
    # Clear Buffer
    #
    # The absorb_final step is crucial because:
    # - All blocks must be complete (136 bytes)
    # - Padding ensures cryptographic security
    # - Final block must be properly processed 
    # - Buffer needs to be cleared for potential reuse
    def absorb_final(self):
        # 1. Create padded block:
        #    - Start with remaining data in buffer
        #    - Add padding using multirate_padding function
        padded = self.buffer + self.padfn(len(self.buffer), self.state.bitrate_bytes)
        # 2. Process the final padded block
        self.absorb_block(padded)
        # 3. Clear the buffer
        self.buffer = []

    # The squeeze_once method extracts a single block of output from the state.
    # This is like wringing out a sponge:
    # 1. First squeeze get some output (rate portion)
    # 2. Then mix the state (permutation)
    # 3. Ready for another squeeze if needed
    # Why permute after squeezing?
    # - Ensures each squeeze produces a different output
    # - Necessary when more than one block of output is needed
    # - Maintains security properties of the hash function
    #
    # Here's a visualization of the process:
    # State Matrix
    #     ↓
    # Extract rate portion (136 bytes)
    #     ↓
    # Apply permutation (keccak_f)
    #     ↓
    # Return extracted block
    # 
    # Note: The squeeze_once method is used by the squeeze method to get multiple blocks (see next method).
    def squeeze_once(self):
        # 1. Extract bytes from the rate portion of the state
        out_block = self.state.squeeze()
        # 2. Apply the permutation function (keccak_f)
        self.permfn(self.state)
        # 3. Return the extracted block
        return out_block

    # The squeeze method extracts a specified number of bytes from the state. The key points are:
    # - Each squeeze_once() returns a full rate block (136 bytes)
    # - We might get more bytes than needed
    # - Final slice ensures exact length
    # - For keccak256, we need 32 bytes (256 bits)
    # - Therefore, one squeeze_once() is already enough since 32 < 136
    #
    # Here's a visual representation:
    #
    # State → Extract Block → Permute → Extract Block → ...
    #          ↓
    #          Output Hash
    #
    # Example:
    #
    # # For keccak256, we need 32 bytes (256 bits)
    # length = 32
    # # First iteration:
    # out = []                 # empty
    # squeeze_once()           # gets 136 bytes
    # out = [byte1...byte136]  # more than we need
    # # Return just the first 32 bytes
    # return out[:32]
    def squeeze(self, length):
        # Initialize empty output array
        out = []
        # Keep squeezing blocks until we have enough bytes
        while len(out) < length:
            out += self.squeeze_once()  # Each squeeze_once() gives us 136 bytes
        # Return exactly the number of bytes requested 
        return out[:length]

# --------------------------------------------------------------------
#                          Keccak-256 Class
# --------------------------------------------------------------------
# Main interface for computing keccak256 hashes.
class KeccakHash:
    # Key points of the initialization code:
    # 1. Parameters:
    #    - Output size: 256 bits (32 bytes)
    #    - Rate (r): 1088 bits (136 bytes) - determines input block size
    #    - Capacity (c): 512 bits - affects security level
    #    - Total width: r + c = 1600 bits (200 bytes)
    # 2. Sponge Construction:
    #      State size (1600 bits)
    #      ┌────────────────────┐
    #      │     Rate (1088)    │ ← Data goes in/out here
    #      ├────────────────────┤
    #      │  Capacity (512)    │ ← Never directly touched
    #      └────────────────────┘
    # 3. Functions passed to Sponge:
    #    - multirate_padding: Adds proper padding to input
    #    - keccak_f: The main permutation function (Keccak-f[1600])
    # 4. Helper Values:
    #    - digest_size: Final output size in bytes (32)
    #    - block_size: Input block size in bytes (136)
    # Notice that these parameters are specifically chosen to match Ethereum's keccak256 implementation.
    # More precisely, the implementation of multirate_padding that is used here slightly differs from the 
    # one used in SHA3-256 which is the NIST standard.
    def __init__(self):
        # For Keccak256: rate=1088, capacity=512 => total 1600 bits
        # The final hash output will be 256 bits (32 bytes)
        self.output_bits = 256
        
        # The sponge construction uses two parameters:
        # 1. bitrate (r): how many bits we absorb/squeeze at a time
        # 2. capacity (c): security parameter, larger capacity = more secure, but slower algorithm
        # Total state size = r + c = 1600 bits (standard Keccak-f[1600])
        bitrate_bits = 1088
        capacity_bits = 512

        # Create the sponge construction with:
        self.sponge = KeccakSponge(
            # rate (r): how much data to absorb at once
            bitrate_bits,
            # width: total state size (1600 bits)
            bitrate_bits + capacity_bits,
            # padding function to use
            multirate_padding,
            # permutation function (Keccak-f[1600])
            keccak_f,
        )

        # Convert number of bits to number of bytes for convenience:
        # 256/8 = 32 bytes
        self.digest_size = bits2bytes(self.output_bits)
        # 1088/8 = 136 bytes
        self.block_size = bits2bytes(bitrate_bits)

    # This method takes bytes as input, converts the bytes to a list of integers (list(data)), and
    # feeds these bytes into the sponge's absorption phase.
    # For example, list(b"ABC") = [65, 66, 67].
    # 
    # Key points:
    # - You can call update() multiple times
    # - Data is buffered until there's enough for a full block (136 bytes)
    # - Remaining data stays in buffer until more is added or hash is finalized
    # - The actual absorption happens block by block
    # 
    # Example of multiple updates:
    #
    # h = KeccakHash()
    # h.update(b"Hello")
    # h.update(b" ")
    # h.update(b"World")
    #
    # This is the same as:
    # h.update(b"Hello World")
    def update(self, data: bytes):
        self.sponge.absorb(list(data))

    # Computes the final hash digest
    def digest(self) -> bytes:
        # Make a copy of current sponge state.
        # This allows you to keep calling update() after digest() and ensures you don't modify the original state
        #
        # Without the copy, consider this scenario:
        #  h = KeccakHash()
        #  h.update(b"Hello")
        #  
        #  # First digest
        #  digest1 = h.digest()    # This would modify the original state
        #  
        #  # Add more data 
        #  h.update(b" World")     # We want to continue using the same hash object
        #
        #  # Second digest
        #  digest2 = h.digest()    # This would fail or give wrong results because the state was already finalized
        #
        # Now, with the copy, we have the following situation instead:
        #  h = KeccakHash()
        #  h.update(b"Hello")
        #
        #  # First digest (works on a copy)
        #  digest1 = h.digest()    # Original state is preserved
        #
        #  # Add more data (still possible because original state wasn't modified)
        #  h.update(b"World")
        #
        #  # Second digest (works on another copy)
        #  digest2 = h.digest()    # Get correct hash of "Hello World"
        #
        # Note that this pattern matches Python's standard hashlib behavior:
        #  import hashlib
        #  h = hashlib.sha256()
        #  h.update(b"Hello")
        #  digest1 = h.digest()
        #  h.update(b" World")
        #  digest2 = h.digest()    # Should work the same way
        #
        # As a conclusion, the copy is necessary because the finalization process (padding and final absorption) modifies the state, but
        # we want to keep the original state intact for potential future updates.
        final = self.sponge.copy()
        # Add padding and process final block (see absorb_final()'s comments for an in-depth explanation)
        final.absorb_final()
        # Extract 32 bytes (256 bits) from the state
        out_bytes = final.squeeze(self.digest_size)  # digest_size = 32
        # Convert list of integers to bytes object
        return KeccakState.ilist2bytes(out_bytes)

    # Simple convenience method that converts the raw bytes from digest() into a hexadecimal string.
    # Raw digests might look like: b'\x1a\x2b\x3c...' (32 bytes)
    # After the conversion, they become: "1a2b3c..." (64 characters, since each byte becomes 2 hex digits)
    def hexdigest(self) -> str:
        return self.digest().hex()

# This function implements the main keccak256 hashing operation.
# It follows the common pattern seen in cryptographic hash functions: initialize -> update -> finalize.
def keccak256(data: bytes) -> bytes:
    # Creates a new hash instance, initializes the internal state (5x5 array), and sets up 
    # parameters (1088-bit rate, 512-bit capacity)
    h = KeccakHash()
    # Takes the input bytes, feeds them into the sponge construction, and processes
    # them through the absorption phase.
    h.update(data)
    # Finalizes the hash (adds padding), squeezes out 32 bytes (256 bits), and returns the raw hash values as bytes.
    return h.digest()

# This is a simple helper function that combines two operations. It:
# 1. Takes bytes as input (data: bytes).
# 2. Calls keccak256(data) which returns the raw hash digest as bytes.
# 3. Converts those bytes to a hexadecimal string using Python's built-in hex() method.
# 4. Returns the hex string (-> str).
def keccak256_hex(data: bytes) -> str:
    return keccak256(data).hex()

# --------------------------------------------------------------------
#                                Main
# --------------------------------------------------------------------

def main():
    # Hardcoded input: "2K53cuR1tY"
    # Creates a bytes object (note the b prefix)
    # In Python, bytes is an immutable sequence of integers beetween 0-255.
    # The default encoding of the b"<some string>" syntax is ASCII.
    # For UTF-8 encoding, use the following instead:
    # msg = "some string".encode('utf-8')
    msg = b"2K53cuR1tY"
    # Apply keccak256 to the bytes msg object.
    digest_hex = keccak256_hex(msg)
    # Print the hash digest in hex encoding.
    print(f"keccak256(\"2K53cuR1tY\") = {digest_hex}")

if __name__ == "__main__":
    main()
