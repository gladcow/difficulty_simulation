#!/usr/bin/env python3

import argparse
import datetime
import math
import random
import statistics
import sys
import time
from collections import namedtuple
from functools import partial


def bits_to_target(bits):
    size = bits >> 24
    assert size <= 0x1d

    word = bits & 0x00ffffff
    assert 0x8000 <= word <= 0x7fffff

    if size <= 3:
        return word >> (8 * (3 - size))
    else:
        return word << (8 * (size - 3))


MAX_BITS = 0x1d00ffff
MAX_TARGET = bits_to_target(MAX_BITS)


def target_to_bits(target):
    assert target > 0
    if target > MAX_TARGET:
        print('Warning: target went above maximum ({} > {})'
              .format(target, MAX_TARGET), file=sys.stderr)
        target = MAX_TARGET
    size = (target.bit_length() + 7) // 8
    mask64 = 0xffffffffffffffff
    if size <= 3:
        compact = (target & mask64) << (8 * (3 - size))
    else:
        compact = (target >> (8 * (size - 3))) & mask64

    if compact & 0x00800000:
        compact >>= 8
        size += 1

    assert compact == (compact & 0x007fffff)
    assert size < 256
    return compact | size << 24


def bits_to_work(bits):
    return (2 << 255) // (bits_to_target(bits) + 1)


def target_to_hex(target):
    h = hex(target)[2:]
    return '0' * (64 - len(h)) + h


TARGET_1 = bits_to_target(486604799)

TARGET_BLOCK_TIME = 150

INITIAL_TIMESTAMP = 1503430225
INITIAL_HASHRATE = 500    # In PH/s.
INITIAL_TARGET = int(pow(2, 256) // (INITIAL_HASHRATE * 1e15) // TARGET_BLOCK_TIME)
INITIAL_BITS = target_to_bits(INITIAL_TARGET)
INITIAL_HEIGHT = 481824
INITIAL_SINGLE_WORK = bits_to_work(INITIAL_BITS)

State = namedtuple('State', 'height wall_time timestamp bits chainwork '
                   'hashrate msg')

states = []


def print_headers():
    print(', '.join(['Height', 'Block Time', 'Unix', 'Timestamp',
                     'Difficulty (bn)', 'Implied Difficulty (bn)',
                     'Hashrate (PH/s)', 'Comments']))


def print_state():
    state = states[-1]
    block_time = state.timestamp - states[-2].timestamp
    t = datetime.datetime.fromtimestamp(state.timestamp)
    difficulty = TARGET_1 / bits_to_target(state.bits)
    implied_diff = TARGET_1 / ((2 << 255) / (state.hashrate * 1e15 * TARGET_BLOCK_TIME))
    print(', '.join(['{:d}'.format(state.height),
                     '{:d}'.format(block_time),
                     '{:d}'.format(state.timestamp),
                     '{:%Y-%m-%d %H:%M:%S}'.format(t),
                     '{:.2f}'.format(difficulty / 1e9),
                     '{:.2f}'.format(implied_diff / 1e9),
                     '{:.0f}'.format(state.hashrate),
                     state.msg]))


def median_time_past(states):
    times = [state.timestamp for state in states]
    return sorted(times)[len(times) // 2]


def next_bits_dgw3(msg, block_count):
    ''' Dark Gravity Wave v3 from Dash'''
    block_reading = -1  # dito
    counted_blocks = 0
    past_difficulty_avg = 0
    past_difficulty_avg_prev = 0
    i = 1
    while states[block_reading].height > 0:
        if i > block_count:
            break
        counted_blocks += 1
        if counted_blocks <= block_count:
            if counted_blocks == 1:
                past_difficulty_avg = bits_to_target(states[block_reading].bits)
            else:
                past_difficulty_avg = ((past_difficulty_avg_prev * counted_blocks) +
                                       bits_to_target(states[block_reading].bits)) // \
                                      (counted_blocks + 1)
        past_difficulty_avg_prev = past_difficulty_avg
        block_reading -= 1
        i += 1
    target_time_span = block_count * TARGET_BLOCK_TIME
    actual_time_span = states[-1].timestamp - states[block_reading].timestamp
    if actual_time_span < (target_time_span // 3):
        actual_time_span = target_time_span // 3
    if actual_time_span > (target_time_span * 3):
        actual_time_span = target_time_span * 3
    target = past_difficulty_avg
    target = target // target_time_span
    target *= actual_time_span
    if target > MAX_TARGET:
        return MAX_BITS
    else:
        return target_to_bits(int(target))

def next_bits_xmr(msg, window):
    last_times = []
    last_difficulties = []
    for state in states[-window:]:
        last_times.append(state.timestamp)
        target = bits_to_target(state.bits)
        difficulty = TARGET_1 / target
        last_difficulties.append(difficulty)
    last_times = sorted(last_times)
    last_times = last_times[window // 6: -window // 6]
    last_difficulties = sorted(last_difficulties)
    last_difficulties = last_difficulties[window // 6: -window // 6]
    time_span = last_times[-1] - last_times[0]
    if time_span == 0:
        time_span = 1
    diff_sum = sum(last_difficulties)
    result_difficulty = (diff_sum * TARGET_BLOCK_TIME + time_span + 1) // time_span
    result_target = int(TARGET_1 // result_difficulty)
    if result_target > MAX_TARGET:
        return MAX_BITS
    return target_to_bits(result_target)


def next_bits_simple_align(msg):
    last_block_time = states[-1].timestamp - states[-2].timestamp
    if last_block_time < 1:
        last_block_time = 1
    target_time = (2 * TARGET_BLOCK_TIME) - last_block_time
    if target_time < 1:
        target_time = 1
    prev_target = bits_to_target(states[-1].bits)
    k = last_block_time / target_time

    if k < 0.7:
        k = 0.7
    if k > 1.3:
        k = 1.3

    result_target = int(prev_target * k)
    if result_target > MAX_TARGET:
        return MAX_BITS
    return target_to_bits(result_target)


def next_bits_lwma(msg, window):
    last_times = []
    last_difficulties = []
    for state in states[-(window + 1):]:
        last_times.append(state.timestamp)
        target = bits_to_target(state.bits)
        difficulty = TARGET_1 / target
        last_difficulties.append(difficulty)

    T = 120
    N = window
    L = 0
    sum_3_ST = 0

    prev_max_TS = last_times[0]
    for i in range(1, N + 1):
        if last_times[i] > prev_max_TS:
            max_TS = last_times[i]
        else:
            max_TS = prev_max_TS + 1
        ST = min(6 * T, max_TS - prev_max_TS)
        prev_max_TS = max_TS
        L += ST * i
        if i > N - 3:
            sum_3_ST += ST

    D = sum(last_difficulties) * T * (N + 1) /(2 * L)
    D = int(D * 99 / 100)

    prev_D = last_difficulties[-1]
    if D < int(67 * prev_D / 100):
        D = int(67 * prev_D / 100)
    if D > int(150 * prev_D / 100):
        D = int(150 * prev_D / 100)

    if sum_3_ST < (8 * T) / 10:
        D = int(prev_D * 1.1)

    if D < 1000:
        D = 1000

    result_target = int(TARGET_1 // D)
    if result_target > MAX_TARGET:
        return MAX_BITS
    return target_to_bits(result_target)


def next_bits_lwma2(msg, window):
    last_times = []
    last_difficulties = []
    for state in states[-(window + 1):]:
        last_times.append(state.timestamp)
        target = bits_to_target(state.bits)
        difficulty = TARGET_1 / target
        last_difficulties.append(difficulty)

    T = 120
    N = window
    L = 0
    sum_3_ST = 0

    prev_max_TS = last_times[0]
    for i in range(1, N + 1):
        if last_times[i] > prev_max_TS:
            max_TS = last_times[i]
        else:
            max_TS = prev_max_TS + 1
        ST = min(6 * T, max_TS - prev_max_TS)
        prev_max_TS = max_TS
        L += ST * i
        if i > N - 3:
            sum_3_ST += ST

    D = sum(last_difficulties) * T * (N + 1) /(2 * L)
    D = int(D * 99 / 100)

    prev_D = last_difficulties[-1]
    if D < int(67 * prev_D / 100):
        D = int(67 * prev_D / 100)
    if D > int(150 * prev_D / 100):
        D = int(150 * prev_D / 100)

    if sum_3_ST < (8 * T) / 10:
        D = int(prev_D * 1.1)

    if D < 1000:
        D = 1000

    result_target = int(TARGET_1 // D)
    if result_target > MAX_TARGET:
        return MAX_BITS
    return target_to_bits(result_target)


def block_time(mean_time):
    # Sample the exponential distn
    sample = random.random()
    lmbda = 1 / mean_time
    res = math.log(1 - sample) / -lmbda
    if res < 1:
        res = 1
    return res


def next_step(algo, scenario):
    # First figure out our hashrate
    msg = []
    hashrate = scenario.hashrate(msg, **scenario.params)
    # Calculate our dynamic difficulty
    bits = algo.next_bits(msg, **algo.params)
    target = bits_to_target(bits)
    # See how long we take to mine a block
    mean_hashes = pow(2, 256) // target
    mean_time = mean_hashes / (hashrate * 1e15)
    time = int(block_time(mean_time) + 0.5)
    wall_time = states[-1].wall_time + time
    # Did the difficulty ramp hashrate get the block?
#    if random.random() < (scenario.dr_hashrate / hashrate):
#        timestamp = median_time_past(states[-11:]) + 1
#    else:
    timestamp = wall_time

    chainwork = states[-1].chainwork + bits_to_work(bits)

    # add a state
    states.append(State(states[-1].height + 1, wall_time, timestamp,
                        bits, chainwork, hashrate, ' / '.join(msg)))


Algo = namedtuple('Algo', 'next_bits params')

Algos = {
    'dgw3': Algo(next_bits_dgw3, {  # 24-blocks, like Dash
        'block_count': 24,
    }),
    'xmr': Algo(next_bits_xmr, {
       'window': 720
    }),
    'sa': Algo(next_bits_simple_align, {
    }),
    'lwma': Algo(next_bits_lwma2, {
        'window': 60
    })
}


def const_hashrate(msg, base_rate):
    return base_rate


def random_oscillations_hashrate(msg, base_rate,  amplitude):
    return base_rate * (1 + amplitude * (random.random() - 0.5))


def inout_hashrate(msg, base_rate, additional_rate):
    height = len(states)
    if(height // 100) % 2:
        return base_rate
    else:
        return base_rate + additional_rate


def fake_ts_hashrate(msg, base_rate):
    return base_rate


Scenario = namedtuple('Scenario', 'hashrate params')

Scenarios = {
    'const': Scenario(const_hashrate, {
        'base_rate': INITIAL_HASHRATE
    }),
    'random': Scenario(random_oscillations_hashrate, {
        'base_rate': INITIAL_HASHRATE,
        'amplitude': 0.5
    }),
    'increase': Scenario(const_hashrate, {
        'base_rate':  2 * INITIAL_HASHRATE
    }),
    'decrease': Scenario(const_hashrate, {
        'base_rate': 0.5 * INITIAL_HASHRATE
    }),
    'inout': Scenario(inout_hashrate, {
        'base_rate': INITIAL_HASHRATE,
        'additional_rate': INITIAL_HASHRATE
    }),
    'fake_timestamp': Scenario(fake_ts_hashrate, {
        'base_rate': INITIAL_HASHRATE
    })
}


def run_one_simul(algo, scenario, print_it):
    states.clear()

    # Initial state is after 2020 steady prefix blocks
    N = 2020
    for n in range(-N, 0):
        state = State(INITIAL_HEIGHT + n, INITIAL_TIMESTAMP + n * TARGET_BLOCK_TIME,
                      INITIAL_TIMESTAMP + n * TARGET_BLOCK_TIME,
                      INITIAL_BITS, INITIAL_SINGLE_WORK * (n + N + 1),
                      INITIAL_HASHRATE, '')
        states.append(state)

    # Run the simulation
    if print_it:
        print_headers()
    for n in range(10000):
        next_step(algo, scenario)
        if print_it:
            print_state()

    # Drop the prefix blocks to be left with the simulation blocks
    simul = states[N:]

    block_times = [simul[n + 1].timestamp - simul[n].timestamp
                   for n in range(len(simul) - 1)]
    difficulties = [TARGET_1 / bits_to_target(simul[n].bits)
                   for n in range(len(simul) - 1)]
    return block_times, difficulties


def main():
    '''Outputs CSV data to stdout.   Final stats to stderr.'''

    parser = argparse.ArgumentParser('Run a mining simulation')
    parser.add_argument('-a', '--algo', metavar='algo', type=str,
                        choices=list(Algos.keys()),
                        default='pid', help='algorithm choice')
    parser.add_argument('-s', '--scenario', metavar='scenario', type=str,
                        choices=list(Scenarios.keys()),
                        default='const', help='scenario choice')
    parser.add_argument('-r', '--seed', metavar='seed', type=int,
                        default=None, help='random seed')
    args = parser.parse_args()

    algo = Algos.get(args.algo)
    scenario = Scenarios.get(args.scenario)
    seed = int(time.time()) if args.seed is None else args.seed

    print("Algo %s,  scenario %s" % (args.algo, args.scenario))

    to_stderr = partial(print, file=sys.stderr)
    to_stderr("Starting seed {}".format(seed))

    random.seed(seed)
    block_times, difficulties = run_one_simul(algo, scenario, True)

    def stats(text, values):
        to_stderr('{} {}'.format(text, values))

    stats("Mean   block time", statistics.mean(block_times))
    stats("StdDev block time", statistics.stdev(block_times))
    stats("Median block time", sorted(block_times)[len(block_times) // 2])
    stats("Max    block time", max(block_times))

    stats("Mean   difficulty", statistics.mean(difficulties))
    stats("StdDev difficulty", statistics.stdev(difficulties))
    stats("Median difficulty", sorted(difficulties)[len(difficulties) // 2])
    stats("Max    difficulty", max(difficulties))


if __name__ == '__main__':
    main()
