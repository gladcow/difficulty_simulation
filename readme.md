  Test script to test different difficulty adjustment algorithms in
different scenarios.

## Usage

test_mining.py [-h] [-a algo] [-s scenario] [-r seed]

 Optional arguments:
  -h, --help            show this help message and exit
  -a algo, --algo       algorithm choice
  -s scenario, --scenario  scenario choice
  -r seed, --seed       random seed

Available algorithms:
   * 'dgw3': Dash
   * 'xmr': Monero
   * 'sa': "simple align" method (planned block time + last block time = 2 * target block time)
   * 'lwma': LWMA-2 method
   * 'qwc' : Qwertycoin

Available scenarios:
   * 'const': constant network hashrate
   * 'random_oscillations': random oscillations of the network hashrate around given value
   * 'increase': increase network hashrate to given value (2 x initial)
   * 'decrease': decrease network hashrate to given value (0.5 x initial)
   * 'inout': some part of the whole network is in and out periodically

