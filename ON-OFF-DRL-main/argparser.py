"""
Code from: https://github.com/gohsyi/cluster_optimization
"""

############################### Import libraries ###############################
import argparse

parser = argparse.ArgumentParser()

# environment setting
parser.add_argument('-n_servers', type=int, default=10)
parser.add_argument('-n_resources', type=int, default=2)
parser.add_argument('-n_tasks', type=int, default=2000,
                    help='Use all tasks by default.')
# Update power parameters (in watts, not kW)
parser.add_argument('-P_0', type=int, default=87, help='Idle power in watts (e.g., 87W)')
parser.add_argument('-P_100', type=int, default=145, help='Full load power in watts (e.g., 145W)')

# Update time parameters (in seconds, but ensure energy calculations use hours)
parser.add_argument('-T_on', type=int, default=30, help='Time to turn on in seconds')
parser.add_argument('-T_off', type=int, default=30, help='Time to turn off in seconds')

# Adjust weights for meaningful impact
parser.add_argument('-w1', type=float, default=1.0, help='Weight for power term in reward')
parser.add_argument('-w2', type=float, default=1.0, help='Weight for latency term in reward')
parser.add_argument('-w3', type=float, default=1.0, help='Weight for penalty term in reward')

args = parser.parse_args()

# Convert power to kW for energy calculations
args.P_0 = args.P_0 / 1000  # Convert to kW
args.P_100 = args.P_100 / 1000  # Convert to kW