# 配置函数
import argparse
parser=argparse.ArgumentParser(description='setting')
parser.add_argument('--ecglen',default=1024,type=int,help='original length')
parser.add_argument('--fs',default=500,type=float,help='signal sampling frequency')
parser.add_argument('--epochs',default=500,type=float,help='epochs')
parser.add_argument('--bs',default=2048,type=int,help='batch size')
parser.add_argument('--lr',default=1e-3,type=float,help='learning rate')
parser.add_argument('--patience',default=50)
parser.add_argument('--time_steps',default=100)
args = parser.parse_args()