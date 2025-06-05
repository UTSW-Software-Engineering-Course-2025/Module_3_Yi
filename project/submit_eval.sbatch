#!/bin/bash
#SBATCH --job-name=genehop_eval
#SBATCH --output=./Logs/genehop_tools_eval_%j.out
#SBATCH --error=./Logs/genehop_tools_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1                     # 单节点
#SBATCH --ntasks=1                    # 单任务
#SBATCH --cpus-per-task=16
#SBATCH --partition=512GB


# 激活 conda 环境
source activate 



# 运行主脚本
python hop_example.py