import argparse
import numpy as np
import random
from pathlib import Path

sum_file = open('result/summary.csv', mode='w')
sum_file.write('model,vec,learning_rate,epoch,loss,acc\n')
for model in ['vit']:
    for vec in ['single', 'double']:
        for rate in ['0.001',]:
            epoch2result = {}
            for i in range(1, 6):
                file_name = f"{vec}-{model}-{rate}-{i}"
                path = file_name = f"result/evals/{file_name}"
                lines = open(path).readlines()
                result = []
                for line in lines:
                    if line.startswith('final'):
                        line = line[:-1].split(' ')
                        acc = float(line[1])
                        epoch, loss = line[0].split('-')
                        epoch = int(epoch[6:])
                        loss = float(loss)
                    else:
                        line = line[:-1].split(' ')
                        epoch, loss = line[0].split('-')
                        acc = float(line[1])
                        epoch = int(epoch)
                        loss = float(loss)
                    result.append([epoch, loss, acc])
                
                result = sorted(result, key=lambda x: x[0])
                for res in result:
                    if res[0] not in epoch2result:
                        epoch2result[res[0]] = []
                    epoch2result[res[0]].append(res[1:])

            for i, key in enumerate(sorted(list(epoch2result.keys()))):
                ave = np.mean(epoch2result[key], axis=0)
                ave = np.round(ave, 3)
                if len(epoch2result[key]) == 5:
                    print(key, len(epoch2result[key]), list(ave))
                    line = f'{model},{vec},{rate},{key},{ave[0]},{ave[1]}\n'
                    sum_file.write(line)
sum_file.close()