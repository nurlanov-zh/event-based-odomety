import matplotlib.pyplot as plt
import sys
import numpy as np

def readTraj(filename):
	traj = []
	with open(filename) as f:
		for l in f.readlines():
			z = float(l.split(" ")[2])
			y = float(l.split(" ")[1])
			x = float(l.split(" ")[0])
			traj.append([x, y, z])
	return traj


def visualize(gt, vo):
	gt = np.transpose(np.array(gt))
	vo = np.transpose(np.array(vo))

	for i, axis in enumerate(vo):
		for j, _ in enumerate(axis):
			vo[i][j] *= (1 / 0.001561)

	for i, axis in enumerate(gt):
		for j, _ in enumerate(axis):
			gt[i][j] *= (1 / 0.001561)

	plt.ylim((-550, 750))
	plt.plot(range(len(vo[0])), gt[0][:len(vo[0])], label="ground truth x", linestyle=":", color='g')
	plt.plot(range(len(vo[0])), vo[0], label="evaluation x", linestyle=":", color='r')

	plt.plot(range(len(vo[1])), gt[1][:len(vo[1])], label="ground truth y", linestyle="--", color='g')
	plt.plot(range(len(vo[1])), vo[1], label="evaluation y", linestyle="--", color='r')

	plt.plot(range(len(vo[2])), gt[2][:len(vo[2])], label="ground truth z", linestyle="-", color='g')
	plt.plot(range(len(vo[2])), vo[2], label="evaluation z", linestyle="-", color='r')
	plt.xlabel('time, [tick]')
	plt.ylabel('coordinate, [scaled values]')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	gt = readTraj(sys.argv[1] + "/groundtruth_aligned.txt")
	vo = readTraj(sys.argv[1] + "/vo_trajectory.txt")

	visualize(gt, vo)
