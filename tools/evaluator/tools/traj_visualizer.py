import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib
import sys
import numpy as np
import math
import OpenEXR
import Imath


def readTraj(filename):
	traj = []
	with open(filename) as f:
		for l in f.readlines():
			x = float(l.split(" ")[3])
			y = float(l.split(" ")[7])
			z = float(l.split(" ")[11])
			traj.append(np.array([x, y, z]))
	return traj


def readMap(filename):
	map = []
	with open(filename) as f:
		for l in f.readlines():
			id = int(l.split(" ")[0])
			x = float(l.split(" ")[1])
			y = float(l.split(" ")[2])
			z = float(l.split(" ")[3])
			map.append([id, x, y, z])
	return map


def readDepth(filename):
	img = golden = OpenEXR.InputFile(filename)
	redstr = img.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
	dw = img.header()['dataWindow']
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
	depth = np.fromstring(redstr, dtype=np.float32)
	return depth.mean()


def visualize_errors(gt, vo, vo2, gt2):
	errors = []
	for i, v in enumerate(vo):
		errors.append(math.sqrt((gt[i] - v).dot((gt[i] - v))))

	num_of_kf = len(errors)
	if vo2 is not None:
		errors2 = []
		for i, v in enumerate(vo2):
			errors2.append(math.sqrt((gt2[i] - v).dot((gt2[i] - v))))
		num_of_kf = min(num_of_kf, len(errors2))
		plt.plot(range(num_of_kf), errors2[:num_of_kf], label="event based evaluation", color='b')

	plt.plot(range(num_of_kf), errors[:num_of_kf], label="evaluation", color='k')
	plt.xlabel('time, [tick]')
	plt.ylabel('error, [m]')
	plt.ylim(0)
	plt.legend()
	plt.show()

	return np.array(errors).mean()


def visualize_traj(gt, vo, vo2):
	gt = np.transpose(np.array(gt))
	vo = np.transpose(np.array(vo))

	if vo2 is not None:
		vo2 = np.transpose(np.array(vo2))

	num_of_kf = len(vo[0])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(vo[0][:num_of_kf], vo[1][:num_of_kf], vo[2][:num_of_kf], color='r', label="evaluation")
	if vo2 is not None:
		ax.plot(vo2[0], vo2[1], vo2[2], color='b', label="event based evaluation")
	ax.plot(gt[0][:num_of_kf], gt[1][:num_of_kf], gt[2][:num_of_kf], color='g', label="ground truth")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.set_zlim((-0.1, 0.1))
	plt.legend()
	plt.show()


def visualize_map(map):
	map_filtered = []
	threshold = 300
	for lm in map:
		if abs(lm[1]) < threshold and abs(lm[2]) < threshold and abs(lm[3]) < threshold:
			map_filtered.append(lm)

	fig = plt.figure()
	distance = []
	for v in map_filtered:
		distance.append(math.sqrt(v[1]**2 + v[2]**2 + v[3]**2))

	ax = fig.add_subplot(111, projection='3d')

	cm = plt.get_cmap("jet")
	cNorm = matplotlib.colors.Normalize(vmin=min(distance), vmax=max(distance))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

	map_filtered = np.transpose(np.array(map_filtered))
	ax.scatter(map_filtered[1], map_filtered[2], map_filtered[3], 
			c=scalarMap.to_rgba(distance), s=5, marker="s")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	# ax.set_xlim((-50, 50))
	# ax.set_ylim((-50, 50))
	# ax.set_zlim((25, 70))
	plt.show()


if __name__ == '__main__':
	gt = readTraj(sys.argv[1] + "/groundtruth_aligned.txt")
	vo = readTraj(sys.argv[1] + "/vo_trajectory.txt")
	map = readMap(sys.argv[1] + "/map.txt")

	mean_depth = 1
	try:
		mean_depth = readDepth(sys.argv[1] + "/depthmaps/frame_00000000.exr")
	except:
		print("No depth provided")

	vo2 = None
	gt2 = None
	if len(sys.argv) == 3:
		vo2 = readTraj(sys.argv[2] + "/vo_trajectory.txt")
		gt2 = readTraj(sys.argv[2] + "/groundtruth_aligned.txt")

	mean_error = visualize_errors(gt, vo, vo2, gt2)
	visualize_traj(gt, vo, vo2)
	visualize_map(map)

	print("Mean relative error: {:02f}%".format(mean_error / mean_depth * 100))