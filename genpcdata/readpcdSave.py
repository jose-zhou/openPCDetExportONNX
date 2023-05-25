import open3d

import numpy as np
import os
import time

def read_pcd(file_path):
	pcd = open3d.io.read_point_cloud(file_path)
	# print(np.asarray(pcd.points))
	# colors = np.asarray(pcd.colors) * 255
	points = np.asarray(pcd.points)
	# print(points.shape, colors.shape)
	return points


if __name__ =='__main__':
	pcdFilePath = '/media/jose/3fe6116f-510d-4aaa-a3ce-5f28635b2709/projectDatas/C5/datas/cloud_pcd_cheku/cloud_pcd/cloud_340.pcd'
	points = read_pcd(pcdFilePath)
	print(points.shape)
	np.save('my_data.npy', points)