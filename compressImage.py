import scipy as sp
import scipy.misc
import numpy as np
from os import listdir, path, makedirs
import matplotlib.pyplot as plt

def PCA(data_mtrx,num_components):
	#tinh cac vector ky vong
	mean_mtrx = np.mean(data_mtrx,axis = 1)
	#tru ma tran du lieu cho cac ky vong
	for c_idx in range(data_mtrx.shape[1]):
		data_mtrx[:,c_idx] = data_mtrx[:,c_idx] - mean_mtrx
	#tinh vector rieng, tri rieng va sap xep theo thu tu giam dan
	eigenvectors, eigenvalues, s = np.linalg.svd(data_mtrx, full_matrices=False)
	#chon k vector rieng co tri rieng lon nhat
	eigenvectors = eigenvectors[:,0:num_components]
	eigenvalues = eigenvalues[0:num_components]
	y = np.dot(eigenvectors.T,data_mtrx)
	return eigenvectors, y, mean_mtrx
def reconstruct(eigenvectors,weights,data_mtrx,mean_mtrx,img_shape):
	recons_imgs = list()
	for c_idx in range(data_mtrx.shape[1]):
		ri = mean_mtrx + np.dot(eigenvectors,weights[:,c_idx])
		recons_imgs.append(ri.reshape(img_shape))
	return recons_imgs
def main():
	img_dir = "C:/Users/hoang/OneDrive/Desktop/compressImage/yalefaces"
	out_dir = path.join(img_dir,"_extract")
	eface_dir = path.join(img_dir,"_compress")
	eigenvector_dir = path.join(img_dir,"_eigenvector")
	weight_dir = path.join(img_dir,"_weights")
	datamtrix_dir = path.join(img_dir,"_datamtrx")
	img_names = listdir(img_dir)
	img_list = list()
	#tao list anh
	for fn in img_names:
		if (not path.isdir(path.join(img_dir,fn)) and not fn.startswith('.') and fn != "Thumbs.db"):
			img = scipy.misc.imread(path.join(img_dir,fn),True)
			img_list.append(img)
	img_shape = img_list[0].shape
	#chuyen n hinh anh thanh ma tran d*n
	imgs_mtrx = np.array([img.flatten() for img in img_list]).T
	#
	#
	f = open("data.txt","w")
	for i in range(imgs_mtrx.shape[0]):
		for j in range(imgs_mtrx.shape[1]):
			f.write(str(imgs_mtrx[i,j]) + " ")
		f.write("\n")
	f.close()
	#
	#
	eigenvectors,weights,mean_mtrx = PCA(imgs_mtrx,8)
	#
	#
	#print(eigenvectors.shape[1])
	for i in range(eigenvectors.shape[1]):
		f = open("eiface"+str(i)+".txt","w");
		for j in range(eigenvectors.shape[0]):
			f.write(str(eigenvectors[j,i]) + " ")
		f.close()
	#
	print(weights.shape)
	for i in range(weights.shape[1]):
		f = open("w"+str(i)+".txt","w");
		for j in range(weights.shape[0]):
			f.write(str(weights[j,i]) + " ")
		f.close()
	#
	f = open("mean.txt","w")
	for i in range(mean_mtrx.shape[0]):
			f.write(str(mean_mtrx[i,])+ " ")
	#print(eigenvectors.shape[0])
	f.close()
	recons_imgs = reconstruct(eigenvectors,weights,imgs_mtrx,mean_mtrx,img_shape)

	if not path.exists(out_dir): makedirs(out_dir)
	if not path.exists(eface_dir): makedirs(eface_dir)
	#if not path.exists(eigenvector_dir): makedirs(eigenvector_dir)
	#if not path.exists(weight_dir): makedirs(weight_dir)
	#if not path.exists(datamtrix_dir): makedirs(datamtrix_dir)
	for idx, img in enumerate(recons_imgs):
		sp.misc.imsave(path.join(out_dir,"img_"+str(idx)+".jpg"),img)
	for idx in range(eigenvectors.shape[1]):
		sp.misc.imsave(path.join(eface_dir,"eface"+str(idx)+".jpg"),eigenvectors[:,idx].reshape(img_shape))
	#print(weights.shape)
	#print(mean_mtrx.shape)
if __name__ == '__main__':
	main()