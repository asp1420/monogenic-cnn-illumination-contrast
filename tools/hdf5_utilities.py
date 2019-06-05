import sys,cv2,os,h5py,csv,time,numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt

def save_HDF5(x,y,path):
    start = time.clock() 
    try:
        m,n,c=x[0].shape;
    except  ValueError:
        m,n=x[0].shape;
        c=0
    full_path=path+''+'_'+str(c)+'-'+str(m)+'-'+str(n)+'.h5'
    hf = h5py.File(full_path, 'w')
    hf.create_dataset('x', data=x, dtype=np.uint8,compression="gzip",compression_opts=1)
    hf.create_dataset('y', data=y, dtype=np.uint8,compression="gzip",compression_opts=1)
    hf.close()
    elapsed = time.clock()
    elapsed = elapsed - start
    print ("The data was saved succesfully in: ",full_path,"using: ", round(elapsed,2),' seconds')
    
    
def load_HDF5(full_path):
    start = time.clock() 
    print('Loading data from',full_path,'...', end='')
    hf = h5py.File(full_path, 'r')
    x = np.array(hf.get('x'))
    y = np.array(hf.get('y'))
    hf.close()
    elapsed = time.clock()
    elapsed = elapsed - start
    print ("finished using: ", round(elapsed,2), " seconds")
    return x,y

def obtain_data_from_images_and_csv(path_dataset,path_csv):
    start = time.clock() 
    names = sorted(os.listdir(path_dataset));
    y = read_csv(path_csv,header=None)
    try:
        m,n,c=cv2.imread(path_dataset+names[0],cv2.IMREAD_UNCHANGED).shape;
    except  ValueError:
        m,n=cv2.imread(path_dataset+names[0],cv2.IMREAD_UNCHANGED).shape;
        c=0
#    x=np.zeros((len(names),m*n*c))
    x=[]
    i=0
    for i in range(0,len(names)):
        tmp_img=cv2.imread(path_dataset+str(i)+'.png',cv2.IMREAD_UNCHANGED)
        if c==3:
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        x.append(tmp_img)
#        x[i]=tmp_img.flatten()
        i=i+1
        percent = float(i) / len(names)
        hashes = '#' * int(round(percent * 20))
        spaces = ' ' * (20 - len(hashes))
        sys.stdout.write("\rLoading {0} images and .csv to vectors x and y:  [{1}] {2}%".format(len(names),hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()
    elapsed = time.clock()
    elapsed = elapsed - start
    print ("\tfinished in: ", round(elapsed,2), " seconds")
    return x,y

def combine_hdf5(path_h5_1,path_h5_2,path_res):
    x1,y1=load_HDF5(path_h5_1)
    x2,y2=load_HDF5(path_h5_2)
    try:
        x3=np.concatenate((x1,x2), axis=3)
    except ValueError:
        x2=x2[:, :, :,np.newaxis]
        x3=np.concatenate((x1,x2), axis=3)
    save_HDF5(x3,y1,path_res)

def save_images_and_csv_from_data(x,y,path_x,path_y):
    start = time.clock() 
    fo=csv.writer(open(path_y,'w'))
    try:
        m,n,c=x[0].shape;
    except  ValueError:
        m,n=x[0].shape;
        c=0
    for i in range(len(x)):
        row=[i,y[i]]
        fo.writerow(row)
        if c==3:
            x[i] = cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_x+str(i)+'.png',x[i])
        percent = float(i+1) / len(x)
        hashes = '#' * int(round(percent * 20))
        spaces = ' ' * (20 - len(hashes))
        sys.stdout.write("\rSaving {0} images:  [{1}] {2}%".format(len(x),hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()
    elapsed = time.clock()
    elapsed = elapsed - start
    print ("\tfinished using: ", round(elapsed,2), " seconds")
    
#def create_rgbd_images(path_rgb,path_depth,path_res):
#    
        

    
    
    
if __name__=='__main__':
#    x,y=obtain_data_from_images_and_csv('D:/Dehazing monogenic/data/cifar_depth_dc/','D:/Dehazing monogenic/data/label.csv')    
#    save_HDF5(x,y,'D:/Dehazing monogenic/data/cifar_depth_dc')
    x1,y1=load_HDF5('D:/Dehazing monogenic/data/cifar_rgb_3-32-32.h5')
#    print(y1[1])
#    plt.imshow(x1[1])
#    plt.show()
#    x2,y2=load_HDF5('D:/Dehazing monogenic/data/cifar_dehaze_dc_3-32-32.h5')
#    print(y2[1])
#    plt.imshow(x2[1])
#    plt.show()
#    x3,y3=load_HDF5('D:/Dehazing monogenic/data/cifar_dehaze_dl_3-32-32.h5')
#    print(y3[1])
#    plt.imshow(x3[1])
#    plt.show()
#    combine_hdf5('D:/Dehazing monogenic/data/cifar_monogenic_dehaze_dl_3-32-32.h5','D:/Dehazing monogenic/data/cifar_dehaze_dl_3-32-32.h5','D:/Dehazing monogenic/data/cifar_dehaze_Dl_and_monogenic')
###    save_images_and_csv_from_data(x,y,'d2/','d/label.csv')
##    


