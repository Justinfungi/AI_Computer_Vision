# AI_ComputerVision

- Inspired by Dr Kai Han
- For recording the progress of CV learning

## Dataset:

### Usually dataset config:

-Data
  - train
    - ClassA
      image_0001.jpg
      image_0002.jpg
      ...
    - ClassB
    ...
  - test
  
  The ClassA folder: naming will be the class label

### Load class dicts
	import glob
	class_names = [name[16:] for name in glob.glob('./data/train/*')] # Grab all name, select the part of the name
	class_names = dict(zip(range(len(class_names)), class_names)) # turn the class name into the corespondend class num
	print("class_names: %s " % class_names)
	
### Load the data
	def load_dataset(path, num_per_class, typeofdata):
	    data = []
	    labels = []
	    for id, class_name in class_names.items():
		print(f"{typeofdata} - Loading images from class: %s" % id)
		img_path_class = glob.glob(path + class_name + '/*')
		if num_per_class > 0:
		    img_path_class = img_path_class[:num_per_class]
		labels.extend([id]*len(img_path_class))
		for filename in img_path_class:
		    data.append(cv2.imread(filename, 0))
	    return data, labels
	    
	n_train_samples_per_class = 100
	n_test_samples_per_class = 100
	train_data, train_label = load_dataset('data/train/', n_train_samples_per_class, "Training dataset")
	test_data, test_label = load_dataset('data/test/', n_test_samples_per_class, "Testing dataset")

	# print data size
	n_train = len(train_label)
	print("n_train: %s" % n_train)
	n_test = len(test_label)
	print("n_test: %s" % n_test)
	
### Resize to tiny image

	tiny_image_size = (16, 16)
	train_data_tiny = list(map(lambda x: cv2.resize(x, tiny_image_size, interpolation=cv2.INTER_AREA).flatten(), train_data))
	train_data_tiny = np.stack(train_data_tiny)
	train_label = np.array(train_label)
	test_data_tiny = list(map(lambda x: cv2.resize(x, tiny_image_size, interpolation=cv2.INTER_AREA).flatten(), test_data))
	test_data_tiny = np.stack(test_data_tiny)
	

# Pytorch implementation
	
### DataLoader
#### Most simplified custmization
	import torch.utils.data as data
	
	class CusDatasetLoader(Dataset):
	    def __init__(self,x,y):
		self.x_data = x
		self.y_data = y
	  
	    def __len__(self):
		return self.len

	    def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]
		
#### Simple data
	data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
