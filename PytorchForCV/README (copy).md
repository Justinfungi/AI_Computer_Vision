# AI_ComputerVision

- Inspired by Dr Kai Han
- For recording the progress of CV learning

### Architectures

#### MLP
	class MLP(nn.Module):
		 def __init__(self):
		 super(MLP, self).__init__()
		 self.flatten = nn.Flatten()
		 self.linear_relu_stack = nn.Sequential(
		 nn.Linear(28*28, 512),
		 nn.ReLU(),
		 nn.Linear(512, 512),
		 nn.ReLU(),
		 nn.Linear(512, 10),
		 )
		 def forward(self, x):
		 x = self.flatten(x)
		 logits = self.linear_relu_stack(x)
		 return logits
	
#### LeNet		
	class LeNet(nn.Module):
		def __init__(self,num_classes=0):
			super(LeNet,self).__init__()
			self.conv1 = nn.Conv2d(3,6,5)
			self.conv2 = nn.Conv2d(6,16,5)
			self.fc_1 = nn.Linear(16*5*5,120)
			self.fc_2 = nn.Linear(120,84)
			self.fc_3 = nn.Linear(84,num_classes)
			
		def forward(self,x):
			out = F.relu(self.conv1(x))
			out = F.max_pool2d(out,2)
			out = F.relu(self.conv2(out))
			out = F.max_pool2d(out,2)
			out = out.view(out.size(0),-1)
			out = F.relu(self.fc_1(out))
			out = F.relu(self.fc_2(out))
			out = self.fc_3(out)
			return out
			
	def lenet(num_classes):
	return LeNet(num_classes=num_classes)
		
		
		
		
		
		
		
		
		
		
		
