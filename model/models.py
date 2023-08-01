# import the necessary packages
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))
	# return our model
	return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		# flatten the volume, then FC => RELU => BN => DROPOUT
		x = Flatten()(x)
		x = Dense(16)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Dropout(0.5)(x)
		# apply another FC layer, this one to match the number of nodes
		# coming out of the MLP
		x = Dense(4)(x)
		x = Activation("relu")(x)
		# check to see if the regression node should be added
		if regress:
			x = Dense(1, activation="linear")(x)
		# construct the CNN
		model = Model(inputs, x)
		# return the CNN
		return model
	
class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        # 如果是实线，不执行以下if语句，如果是虚线，则执行以下if语句，将W(x)核F(x)维度一致
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()
        
        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        # 如果是实线，不执行以下if语句，如果是虚线，则执行以下if语句，将W(x)核F(x)维度一致
        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

	def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
		super(ResNet18, self).__init__()
		self.num_blocks = len(block_list)  # 共有几个block
		self.block_list = block_list
		self.out_filters = initial_filters
		# 第一个卷积层：CBA
		self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
		self.b1 = BatchNormalization()
		self.a1 = Activation('relu')
		self.blocks = tensorflow.keras.models.Sequential()
		# 构建ResNet网络结构
		# 外层循环层数由参数列表的循环个数决定，如model = ResNet18([2, 2, 2, 2])，则循环4次，有4个橙色块，即8个ResNet块
		for block_id in range(len(block_list)):  # 第几个橙色块
			for layer_id in range(block_list[block_id]):  # 第几个ResNet块

				if block_id != 0 and layer_id == 0:  # 除第一个橙色块，以下三个橙色块的第一个ResNet块外，都是虚线，定义residual_path=True
					block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
				else:  # 第一个橙色块，以下三个橙色块的第一个ResNet块是实线，定义residual_path=False
					block = ResnetBlock(self.out_filters, residual_path=False)
				self.blocks.add(block)  # 将构建好的block加入resnet
			self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍

		# 全局池化
		# self.p1 = tensorflow.keras.layers.GlobalAveragePooling2D()
		# Dense层
		# self.f1 = tensorflow.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tensorflow.keras.regularizers.l2())



	def call(self, width, height, depth):
		inputShape = (height, width, depth)
		chanDim = -1
		# define the model input
		inputs = Input(shape=inputShape)
		x = self.c1(inputs)
		x = self.b1(x)
		x = self.a1(x)
		x = self.blocks(x)
		# change here into dense layer and add flatten
		x = Flatten()(x)
		y = Dense(4)(x)
		#x = self.p1(x)
		#y = self.f1(x)

		# self added layer
		y = Activation("relu")(y)
		
		model = Model(inputs, y)
		# return y
		return model