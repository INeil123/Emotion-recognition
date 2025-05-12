import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# 确保已安装pillow
try:
    from PIL import Image
except ImportError:
    print("正在安装pillow...")
    os.system('pip install pillow')
    from PIL import Image
    
# 数据集路径
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# 图像参数
img_size = 48
batch_size = 64

# 数据增强与生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# 检查数据集目录是否存在
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError(f"请确保 '{train_dir}' 和 '{test_dir}' 目录存在")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# 构建模型
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator
)

# 保存模型
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/emotion_model.h5')
print("模型已保存到 models/emotion_model.h5")