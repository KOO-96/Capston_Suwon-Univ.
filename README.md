# Capston_Suwon-univ-1.

# week1 
[8/30 ~ 9/6] [팀 병합]

구성준 [[Github]](https://github.com/KOO-96)  
김백운 [[Github]](https://github.com/kimbw0615)   
김재윤 [[Github]](https://github.com/KIMJAEYUN67)   

# week2
week2 [9/6 ~ 9/13]
## 첫 회의 : 방향성 잡기 -> (공모전 vs 프로젝트(연구))  
공모전 : 결과를 보여 줄 수 있고, **목적이 뚜렷하다**라는 장점, but 올라온 공모전이 많지 않고 기간이 너무 길다는 단점  
프로젝트(연구) : **목적을 정하지 않는 이상 애매모호**해진다라는 단점, but 실력 증진에 도움이 될 것이라는 장점  

-> 기나긴 회의 끝에 실력증진을 위한 프로젝트(연구) 선택

# week3 [9/13 ~ 9/19] ~ week9 [10/18 ~ 10/25]  
> 중간발표   
 
objective: 패션 아이템을 활용한 추천 알고리즘  
data : Fashion mnist dataset  

CNN딥러닝을 이용한 classification  
result : 20epoch, 91%  

---
### fashion_mnist dataset이란?
![fashion_mnist dataset](https://codetorial.net/tensorflow/_images/fashion_MNIST_sample.png)

Fashion MNIST 데이터셋은 위 그림과 같이 운동화, 셔츠, 샌들과 같은 작은 이미지들의 모음이며, 기본 MNIST 데이터셋과 같이 열 가지로 분류될 수 있는 28×28 픽셀의 이미지 70,000개로 이루어져 있습니다.
0 : T-shirt/top\
1 : Trouser\
2 : Pullover\
3 : Dress\
4 : Coat\
5 : Sandal\
6 : Shirt\
7 : Sneaker\
8 : Bag\
9 : Ankel boot
Fashion_mnist데이터는 총 10개의 라벨로 구성이 되어있습니다.  

CNN 클래스 작성   
```
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

class CNN:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
        maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flatten = Flatten()(maxpool2)
        fc1 = Dense(128, activation='relu')(flatten)
        output_layer = Dense(self.num_classes, activation='softmax')(fc1)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        return x_train, y_train, x_test, y_test

    def train(self, x_train, y_train, x_test, y_test, epochs=20, batch_size=128):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
        return history

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        return test_loss, test_acc
```
시각화를 통해 테스트 이미지의 예측값(실제값)을 확인
![image](https://github.com/KOO-96/Capston_Suwon-univ./assets/113090595/a081e1dd-599d-4deb-9d4a-da33294c78f2)

학부생 딥러닝(자연어처리방식)에서 배운 것을 활용해보고자 벡터화를 통해 유사행렬을 만든 다음 유사한 이미지를 추출  
Risk : 세션 종료로 인한 실행 불가능.   

Risk 해결방안으로 이미지 원-핫 인코딩(One-hot encoding)을 통해 확인 -> Risk: 1과 0 값으로만 나온다, 정답이 틀린 경우도 1로 값이 나오는 경우가 발생.

Result
![image](https://github.com/KOO-96/Capston_Suwon-univ./assets/113090595/f06bdfb7-e756-4ef4-8865-cf9d01a97e25)

 
# week10 [ ~~ 12/6]
> ![최종발표]()
