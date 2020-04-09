# Primeiro exemplo de código contido no livro
# Deep Learning with Python, de François Chollet

# importando dataset mnist de imagens de digitos escritos à mão
from keras.datasets import mnist

# o dataset mnist vem precarregado no Keras, na forma de 4 arrays Numpy
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# prints de infos dos arrays
print('Shape train_images: ')
print(train_images.shape)
print('len train_labels: ' + str(len(train_labels)))
print(train_labels)

print('Shape test_images: ')
print(test_images.shape)
print('len test_labels: ' + str(len(test_labels)))
print(test_labels)

# O workflow será o seguinte: Primeiro, nós alimentaremos a rede neural com os dados de treinamento,
# train_images e train_labels. A rede então irá aprender a associar imagens e labels. Finalmente, nós
# iremos pedir que a rede produza predições para as imagens de teste, test_images, e nós verificaremos
# quando essas predições batem com os lables de test_labels

# O core building block de um rede neural é a camada, um módulo de processamento de dados
# que pode ser entendido como um filtro de dados. Alguns dados entram, e saem em uma forma
# mais útil. Especificamente, camadas extraem representações de seus dados de entrada -
# felizmente, representações essas que são mais significativas para o problema em questão.
# A maior parte do que se entende por deep-learning consiste em encadear camadas simples
# que irão implementar uma forma de destilação progressiva de dados. Um modelo de deep-learning
# é como uma peneira para processamento de dados, feito de uma sucessão de filtros de dados
# crescentemente refinados - a saber, as camadas.

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# passo de compilação
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# preparando os dados de imagem (image_data) -> mudando shape e tipo do array
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# precisamos também codificar os labels
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Pronto! All set. Vamos treinar a rede - no Keras, isso é feito via o método fit.
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Após treinar, também precisamos checar se o modelo performa bem no conjunto de testes, 
# via o método evaluate
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)

# Salvando o modelo em um arquivo
network.save("model.h5")
print("Saved model to disk")