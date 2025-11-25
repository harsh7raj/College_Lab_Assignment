def show_lab1():
    code = """
# Program 1: Basic Neuron Model with Different Activation Functions

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')  

# Basic Neuron

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def forward(self, x):
        return 1 / (1 + np.exp(-(np.dot(x, self.weights) + self.bias)))

input_size = 5
num_samples = 100
inputs = np.random.rand(num_samples, input_size)

neuron = Neuron(input_size)
outputs = neuron.forward(inputs)
print("Neuron outputs:\\n", outputs)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def softmax(x): return np.exp(x) / np.sum(np.exp(x))

# Plot 
def plot_activation(x, y, title):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Plots

x = np.linspace(-10, 10, 200)

plot_activation(x, sigmoid(x), "Sigmoid Activation Function")
plot_activation(x, tanh(x), "Hyperbolic Tangent Activation Function")
plot_activation(x, relu(x), "ReLU Activation Function")

t = np.linspace(-5, 5, 20)
plot_activation(t, softmax(t), "Softmax Activation Function")
"""
    print(code)
# This file contains all lab programs with show_labX() printers.

def show_lab2():
    code = """
# Program 2: Single Layer Perceptron

import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = yi - pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def predict(self, x):
        return 1 if (np.dot(x, self.weights) + self.bias) >= 0 else 0

# Example usage

X_train = np.array([[2, 3], [1, 4], [3, 5], [4, 2]])
y_train = np.array([0, 0, 1, 1])

perceptron = Perceptron(learning_rate=0.01, epochs=100)
perceptron.fit(X_train, y_train)

new_data_point = np.array([2, 4])
prediction = perceptron.predict(new_data_point)

print(f"Prediction for {new_data_point}: {prediction}")
"""
    print(code)

def show_lab3():
    code = """
# Program 3: Multilayer Perceptron (MLP) Demonstrating Backpropagation Algorithm

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load & prepare data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

X = df.iloc[:, :-1]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Plot accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss")
plt.legend()
plt.grid(True)
plt.show()
"""
    print(code)


def show_lab4():
    code = """
# Program 4: Gradient Descent (GD) vs Stochastic Gradient Descent (SGD)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Load + preprocess
X, y = load_iris().data, load_iris().target.reshape(-1, 1)
y = OneHotEncoder(sparse_output=False).fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

# Model builder
def build(): 
    return Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

# Train GD
gd = build()
gd.compile(optimizer=SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
gd_hist = gd.fit(
    X_train, y_train, 
    epochs=50, batch_size=32, 
    validation_data=(X_test, y_test), 
    verbose=0
)

# Train SGD
sgd = build()
sgd.compile(optimizer=SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
sgd_hist = sgd.fit(
    X_train, y_train, 
    epochs=50, batch_size=1, 
    validation_data=(X_test, y_test), 
    verbose=0
)

# Plotter
def plot(h, key, title):
    plt.plot(h['accuracy'], label='Train '+key)
    plt.plot(h['val_accuracy'], label='Val '+key)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(h['loss'], label='Train '+key)
    plt.plot(h['val_loss'], label='Val '+key)
    plt.title(title + " Loss")
    plt.legend()
    plt.grid()
    plt.show()

plot(gd_hist.history, "GD", "GD Accuracy")
plot(sgd_hist.history, "SGD", "SGD Accuracy")
"""
    print(code)

def show_lab5():
    code = """
# Program 5: Dropout vs Gradient Clipping vs Early Stopping (Multitask Learning)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, callbacks
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split

# ===================== PART A =====================

# Dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

# Dropout model
drop_model = Sequential([
    Input(shape=(20,)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
drop_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
drop_hist = drop_model.fit(
    Xtr, ytr, epochs=30, batch_size=32,
    validation_data=(Xte, yte), verbose=0
)

# Gradient clipping model
clip_model = Sequential([
    Input(shape=(20,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
clip_model.compile(
    optimizer=Adam(clipnorm=1.0),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
clip_hist = clip_model.fit(
    Xtr, ytr, epochs=30, batch_size=32,
    validation_data=(Xte, yte), verbose=0
)

# Plot accuracy comparison
plt.plot(drop_hist.history['accuracy'], label='Dropout Train')
plt.plot(drop_hist.history['val_accuracy'], label='Dropout Val')
plt.plot(clip_hist.history['accuracy'], label='Clip Train')
plt.plot(clip_hist.history['val_accuracy'], label='Clip Val')
plt.legend()
plt.grid()
plt.title("Dropout vs Gradient Clipping Accuracy")
plt.show()


# ===================== PART B =====================

digits = load_digits()
X = digits.images / 16.0
y = digits.target
yp = y % 2  # parity (even/odd)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
ypr = yp[:len(ytr)]

# Show samples
plt.figure(figsize=(5, 5))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(Xtr[i], cmap='gray')
    plt.axis("off")
plt.show()

# Multitask model
inp = Input(shape=(8, 8))
x = Flatten()(inp)
x = Dense(64, 'relu')(x)
digit_out = Dense(10, 'softmax')(x)
parity_out = Dense(1, 'sigmoid')(x)

mt_model = tf.keras.Model(inp, [digit_out, parity_out])
mt_model.compile(
    optimizer="adam",
    loss=["sparse_categorical_crossentropy", "binary_crossentropy"],
    metrics=[["accuracy"], ["accuracy"]]
)

es = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

hist = mt_model.fit(
    Xtr, [ytr, ypr],
    validation_split=0.2,
    epochs=30,
    callbacks=[es],
    verbose=1
)

# Early stopping visualization
stop_ep = np.argmin(hist.history['val_loss']) + 1
plt.plot(hist.history['loss'], label='Train')
plt.plot(hist.history['val_loss'], label='Val')
plt.axvline(stop_ep, color='r', linestyle='--', label=f'Stopped @ Epoch {stop_ep}')
plt.legend()
plt.grid()
plt.title("Early Stopping - Loss Curve")
plt.show()
"""
    print(code)

def show_lab6():
    code = """
# Program 6: Adversarial Training, Tangent Distance, and Tangent Classifier

import numpy as np
import matplotlib.pyplot as plt

print("---- Implementing Adversarial Training, Tangent Distance and Tangent Classifier ----")

# ---------------- DATA ----------------
np.random.seed(0)
n = 200
X = np.random.randn(n, 2)
y = (X[:, 0] * X[:, 1] > 0).astype(int).reshape(-1, 1)

print("\\n[1] Adversarial Training Started...")

# ---------------- MODEL ----------------
W = np.random.randn(2, 1)
b = 0.0
sig = lambda z: 1 / (1 + np.exp(-z))
lr, eps = 0.1, 0.2

loss_history = []

# ---------------- ADVERSARIAL TRAINING ----------------
for i in range(1, 101):   # 100 iterations
    p = sig(X @ W + b)
    grad_x = (p - y) @ W.T
    X_adv = X + eps * np.sign(grad_x)

    X_all = np.vstack([X, X_adv])
    y_all = np.vstack([y, y])

    p_all = sig(X_all @ W + b)
    e = p_all - y_all

    W -= lr * X_all.T @ e / len(X_all)
    b -= lr * e.mean()

    loss = np.mean(e ** 2)
    loss_history.append(loss)

    if i <= 5 or i % 20 == 0:
        print(f"Iteration {i} loss = {loss}")

print("[1] Adversarial Training Completed.")

# ---------------- TANGENT DISTANCE ----------------
print("\\n[2] Computing Tangent Distance...")

def tan_dist(x, z):
    t = np.array([-x[1], x[0]])           
    a = np.dot(z - x, t) / np.dot(t, t)
    return np.linalg.norm((x + a * t) - z)

sample_td = tan_dist(X[0], X[1])
print(f"Tangent distance = {sample_td}")

# ---------------- TANGENT CLASSIFIER ----------------
print("\\n[3] Tangent Classifier Evaluation Started...")

def tan_classify(x, Xtr, ytr):
    d = [tan_dist(x, xi) for xi in Xtr]
    return ytr[np.argmin(d)]

pred = np.array([tan_classify(x, X, y.ravel()) for x in X]).reshape(-1, 1)
acc = (pred == y).mean()

print(f"[3] Tangent Classifier Accuracy: {acc*100:.2f}%")
print("\\n---- Program Successfully Implemented (Three Techniques Completed) ----")

# ---------------- LOSS PLOT ----------------
plt.plot(loss_history)
plt.title("Adversarial Training Loss (Loss vs Iterations)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()
plt.show()
"""
    print(code)

def show_lab7():
    code = """
# Program 7: MNIST Dataset Classification using CNN

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load + preprocess MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., None] / 255.0, x_test[..., None] / 255.0

# One-hot encode
y_train = tf.keras.utils.to_categorical(y_train)
y_test  = tf.keras.utils.to_categorical(y_test)

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)

# Evaluate
print("Test Accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])

# Plots
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()


OR



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Load + preprocess MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[...,None]/255.0, x_test[...,None]/255.0

# One-hot encode
y_train = tf.keras.utils.to_categorical(y_train)
y_test  = tf.keras.utils.to_categorical(y_test)

# Build CNN
model = Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    x_train, y_train,
    epochs=5, batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)

# Evaluate
print("Test Accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])

# Plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss"); plt.legend()

plt.tight_layout(); plt.show()
"""
    print(code)

def show_lab8():
    code = """
# Program 8: IMDB Sentiment Analysis using RNN

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import matplotlib.pyplot as plt

max_features, maxlen = 10000, 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test  = pad_sequences(x_test,  maxlen=maxlen)

model = Sequential([
    Embedding(max_features, 50, input_length=maxlen),
    SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", acc)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")

plt.tight_layout()
plt.show()
"""
    print(code)


def show_lab9():
    code = """
# Program 9: Yahoo Finance Stock Prediction using GRU

import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# ---- Load Data ----
ticker = "AAPL"   # change as needed
df = yf.download(ticker, period="5y")["Close"].values.reshape(-1, 1)

# ---- Scale ----
mn, mx = df.min(), df.max()
data = (df - mn) / (mx - mn)

# ---- Create sequences ----
seq = 60
X, y = [], []
for i in range(len(data) - seq):
    X.append(data[i:i+seq])
    y.append(data[i+seq])
X, y = np.array(X), np.array(y)

# ---- GRU Model ----
model = Sequential([
    GRU(64, input_shape=(seq, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# ---- Predictions ----
pred = model.predict(X)
pred_rescaled = pred * (mx - mn) + mn
y_rescaled = y * (mx - mn) + mn

# ---- Plot ----
plt.figure(figsize=(10,5))
plt.plot(y_rescaled, label="Actual")
plt.plot(pred_rescaled, label="Predicted")
plt.title(f"{ticker} Stock Price Prediction (GRU)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()
"""
    print(code)


def show_lab10():
    code = """
# Program 10: Stock Prediction using LSTM

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---- Load data ----
data = yf.download("AAPL", start="2010-01-01", end="2023-01-01")[['Open']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ---- Create sequences ----
X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

# ---- Train/Test split ----
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---- Build LSTM ----
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# ---- Train ----
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# ---- Predict ----
pred = scaler.inverse_transform(model.predict(X_test))
real = scaler.inverse_transform(y_test)

# ---- Plot ----
plt.plot(real, label='Real')
plt.plot(pred, label='Predicted')
plt.title("LSTM Stock Prediction")
plt.legend()
plt.show()
"""
    print(code)


