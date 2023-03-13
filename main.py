from processData import *
import numpy as np
import matplotlib.pyplot as plt


images, labels = get_mnist()
w_i_h1 = np.random.uniform(-0.5, 0.5, (30, 784)) #  input -> hidden
w_h1_h2 = np.random.uniform(-0.5, 0.5, (20, 30)) #  hidden -> hidden
w_h2_o = np.random.uniform(-0.5, 0.5, (10, 20)) #  hidden -> output
b_i_h1 = np.zeros((30, 1))
b_h1_h2 = np.zeros((20, 1))
b_h2_o = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 10
for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)

        h1_pre = b_i_h1 + w_i_h1 @ img
        h1 = 1 / (1 + np.exp(-h1_pre))
        h2_pre = b_h1_h2 + w_h1_h2 @ h1
        h2 = 1 / (1 + np.exp(-h2_pre))
        o_pre = b_h2_o + w_h2_o @ h2
        o = 1 / (1 + np.exp(-o_pre))

        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))


        delta_o = o - l
        w_h2_o += -learn_rate * delta_o @ np.transpose(h2)
        b_h2_o += -learn_rate * delta_o
        delta_h2 = np.transpose(w_h2_o) @ delta_o * (h2 * (1 - h2))
        w_h1_h2 += -learn_rate * delta_h2 @ np.transpose(h1)
        b_h1_h2 += -learn_rate * delta_h2
        delta_h1 = np.transpose(w_h1_h2) @ delta_h2 * (h1 * (1 - h1))
        w_i_h1 += -learn_rate * delta_h1 @ np.transpose(img)
        b_i_h1 += -learn_rate * delta_h1

    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    h1_pre = b_i_h1 + w_i_h1 @ img.reshape(784, 1)
    h1 = 1 / (1 + np.exp(-h1_pre))
    h2_pre = b_h1_h2 + w_h1_h2 @ h1
    h2 = 1 / (1 + np.exp(-h2_pre))
    o_pre = b_h2_o + w_h2_o @ h2
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(o.argmax())
    plt.show()