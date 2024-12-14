import matplotlib.pyplot as plt
import random

def generate_linear_regression_data(num_samples=100, slope=1.0, intercept=0.0, interval_start = -10, interval_end = 10, noise_level=0.1) -> list[tuple[float, float]]:
    data = []
    for _ in range(num_samples):
        x = random.uniform(interval_start, interval_end)
        y = slope * x + intercept
        noise = y * random.uniform(-noise_level, noise_level)
        y += noise

        data.append((x, y))
    
    return data


def mean_squared_error(y: list[float], y_hat: list[float]) -> float:
    mse: float = 0
    for i in range(len(y)):
        mse += (y[i] - y_hat[i])**2
    return mse / len(y)


def gradient_descent(m: float, b: float, training_data: list[tuple[float, float]], learning_rate: float) -> tuple[float, float]:
    target = [data_point[1] for data_point in training_data] 
    actual = [m * data_point[0] + b for data_point in training_data]
    x_values = [data_point[0] for data_point in training_data]

    n = len(training_data)

    m_gradient: float = 0
    b_gradient: float = 0

    for x, y, y_hat in zip(x_values, target, actual):
        m_gradient += -(2 / n) * x * (y - y_hat)
        b_gradient += -(2 / n) * (y - y_hat)

    m -= m_gradient * learning_rate
    b -= b_gradient * learning_rate

    return (m, b)




# training data
training_data: list[tuple] = generate_linear_regression_data(100, 0.4, -2, noise_level=0.5)
x_values, y_values = zip(*training_data)

plt.scatter(x_values, y_values, color='blue', label='Data Points')


# gradient descent
lr = 0.01
m = 0.8
b = 2

for i in range(1000):
    m, b = gradient_descent(m, b, training_data, lr)
    y_hat = [m * x + b for x in x_values]
    mse = mean_squared_error(y_values, y_hat)
    
    if i % 100 == 0:  
        print(f"Epoch: {i}, MSE: {mse:.4f}")

plt.plot(x_values, y_hat, color='red', label='Fitted Line')
plt.legend()
plt.show()
