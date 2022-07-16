import math
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle


NEXT_FRAME = 12


if __name__ == "__main__":
    # with open("./downstream/pitch_libritts/.cache/train-yaapt-labels-20.pkl", 'rb') as f:
    #     label = pickle.load(f)
    with open("./downstream/energy/.cache/train-labels-20.pkl", 'rb') as f:
        label = pickle.load(f)
    
    x_data, y_data = [], []
    for path, values in tqdm(label.items()):
        if len(values) <= NEXT_FRAME:
            continue
        for timestep in range(len(values) - NEXT_FRAME):
            t_x, t_y = timestep, timestep + NEXT_FRAME
            if values[t_x] == 0 or values[t_y] == 0:
                continue
            x_data.append(math.log(values[t_x]))
            y_data.append(math.log(values[t_y]))

    print("Total training data samples: ", len(x_data))
    
    # with open("./downstream/pitch_libritts/.cache/test-yaapt-labels-20.pkl", 'rb') as f:
    #     test_label = pickle.load(f)
    with open("./downstream/energy/.cache/test-labels-20.pkl", 'rb') as f:
        test_label = pickle.load(f)
    
    x_test_data, y_test_data = [], []
    for path, values in tqdm(test_label.items()):
        if len(values) <= NEXT_FRAME:
            continue
        for timestep in range(len(values) - NEXT_FRAME):
            t_x, t_y = timestep, timestep + NEXT_FRAME
            if values[t_x] == 0 or values[t_y] == 0:
                continue
            x_test_data.append(math.log(values[t_x]))
            y_test_data.append(math.log(values[t_y]))

    print("Total testing data samples: ", len(x_test_data))

    x_data = np.array(x_data).reshape(-1, 1)
    y_data = np.array(y_data)
    x_test_data = np.array(x_test_data).reshape(-1, 1)
    y_test_data = np.array(y_test_data)
    print(x_data.shape, y_data.shape, x_test_data.shape, y_test_data.shape)

    # Always scale the input. The most convenient way is to use a pipeline.
    print("Start fitting...")
    reg = make_pipeline(SGDRegressor(max_iter=1000, tol=1e-3, eta0=1e-3, penalty=None, n_iter_no_change=20, verbose=1))
    reg.fit(x_data, y_data)
    print("End fitting...")

    predictions = reg.predict(x_test_data)
    final_loss = np.mean((predictions - y_test_data) ** 2)
    print("Linear model MSE: ", final_loss)
