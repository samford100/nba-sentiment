import numpy as np
# import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf

'''
select the autoregression j = 5 (go back 5 games)
win(0) = regress([
    sen_com(-5), win(-5),
    sen_com(-4), win(-4),
    sen_com(-3), win(-3),
    sen_com(-2), win(-2),
    sen_com(-1), win(-1),
    sen_com(0), team_name
])
'''

"""
reads the csv to a map of team_name to date ordered data
"""


def generate_timeseries_df(df):
    groups = df.groupby(['team'])
    x = []
    for team, data in groups:
        vecs = gen_vecs(team, data, w=3)
        x = x + vecs
    # print(x)
    # df = pd.DataFrame(x, columns=["sen_com-4", "par_com-4", "won-4",
    #                               "sen_com-3", "par_com-3", "won-3",
    #                               "sen_com-2", "par_com-2", "won-2",
    #                               "sen_com-1", "par_com-1", "won-1",
    #                               "sen_com-0", "par_com-0", "won",
    #                               "team", "is_tanking"])
    # df = pd.DataFrame(x, columns=["sen_com-2", "won-2",
    #                               "sen_com-1", "won-1",
    #                               "sen_com-0", "won",
    #                               "team", "is_tanking"])

    df = pd.DataFrame(x, columns=["sen_com-2", "par_com-2", "won-2",
                                  "sen_com-1", "par_com-1", "won-1",
                                  "sen_com-0", "par_com-0", "won",
                                  "team", "is_tanking"])

    # print(df.head)
    return df


'''
generate vectors for each team with parameter j
'''


def gen_vecs(team, df, w):
    tanking_teams = ['Atlanta', 'Memphis', 'Sacramento', 'Dallas', 'Phoenix', 'Charlotte',
                     'LA Clippers', 'New York', 'Brooklyn', 'Chicago', 'LA Lakers', 'Orlando']

    # print(df)
    # x = [df[i:i+w] for i in range(len(df) - w)]
    x = []
    for i in range(len(df) - w):
        slice = df[i:i+w][['sen_com', 'par_com', 'won']]
        # print(slice)
        # print(slice.values.shape)
        vec = slice.values.reshape(1, w*3)
        vec = vec[0]
        vec = np.append(vec, [team])
        vec = np.append(vec, [team in tanking_teams])
        # vec = np.append(vec, team)
        # print(vec)
        x.append(vec)
        # break

    return x


def read_csv(file):
    df = pd.read_csv(file)
    # df = df.drop(['date'], axis=1)
    df = generate_timeseries_df(df)
    y = df[['won']]
    # X = df.drop(['won', 'date'], axis=1)
    X = df.drop(['won'], axis=1)
    Z = onehot(X)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     Z, y, test_size=0.2)

    splitter = round(len(Z) * .7)
    print(splitter)

    X_train, X_test = Z[:splitter], Z[splitter+1:]
    y_train, y_test = y[:splitter], y[splitter+1:]
    print(len(X_train))
    print(len(X_test))
    print(len(Z))
    # train_tf(X_train, y_train, X_test, y_test)

    y_hat = train(X_train, y_train.values.ravel(), X_test)
    right = 0
    for i in range(len(y_hat)):
        if y_hat[i] == y_test.values[i]:
            right += 1

    print('' + str(right / len(y_hat)) + "%")


def onehot(X):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = label_encoder.fit_transform(X['team'])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    inverted = label_encoder.inverse_transform(
        [np.argmax(onehot_encoded[0, :])])
    s = pd.DataFrame(onehot_encoded)
    Z = pd.concat([X, s], axis=1)
    Z = Z.drop(['team'], axis=1)
    return Z


def multilayer_perceptron(x, weights, biases):
    # hidden layer 1 with relu activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # hidden layer 2 with relu activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def train_tf(X_train, y_train, X_test, y_test):
    # initialize variables
    num_inputs = 11
    print(num_inputs)
    num_hidden_1, num_hidden_2 = 10, 10
    num_classes = 2
    learning_rate = .001
    training_epochs = 100

    weights = {
        'h1': tf.Variable(tf.random_normal([num_inputs, num_hidden_1])),
        'h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # construct model with placeholders
    x_ = tf.placeholder("float", [None, num_inputs])
    y_ = tf.placeholder("float", [None, num_classes])
    model = multilayer_perceptron(x_, weights, biases)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y_))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    # train
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(X_train)/100)
            X_batches = np.array_split(X_train, total_batch)
            y_batches = np.array_split(y_train, total_batch)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = X_batches[i], y_batches[i]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x_: batch_x,
                                                              y_: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % 1 == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_train, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
    global result
    result = tf.argmax(model, 1).eval({x: X_test, y: y_test})


def train(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    # clf = SVC()
    # clf = RandomForestClassifier(n_estimators=100)
    clf = LogisticRegression()

    # hidden_layer_sizes = (40, 40, 30, 40, 30, 50)  # best 68% relu
    # hidden_layer_sizes = (40, 40, 30, 50, 30, 40)
    # hidden_layer_sizes = (60, 12, 12)
    hidden_layer_sizes = (20, 20)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu',
    #                     hidden_layer_sizes=hidden_layer_sizes, random_state=1)

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    print(hidden_layer_sizes)
    return y_hat


read_csv('./nba_sentiment.csv')
