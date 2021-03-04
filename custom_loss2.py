import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Nadam
from keras.constraints import min_max_norm, unit_norm, max_norm

def odds_loss_orig(y_true, y_pred):
    """
    The function implements the custom loss function - profit/loss per dollar stake.
    
    Inputs
    true : a vector of dimension batch_size, 5. A label encoded version of the output.
    pred : a vector of probabilities of dimension batch_size, 3.
    
    Returns 
    the loss value
    """
    """arr = np.array([[0.0, 0.0, 0.0]])
    arr[0, K.eval(K.argmax(y_pred))[0]] = 1
    y_pred = K.variable(arr, dtype='float32')"""
    
    player1_win = y_true[:,0:1]
    player2_win = y_true[:,1:2]
    no_bet = y_true[:,2:3]
    player1_odds = y_true[:,3:4]
    player2_odds = y_true[:,4:5]
    
    #gain_loss_vector = get_vector(K.eval(y_pred[0][0]), K.eval(y_pred[0][1]), K.eval(K.argmax(y_pred[0])), K.eval(player1_win[0,0]), K.eval(player2_win[0,0]), K.eval(player1_odds[0,0]), K.eval(player2_odds[0,0]))
    gain_loss_vector = K.concatenate([player1_win*(player1_odds-1) + player2_win*(-1), 
                                      player2_win*(player2_odds-1) + player1_win*(-1), 
                                      K.zeros_like(player1_odds)], axis=1)
    
    return -K.mean(K.sum(gain_loss_vector * y_pred, axis=1))
    

def odds_loss(y_true, y_pred):
    """
    The function implements the custom loss function - profit/loss per dollar stake.
    
    Inputs
    true : a vector of dimension batch_size, 5. A label encoded version of the output.
    pred : a vector of probabilities of dimension batch_size, 3.
    
    Returns 
    the loss value
    """
    player1_win = y_true[:,0:1]
    player2_win = y_true[:,1:2]
    player1_odds = y_true[:,2:3]
    player2_odds = y_true[:,3:4]
    
    gain_loss_vector = K.concatenate([player1_win*(player1_odds-1) + player2_win*(-1), 
                                      player2_win*(player2_odds-1) + player1_win*(-1)], axis=1)

    
    return -K.mean(K.sum(gain_loss_vector * y_pred, axis=1))
    

"""
if 1.3 < df.loc[y_test.index]['AvgP1'].iloc[2] and 1.3 < df.loc[y_test.index]['AvgP2'].iloc[2]:
    p1_win = y_test.iloc[2]
    p2_win = 1-y_test.iloc[2]
    no_bet = 0
else:
    p1_win = 0
    p2_win = 0
    no_bet = 1             
true = K.variable(np.array([[p1_win, p2_win, no_bet, df.loc[y_test.index]['AvgP1'].iloc[2], df.loc[y_test.index]['AvgP2'].iloc[2]]], dtype='float32'))
pred = K.variable(np.array([[0.5, 0.3, 0.2]]), dtype='float32')
K.eval(odds_loss(true, pred))
"""

def mae(y_true, y_pred):
    return -K.mean(y_true[:,0:2]*y_pred)
K.eval(mae(true, pred))

def total_loss(y_true, y_pred):
    return 0.5*odds_loss(y_true, y_pred) + 0.5*mae(y_true, y_pred)

def acc_score(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_pred[:], axis=1), K.argmax(y_true[:,:2], axis=1)))
K.eval(acc_score(true, pred))

"""
def get_model(input_dim, output_dim, base=1000, multiplier=0.25, p=0.2):
    inputs = Input(shape=(input_dim,))
    l = BatchNormalization()(inputs)
    l = Dropout(p)(l)
    n = base
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(p)(l)
    n = int(n * multiplier)
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(p)(l)
    n = int(n * multiplier)
    l = Dense(n, activation='relu')(l)
    outputs = Dense(output_dim, activation='softmax')(l)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss=odds_loss)
    return model
    
    model = get_model(14, 1)
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          epochs=1, batch_size=1, callbacks=[EarlyStopping(patience=25),
                                                ModelCheckpoint('odds_loss.hdf5',
                                                                save_best_only=True)])
"""

def build_fn(input_dim, output_dim):
    def create_model():
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, kernel_constraint=max_norm(3), kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, input_dim=input_dim, kernel_constraint=max_norm(3), kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        """model.add(Dense(8, kernel_constraint=max_norm(3), kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))"""
        model.add(Dense(output_dim, kernel_constraint=max_norm(3), kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        model.compile(optimizer='nadam', loss=odds_loss, metrics=[acc_score], run_eagerly=True) #loss=odds_loss
        return model
    return create_model


X_full.sort_index(inplace=True)
y_full = pd.Series(data['P1Win'].sort_index())

df = X_full.copy()
df['AvgP1'] = player1_avg_odds
df['AvgP2'] = player2_avg_odds

X_full = X_full[((1.2 < df['AvgP1'])&(df['AvgP1'] < 4)) & ((1.2 < df['AvgP2'])&(df['AvgP2'] < 4))]
y_full = y_full[((1.2 < df['AvgP1'])&(df['AvgP1'] < 4)) & ((1.2 < df['AvgP2'])&(df['AvgP2'] < 4))]

"""
# KEY: 0 = lose, 1 = win, 2 = no bet                                          
def get_data_orig(X_full, y_full):
    y = np.zeros((len(X_full), 5))
    i = 0
    for ind, item in y_full.iteritems():
        if 0.3 < abs(X_full.loc[ind]['OddsDiff']) < 2: # if both odds > 1.3, bet
            if item == 1:
                y[i,0] = 1.0
            elif item == 0:
                y[i,1] = 1.0
        else: # else, no bet
            y[i,2] = 1.0
        y[i,3] = df.loc[ind]['AvgP1']
        y[i,4] = df.loc[ind]['AvgP2']
        i += 1
    return X_full.values, y, y_full.values
"""


def get_data(X_full, y_full):
    y = np.zeros((len(X_full), 4))
    i = 0
    for ind, item in y_full.iteritems():
        if item == 1:
            y[i,0] = 1.0
        elif item == 0:
            y[i,1] = 1.0
        y[i,2] = df.loc[ind]['AvgP1']
        y[i,3] = df.loc[ind]['AvgP2']
        i += 1
    return X_full.values, y, y_full.values

X, y, outcome = get_data(X_full, y_full)

"""X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.09, shuffle=True, random_state=1, stratify=[val[0] for val in y])
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.09, shuffle=True, random_state=1, stratify=[val[0] for val in y])

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=1, stratify=[val[0] for val in train_y])
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, shuffle=True, random_state=1, stratify=[val[0] for val in train_y])
"""

# Assign more weight to Grand Slam samples, and less weight to ATP500 and ATP250 samples
sample_weight = pd.Series([1]*len(X_train), index=X_train.index)
grand_slams = X_train[atp_bet_data.loc[X_train.index]['Series']=='Grand Slam']
masters = X_train[(atp_bet_data.loc[X_train.index]['Series']=='Masters Cup')|(atp_bet_data.loc[X_train.index]['Series']=='Masters 1000')]
atp_500 = X_train[atp_bet_data.loc[X_train.index]['Series']=='ATP500']
atp_250 = X_train[atp_bet_data.loc[X_train.index]['Series']=='ATP250']


for ind in grand_slams.index:
    sample_weight.loc[ind] *= 1.2
for ind in masters.index:
    sample_weight.loc[ind] *= 0.7
for ind in atp_500.index:
    sample_weight.loc[ind] *= 0.7
for ind in atp_250.index:
    sample_weight.loc[ind] *= 0.5


clf = KerasClassifier(build_fn(9,2), batch_size=128, epochs=200, validation_data=(valid_x, valid_y), sample_weight=sample_weight.values, callbacks=[EarlyStopping(patience=15), ModelCheckpoint('odds_loss.hdf5', save_best_only=True)])

history = clf.fit(train_x, train_y)

"""
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, X_full, y_full, cv=cv)

kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(clf, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
"""
#71.8, 72.6
# weights, 179, 170, 261

# Get predictions for validation set
y_prob = [max(val) for val in clf.predict_proba(X_valid)]
y_pred = [np.argmax(val) for val in clf.predict_proba(X_valid)]

# Evaluate the model
# Get validation accuracy score
score = accuracy_score([np.argmax(val[:2]) for val in valid_y], y_pred)
print('Accuracy:', score)

# Get validation confusion matrix
print('Confusion matrix: ')
print(confusion_matrix([np.argmax(val[:2]) for val in valid_y], y_pred))

# Get profit
profit = 0
stake = 5
i = 0
count = 0
correct = 0

for ind, row in X_valid.iterrows():
    if y_pred[i] == 0 and 1.2 < df.loc[ind]['AvgP1'] and 1.2 < df.loc[ind]['AvgP2'] and y_prob[i] > 0.97:
        if y_valid[ind] == 1:
            player1_odds = df.loc[ind]['AvgP1']
            profit += (player1_odds-1)*stake
            correct += 1
            print(f"WIN: {player1[ind]} to beat {player2[ind]}")
        else:
            profit -= stake
            print(f"LOSS: {player1[ind]} to beat {player2[ind]}")
        count += 1
        print(round(profit, 2))
    elif y_pred[i] == 1 and 1.2 < df.loc[ind]['AvgP1'] and 1.2 < df.loc[ind]['AvgP2'] and y_prob[i] > 0.97:
        if y_valid[ind] == 0:
            player2_odds = df.loc[ind]['AvgP2']
            profit += (player2_odds-1)*stake
            correct += 1
            print(f"WIN: {player2[ind]} to beat {player1[ind]}")
        else:
            profit -= stake  
            print(f"LOSS: {player2[ind]} to beat {player1[ind]}")
        count += 1
        print(round(profit, 2))
    i += 1

print()
print("VALIDATION SET:")
print(f"Accuracy: {round(correct/count, 4)}")
print(f"Profit: ${round(profit, 2)}")
print(f"Return: {round(100*profit/(stake*count), 2)}%")
print(f"No. bets: {count}")
print(f"Avg. profit per bet: ${round(profit/count, 2)}")



# Get predictions for test set
y_prob = [max(val) for val in clf.predict_proba(X_test)]
y_pred = [np.argmax(val) for val in clf.predict_proba(X_test)]


# Evaluate the model
# Get test accuracy score
score = accuracy_score([np.argmax(val[:2]) for val in test_y], y_pred)
print('Accuracy:', score)

"""
# Evaluate the model
# Get test bet accuracy score
score = 0
count = 0
i = 0
for val in y_pred:
    if val != 2:
        if np.argmax(test_y[i,:3]) == val:
            score += 1
        count += 1
    i += 1
print('Bet accuracy:', score/count)
"""

# Get test confusion matrix
print('Confusion matrix: ')
print(confusion_matrix([np.argmax(val[:2]) for val in test_y], y_pred))


# Get profit
profit = 0
stake = 5
i = 0
count = 0
correct = 0

for ind, row in X_test.iterrows():
    if y_pred[i] == 0 and 1.2 < df.loc[ind]['AvgP1'] and 1.2 < df.loc[ind]['AvgP2'] and 0.8 < y_prob[i]:
        if y_test[ind] == 1:
            player1_odds = df.loc[ind]['AvgP1']
            profit += (player1_odds-1)*stake
            correct += 1
            print(f"WIN: {player1[ind]} to beat {player2[ind]}")
        else:
            profit -= stake
            print(f"LOSS: {player1[ind]} to beat {player2[ind]}")
        count += 1
        print(round(profit, 2))
    elif y_pred[i] == 1 and 1.2 < df.loc[ind]['AvgP1'] and 1.2 < df.loc[ind]['AvgP2'] and 0.8 < y_prob[i]:
        if y_test[ind] == 0:
            player2_odds = df.loc[ind]['AvgP2']
            profit += (player2_odds-1)*stake
            correct += 1
            print(f"WIN: {player2[ind]} to beat {player1[ind]}")
        else:
            profit -= stake
            print(f"LOSS: {player2[ind]} to beat {player1[ind]}")
        count += 1
        print(round(profit, 2))
    i += 1

print()
print("TEST SET:")
print(f"Accuracy: {round(correct/count, 4)}")
print(f"Profit: ${round(profit, 2)}")
print(f"Return: {round(100*profit/(stake*count), 2)}%")
print(f"No. bets: {count}")
print(f"Avg. profit per bet: ${round(profit/count, 2)}")




"""
# Kelly criterion
init_capital = 100
capital = 100
staked = 0
i = 0
count = 0
correct = 0

for ind, row in X_test.iterrows():
    if y_pred[i] == 0 and 1 < y_prob[i]*df.loc[ind]['AvgP1'] and 1.5 < df.loc[ind]['AvgP1'] and 1.5 < df.loc[ind]['AvgP2'] and 0.9 < y_prob[i]:
        perc = ((df.loc[ind]['AvgP1']-1)*y_prob[i]-(1-y_prob[i]))/(df.loc[ind]['AvgP1']-1)
        stake = 0.1*perc*capital#max(5, min(perc*capital, 10))
        staked += stake
        if y_test[ind] == 1:
            player1_odds = df.loc[ind]['AvgP1']
            capital += (player1_odds-1)*stake
            correct += 1
            print(f"WIN: {player1[ind]} to beat {player2[ind]}")
        else:
            capital -= stake
            print(f"LOSS: {player1[ind]} to beat {player2[ind]}")
        count += 1
        print(stake)
        print(round(capital, 2))
    elif y_pred[i] == 1 and 1 < y_prob[i]*df.loc[ind]['AvgP2'] and 1.5 < df.loc[ind]['AvgP1'] and 1.5 < df.loc[ind]['AvgP2'] and 0.9 < y_prob[i]:
        perc = ((df.loc[ind]['AvgP2']-1)*y_prob[i]-(1-y_prob[i]))/(df.loc[ind]['AvgP2']-1)
        stake = 0.1*perc*capital#max(5, min(perc*capital, 10))
        staked += stake
        if y_test[ind] == 0:
            player2_odds = df.loc[ind]['AvgP2']
            capital += (player2_odds-1)*stake
            correct += 1
            print(f"WIN: {player2[ind]} to beat {player1[ind]}")
        else:
            capital -= stake  
            print(f"LOSS: {player2[ind]} to beat {player1[ind]}")
        count += 1
        print(stake)
        print(round(capital, 2))
    i += 1

print()
print("TEST SET:")
print(f"Accuracy: {round(correct/count, 4)}")
print(f"Profit: ${round(capital-init_capital, 2)}")
print(f"Return: {round(100*(capital-init_capital)/(staked), 2)}%")
print(f"No. bets: {count}")
print(f"Avg. profit per bet: ${round((capital-init_capital)/count, 2)}")
"""


