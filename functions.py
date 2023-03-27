import numpy as np
from time import time

def train_regression(clf, X_train, y_train): # define the training model function
# Record training duration
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print ("training time {:4f} seconds".format(end - start))

def predict_labels(clf, features, target): # define the labels function
    # Record Forecast Time
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print ("prediction time in {:4f} seconds".format (end - start))
    return (target, y_pred)

def train_predict(clf, X_train, y_train, X_test, y_test): # define the predicting function
    # Indicate the classifier and the training set size
    print ("training {} model, sample size {}.".format(clf.__class__.__name__, len (X_train)))
    # Training model
    train_regression(clf, X_train, y_train)
# Assessment model on test set
    target, y_pred = predict_labels(clf, X_test, y_test)
    #print ("Accuracy on training set: {:4f}.".format (acc))
    return(target, y_pred)

def hist_sampling(uplim = 60000, lowE = 0, highE = 2, nbins = 20, mcE=0):
    Ebins = np.logspace(lowE,highE,nbins)
    y_all = mcE
    # bin sampling
    for i in range(len(Ebins)-1):
        mask = (y_all>Ebins[i]) & (y_all<Ebins[i+1])
        y_all_temp = y_all[mask]
        if len(y_all_temp)>uplim:
            print(len(y_all_temp))
            mini_mask = np.random.randint(len(y_all_temp), size=uplim)
            mask = (y_all>Ebins[i]) & (y_all<Ebins[i+1])
            s = mask.sum()
            t_mask = np.zeros(s, dtype=bool)
            t_mask[np.random.choice(s, uplim, replace=False)] = True
            mask[mask] = t_mask
            print("looped mask: ", len(mask), "n true: ", mask.sum())
            if i == 0:
                joint_mask = mask#[mask]
            else:
                joint_mask = joint_mask | mask#[mask]
            print("###### THIS IS JOINT MASK #######", "n true: ", joint_mask.sum())

        else:
            print(len(y_all_temp))
            mini_mask = np.random.randint(len(y_all_temp), size=uplim)
            mask = (y_all>Ebins[i]) & (y_all<Ebins[i+1])
            #print(len(mask),"n true: ", mask.sum())
            s = mask.sum()
            t_mask = np.zeros(s, dtype=bool)
            t_mask[np.random.choice(s, len(y_all_temp), replace=False)] = True
            mask[mask] = t_mask
            print("looped mask: ", len(mask), "n true: ", mask.sum())
            if i == 0:
                joint_mask = mask#[mask]
            else:
                joint_mask = joint_mask | mask#[mask]
            print("###### THIS IS JOINT MASK #######", "n true: ", joint_mask.sum()) 
    return (joint_mask)

def bootstrap_sampling(data, uplim = 400, lowE = 0, highE = 2, nbins = 20):
    Ebins = np.logspace(lowE,highE,nbins)
    ran_state = 0
    histo = np.histogram(data['mc_energy_dst'],bins = Ebins,weights = data['weights'])
    cdata = data.copy()
    for i in range(len(Ebins)-1):
        print("Doing for bin " + str(i+1)+ " out of "+ str(len(Ebins))+ " in the energy range ["+str(Ebins[i]) + ", "+str(Ebins[i+1]) + "]")
        mask = (data['mc_energy_dst']>Ebins[i]) & (data['mc_energy_dst']<Ebins[i+1])
        data_temp = data[mask]
        if i == 0:
            frac_use = (uplim)/np.mean(data_temp['weights'])/len(data_temp['weights'])
            resampled_df = data_temp.sample(frac = frac_use,random_state=ran_state, replace = True)
        else:
            frac_use = (uplim)/np.mean(data_temp['weights'])/len(data_temp['weights'])
            resampled_df = resampled_df.append(data_temp.sample(frac = frac_use,random_state=ran_state, replace = True), ignore_index=True)

    #cdata = cdata.append(resampled_df, ignore_index=True)
    return(resampled_df)

def bootstrap_sampling_no_w(data, uplim = 500000, lowE = 0, highE = 2, nbins = 20):
    Ebins = np.logspace(lowE,highE,nbins)
    ran_state = 0
    histo = np.histogram(data['mc_energy_dst'],bins = Ebins,weights = data['weights'])
    cdata = data.copy()
    for i in range(len(Ebins)-1):
        print("Doing for bin " + str(i+1) + " out of "+ str(len(Ebins)))
        mask = (data['mc_energy_dst']>Ebins[i]) & (data['mc_energy_dst']<Ebins[i+1])
        data_temp = data[mask]
        if i == 0:
            frac_use = (uplim)/len(data_temp['weights'])
            resampled_df = data_temp.sample(frac = frac_use,random_state=ran_state, replace = True)
        else:
            frac_use = (uplim)/len(data_temp['weights'])
            resampled_df = resampled_df.append(data_temp.sample(frac = frac_use,random_state=ran_state, replace = True), ignore_index=True)

    #cdata = cdata.append(resampled_df, ignore_index=True)
    return(resampled_df)
