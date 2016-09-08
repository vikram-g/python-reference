import MySQLdb
import numpy
import pandas as pd
import numpy as np
import csv
import os
from scipy import stats
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.cross_validation import train_test_split
import xgboost as xgb
############################################################################################ Data Loading #################################################################################################
def connect():
    """
        This is a function create a connection to MySQL
        
    """
    try: 
        db = MySQLdb.connect(user="username", passwd="password", host='host_ip_address')
        print "MySQL is connected"
        return (db)
    except MySQLdb.Error, e:
        print "Unable to connect MySQL Error %d: %s" % (e.args[0], e.args[1])
        raise

def getData(db_table, reqCols):
    db=connect()
    cur=db.cursor()
    query="""select %s from %s""" %(",".join(reqCols),db_table)
    #print query
    cur.execute(query)
    data= pd.DataFrame(list(cur.fetchall()))
    data.columns = reqCols
    data = data.replace('',np.nan)
    dta = data.where((pd.notnull(data)), None)
    return data

def loadNormally():
    master_table_A = 'MerkleResponse.Data_Source'
    master_table_B = 'MerkleResponse_B.Data_Source_Campaign_B'
    colTypes = pd.read_csv('colsType.csv')
    # Continuous variables
    reqCols = np.array(colTypes.Feature)
    data_a = getData(master_table_A, reqCols) # Loads the data from database
    data_b = getData(master_table_B, reqCols)
    data = data_a.append(data_b, ignore_index=True)
    return data

def findMissingCols(df):
    '''
    Returns list of columns that have missing values in them
    '''
    return (df.columns[df.isnull().any()])

def identifyDuplicates(df):
    '''
    This is a memory efficient way to identify columns that are duplicates - In case of duplicates only one column name is returned
    '''
    groups = df.columns.to_series().groupby(df.dtypes).groups
    dups = []
    for t, v in groups.items():
        dcols = df[v].to_dict(orient="list")
        vs = dcols.values()
        ks = dcols.keys()
        lvs = len(vs)
        for i in range(lvs):
            for j in range(i+1,lvs):
                if vs[i] == vs[j]: 
                    dups.append(ks[i])
                    break
    return dups
############################################################################################ Summary Stat #################################################################################################
def summarize(db_table, reqCols):
    '''
    This function uses heuristics to identify type of columns
    '''
    db=connect()
    cur=db.cursor()
    query="""select %s from %s""" %(",".join(reqCols),db_table)
    #print query
    cur.execute(query)
    data= pd.DataFrame(list(cur.fetchall()))
    data.columns = reqCols
    data = data.replace('',np.nan)
    colTypes = {'Nulls':[],'Integers':[],'Numeric':[],'String':[],'Binary':[],'Categorical':[]}
    for cols in reqCols:
        if sum(pd.isnull(data.ix[:,cols])) == data.shape[0]:
            colTypes['Nulls'].append(cols) 
        else:
            try:
                x = data.ix[:,cols].astype(float)
                if np.nansum(x)%1 > 0:
                    colTypes['Integers'].append(cols) 
                elif np.nansum(x)%1 == 0:
                    if len(pd.Series.unique(x)) <= 2:
                        colTypes['Binary'].append(cols) 
                    elif len(pd.Series.unique(x)) < 5:
                        colTypes['Categorical'].append(cols) 
                    elif len(pd.Series.unique(x)) > 70:
                        colTypes['Integers'].append(cols) 
                    else:
                        colTypes['Numeric'].append(cols) 
            except ValueError:
                colTypes['String'].append(cols) 
    #pd.DataFrame(colTypes['Numeric']).to_csv('C:\\ADF\\Model Building\\num_cols.csv')
    return colTypes


def summarize_ints(data, colTypes, outputFileName):
    df = data.ix[:,tuple(colTypes.Feature[colTypes.DataType == 'continuous'])].convert_objects(convert_numeric=True)
    summaryStat = pd.DataFrame(df.count(axis = 0))
    summaryStat.columns = ['Count']
    summaryStat['Sum'] = df.sum(axis = 0, skipna = True)
    summaryStat['Max'] = df.max(axis = 0, skipna = True)
    summaryStat['Min'] = df.min(axis = 0, skipna = True)
    summaryStat['Mean'] = df.mean(axis = 0, skipna = True)
    summaryStat['Median'] = df.median(axis = 0, skipna = True)
    summaryStat['SD'] = df.std(axis = 0, skipna = True)
    summaryStat['Quantile_25'] = df.quantile(q = 0.25,axis = 0)
    summaryStat['Quantile_50'] = df.quantile(q = 0.5,axis = 0)
    summaryStat['Quantile_75'] = df.quantile(q = 0.75,axis = 0)
    summaryStat['Skew'] = df.skew(axis = 0, skipna = True)
    summaryStat['kurtosis '] = df.kurt(axis = 0, skipna = True)
    summaryStat.to_csv(os.path.join(outputFileName + '.' + 'csv'))

def summarize_categories(data, colTypes, outputFileName):
    df = data.ix[:,tuple(colTypes.Feature[colTypes.DataType != 'continuous'])]
    summaryStat = df.astype('string').describe(include = 'all')
    summaryStat.to_csv(os.path.join(outputFileName + '.' + 'csv'))


############################################################################################ Data Sampling #################################################################################################
def sampler(data, dv, selRatio = 0.8, seedVal = 10):
    np.random.seed(seedVal)
    devRows = np.sort(np.random.choice(data.shape[0],  int(data.shape[0] * selRatio), replace = False))
    validRows = np.sort(list(set(range(data.shape[0])) - set(devRows)))
    x_dev = pd.DataFrame(data.ix[devRows,:])
    x_val = pd.DataFrame(data.ix[validRows,:])
    y_dev = pd.DataFrame(data.ix[devRows,dv])
    y_val = pd.DataFrame(data.ix[validRows,dv])
    return (x_dev, y_dev, x_val, y_val)

def sampler_cv(data, dv, fold, seedVal = 10):
    np.random.seed(seedVal)
    choices = np.random.choice(5,data.shape[0],replace = True)
    devRows = [False if x == fold else True for x in choices]
    validRows = [True if x == fold else False for x in choices]
    x_dev = pd.DataFrame(data.ix[devRows,:])
    x_val = pd.DataFrame(data.ix[validRows,:])
    y_dev = pd.DataFrame(data.ix[devRows,dv])
    y_val = pd.DataFrame(data.ix[validRows,dv])
    return (x_dev, y_dev, x_val, y_val)


def Ultimatesampler(data, sampling_ratio=(0.4,0.2,0.2,0.2), seedVal=100):
    np.random.seed(seedVal)
    dev_profile_val1,val2=train_test_split(data,test_size=sampling_ratio[3])
    dev_profile,val1=train_test_split(dev_profile_val1,test_size=sampling_ratio[2]/(1-sampling_ratio[3]))
    dev,profile=train_test_split(dev_profile,test_size=sampling_ratio[1]/(sampling_ratio[0]+sampling_ratio[1]))
    return (dev,profile,val1,val2)

############################################################################################ Data Preparation #################################################################################################
# One Hot encoding
def oneHotEncode(df, feats):
    '''
    Takes a data set and columns needed to encode
    Removes the original columns, encodes those one and appends and returns the data set
    '''
    dummies = pd.get_dummies(df[feats], dummy_na = True, columns = feats) 
    df = df.drop(feats, axis = 1, inplace = False)
    return (df.join(dummies))

# Match the columns
def matchColumns(x_dev, x_val):
    '''
    Done after one hot encoding
    If there are few columns missing (due to that level missing in one of the dataset) that column is removed
    The rationale behind removing them is such columns are never going to add predictive power and also some future data might not have them
    '''
    commonCols = [x for x in x_dev.columns if x in x_val.columns]
    x_dev = x_dev.ix[:,commonCols]
    x_val = x_val.ix[:,commonCols]
    return (x_dev, x_val)


# Baysian Risk table
def baysianRiskTable(val,train, feats, dv = 'response_application'):
    '''
    Takes training as well as dev sample and the columns to be replaced with risk tables 
    Returns the data set with required columns replaced with mean response rate of the that level of training data
    '''
    train[dv] = train[dv].astype('int')
    val[dv] = val[dv].astype('int')
    gamma = train[dv].sum()/train[dv].size
    pos, neg = sum(train[dv]), len(train[dv]) - sum(train[dv])
    numer = pos * gamma
    denom = (pos + neg) * gamma
    train[feats] = train[feats].fillna(-9999)
    val[feats] = val[feats].fillna(-9999)
    for cols in feats:
        lookUp_numer = train.groupby(cols)[dv].agg(np.sum)
        lookUp_denom = train.groupby(cols)[dv].agg(np.size)
        #posNumerDev = train[cols].map(lookUp_numer).fillna(0)
        #posDinomDev = train[cols].map(lookUp_denom).fillna(0)
        #train[cols] = (posNumerDev + numer)/(denom + posDinomDev)
        posNumerVal = val[cols].map(lookUp_numer).fillna(0)
        posDinomVal = val[cols].map(lookUp_denom).fillna(0)
        val[cols] = (posNumerVal + numer)/(denom + posDinomVal)
    return (val)

# Risk table combining multiple columns together
def combinedBaysianRiskTable(train, val, feats, returnColumn, dropOriginal = True, dv = 'response_application'):
    '''
    Calcultes Baysian dv score combining all the columns together
    Returns the data set with required columns replaced/added (depending on parameter dropOriginal) with mean response rate of the that level of training data
    '''
    train[dv] = train[dv].astype('int')
    val[dv] = val[dv].astype('int')
    gamma = train[dv].sum()/train[dv].size
    pos, neg = sum(train[dv]), len(train[dv]) - sum(train[dv])
    numer = pos * gamma
    denom = (pos + neg) * gamma
    train[returnColumn] = train.ix[:,feats].astype('str').apply(lambda x: ''.join(x), axis=1)
    val[returnColumn] = val.ix[:,feats].astype('str').apply(lambda x: ''.join(x), axis=1)
    lookUp_numer = train.groupby(returnColumn)[dv].agg(np.sum)
    lookUp_denom = train.groupby(returnColumn)[dv].agg(np.size)
    posNumerDev = train[returnColumn].map(lookUp_numer).fillna(0)
    posDinomDev = train[returnColumn].map(lookUp_denom).fillna(0)
    train[returnColumn] = (posNumerDev + numer)/(denom + posDinomDev)
    posNumerVal = val[returnColumn].map(lookUp_numer).fillna(0)
    posDinomVal = val[returnColumn].map(lookUp_denom).fillna(0)
    val[returnColumn] = (posNumerVal + numer)/(denom + posDinomVal)
    if dropOriginal:
        train = train.drop(feats, inplace = False, axis = 1)
        val = val.drop(feats, inplace = False, axis = 1)
    return (train, val)


# MDL discretization
def descritizer(train, val, dv, n_splits):
    '''
    This takes one dev and val at a time along with dv (dev sample)  
    n_splits is number of bins that should be present in the target data set
    Returns the discritized dev and val columns
    '''
    nonNA_Index = [x for x in train.index if train[x] == train[x]] # Removing the missing values
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_features = None, max_leaf_nodes = n_splits - 1, min_samples_split = len(nonNA_Index)/5, class_weight = 'auto', random_state  = 1)
    X = pd.DataFrame(train[nonNA_Index].astype('float64')) # Picking only non missing values
    clf.fit(X, dv.ix[nonNA_Index,:])
    cuts = np.sort([x for x in clf.tree_.threshold if x != -2])
    cuts = np.insert(cuts,0,min(X.ix[:,0]) - 0.1) # Adding the lower end of the bins
    #train = np.digitize(train.fillna(-9999).astype('float64'), cuts, right = True)
    minVal = val.fillna(99999).astype('float64').min()
    maxVal = val.fillna(-9999).astype('float64').max()
    cuts[0] = cuts[0] if minVal - 0.1 > cuts[0] else minVal - 0.1
    cuts[-1] = cuts[-1] if maxVal < cuts[-1] else maxVal
    val = np.digitize(val.fillna(-9999).astype('float64'), cuts, right = True)
    return (val)

def descritizer_df(val, train, dv, n_splits, feats):
    '''
    This call the function "descritizer" to descritize all the provided columns
    '''
    dv = pd.DataFrame(dv)
    for cols in feats:
        val[cols] = descritizer(train[cols], val[cols], dv, n_splits)
    return (val)

# Features with minimum number of levels
def featsWithMinLevels(dev, feats, minLevels):
    '''
    Returns a list of column names which has >= minLevels number of levels in the category
    '''
    noLevels = dev.ix[:,feats].astype('str').describe().ix[1,:]
    returnList = [str(x) for x in noLevels.index[noLevels >= minLevels]]
    return (np.array(returnList))

# Create indicators for special values
def createSpecialValueFlags(dev, val1,val2, val3, feats):
    '''
    Identify special values in the given data frame and create an indicator for them
    A variable is identified special by following way:  
        - Should occur more than 10% of time
        - Should occur more than 5 times than would be normally expected if all levels in a column is expected to occur equally likely
                i.e if a column has 4 levels then each of them would normally be expected to be present in 25% of the data. Special value should occur 5 times that
    Along with dev and val dataframe this also returns the names of added columns
    '''
    addedCols = []
    for cols in feats:
        aggTable = dev.groupby(cols)[cols].agg(np.size)/dev[cols].count()
        importantIndex = aggTable.index[(aggTable > 0.1) & (aggTable > 5*(1/int(aggTable.shape[0])))]
        if len(importantIndex) > 0:
            for vals in importantIndex:
                addedCols = addedCols + [cols + '_imp_' + str(vals)]
                val1[cols + '_imp_' + str(vals)] = [1 if x == vals else 0 for x in val1[cols]]
                val2[cols + '_imp_' + str(vals)] = [1 if x == vals else 0 for x in val2[cols]]
                val3[cols + '_imp_' + str(vals)] = [1 if x == vals else 0 for x in val3[cols]]
    return (val1 ,val2, val3, addedCols)

# Create flags for missing values
def createMissingValueFlags(data,feats):
    '''
    For all columns sent, identifies the one with missing values and creates a new column for those.
    Original column is left as such
    '''
    for cols in feats:
        if data[cols].isnull().values.any(): # Checking if there is at least one missing value
            data[cols + '#Missing'] = [1 if x != x else 0 for x in data[cols]]
    return (data)

# Scale and normalize
def normalizer(train, val1, val2, val3, feats, replace_missing = False):
    '''
    Takes a dev and val sample along with list of columns to be normalized (feats)
    Normalizes the data based on dev sample mean and sd
    If replace_missing is set to True - missing values are replaced by 0 mean before imputation
    Returns the data set with columns in feats transformed
    '''
    if replace_missing:
        meanImputer = Imputer(missing_values='NaN', strategy='median', axis=0)
        meanImputer.fit(train[feats])
        val1[feats] = pd.DataFrame(meanImputer.transform(val1[feats]), columns = feats, index = val1.index)
        val2[feats] = pd.DataFrame(meanImputer.transform(val2[feats]), columns = feats, index = val2.index)
        val3[feats] = pd.DataFrame(meanImputer.transform(val3[feats]), columns = feats, index = val3.index)
    train[feats] = train[feats].astype('float64')
    mean_struct = train[feats].mean(skipna = True)
    std_struct = train[feats].std(skipna = True)
    for cols in range(len(feats)):
        val1[feats[cols]] = (val1[feats[cols]].astype('float64') - mean_struct[cols])/std_struct[cols]
        val2[feats[cols]] = (val2[feats[cols]].astype('float64') - mean_struct[cols])/std_struct[cols]
        val3[feats[cols]] = (val3[feats[cols]].astype('float64') - mean_struct[cols])/std_struct[cols]
    return (val1, val2, val3)

def scale0_10(x):
    '''
    Takes a column and returns all the values scaled at 0 - 1 range
    '''
    minVal = x.fillna(99999).astype('int').min()
    maxVal = x.fillna(-9999).astype('int').max()
    x = (x - minVal)/(maxVal - minVal)
    return (x)

def respRate(feat, dv):
    '''
    This takes a column and the dv value.
    Then bins the column and finds avg response rate in dv
    Also finds correlation between this mean response and range(10)
    Returns the correlation result and aggregated mean response rate
    '''
    feat = feat.astype('float64')
    dv = dv.astype('float64')
    dv = dv[~numpy.isnan(feat)]
    feat = feat[~numpy.isnan(np.array(feat))]
    feat = pd.DataFrame(feat)
    
    bins = [float(i[1:i.index(',')]) for i in pd.cut(feat,10,right = False).categories]
    binnedVar = np.digitize((feat.ix[:,0]),bins)
    op = pd.DataFrame([binnedVar,dv]).transpose()
    aggVal = op.groupby([0])[1].mean()
    corResult = abs(stats.spearmanr(aggVal, range(0,len(aggVal)))[0])
    #chiSq = stats.chisquare(aggVal)[0]
    return (corResult, aggVal, bins)

def pca(dev, val, feats,clust_name, should_scale = False, n_component = 'mle'):
    if should_scale:
        dev, val = normalizer(dev, val, feats)
    pca = PCA(n_components = n_component, copy = True, whiten = True)
    pca.fit(dev.ix[:,feats].fillna(0))
    dev_comps = pd.DataFrame(pca.transform(dev.ix[:,feats].fillna(0)), columns = [clust_name + '@' + str(x) for x in range(pca.n_components_)], index = dev.index)
    val_comps = pd.DataFrame(pca.transform(val.ix[:,feats].fillna(0)), columns = [clust_name + '@' + str(x) for x in range(pca.n_components_)], index = val.index)
    dev = dev.drop(feats, inplace = False, axis = 1)
    val = val.drop(feats, inplace = False, axis = 1)
    dev = dev.append(dev_comps)
    val = val.append(val_comps)
    return (dev, val)
############################################################################################ Model Building #################################################################################################
def ceate_feature_map(features):
    '''
    This stores the feature map .fmap file to be used to obtain feature importance in XGBoost
    '''
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def dataPrepScoring(dev, val, y_dev, y_val, fileName):
    '''
    Takes in all the files needed for scoring the model and file name (WITHOUT extension)
    First it drops the duplicate columns
    Does grid search over pre-defined set of parameters and indentifies the best model (highest Val KS where the Dev KS is not more than 20% of Val KS)
    Obtains the coefficients for this particular parameter and writes them to the file
    In the written file - one of the column name would have the best parameter value
    '''
    # Dropping duplicate columns before building the model
    dupCols = identifyDuplicates(x_dev)
    x_dev.drop(dupCols, inplace = True, axis = 1)
    x_val.drop(dupCols, inplace = True, axis = 1)
    if (y_dev.columns[0] in dev.columns): # Remove the dv is it is in the data set
        dev = dev.drop(y_dev.columns[0], inplace = False, axis = 1)
        val = val.drop(y_val.columns[0], inplace = False, axis = 1)
    fileName = r'Recipe_op/' + fileName
    print ('Starting L1 Model')
    maxKs = 0
    all_ks_dev = {}
    all_ks_val = {}
    c_values = [0.001,0.0025,0.0035,0.0045,0.005,0.0055,0.0065,0.0075,0.01,0.02]
    for each_c in c_values:
        print "Model Building Started"
        LR_model = LogisticRegression(C= each_c, penalty = 'l1', tol = 0.01, class_weight = 'auto')
        LR_model.fit(dev, y_dev)
        print "Model Building Finished"
        all_ks_dev[each_c]=[calculate_max_ks(LR_model, dev, y_dev)]
        all_ks_val[each_c]=[calculate_max_ks(LR_model, val, y_val)]
        print all_ks_val
    ks_vals = pd.DataFrame([c_values, [all_ks_dev[x][0] for x in c_values], [all_ks_val[x][0] for x in c_values]]).transpose()
    ks_vals.columns = ['C_Value', 'Dev_KS','Val_KS']
    #best_ks = max(ks_vals.Val_KS[ks_vals.Dev_KS/ks_vals.Val_KS < 1.2])
    best_ks = max(ks_vals.Val_KS)
    best_c = min(ks_vals.C_Value[ks_vals.Val_KS == best_ks])
    LR_model = LogisticRegression(C= best_c, penalty = 'l1', tol = 0.01, class_weight = 'auto')
    #ks_5fold = getPredictions(dev, val, y_dev, y_val, location = fileName, classifier = LR_model, model = 'l1')
    LR_model.fit(dev, y_dev)
    modelCoefs = pd.DataFrame([dev.columns.values, LR_model.coef_[0].transpose()]).transpose()
    modelCoefs['absCoeffs'] = abs(modelCoefs[1])
    modelCoefs.columns = ['Features' + str(best_c), 'coeffs', 'absCoeffs']
    modelCoefs.to_csv(fileName + '_L1_Coeffs.csv')
    ks_vals.to_csv(fileName + '_L1_KS_Scores.csv')
    print ('L1 Model over. Best C value :%f and Best KS Score :%f' %(best_c,best_ks))
    print ('Starting Random Forest Model')
    iter = 0
    maxKs = 0
    ks = pd.DataFrame([0.0,0.0,0.0,0.0]).transpose()
    ks.columns = ['depth','ntree','Dev_KS','Val_KS']
    colsSelect = dev.columns.values
    for depth in [2,3,4,5]:
      for est in [500]:
        for crit in ['entropy']:
          all_ks_dev = {}
          all_ks_val = {}
          clf = RandomForestClassifier(criterion= crit, n_estimators=est, class_weight  = 'auto', max_depth = depth, max_features = 'sqrt', oob_score  = True, 
                                      n_jobs  = -1, random_state = 1, verbose  = 1, bootstrap = True, min_samples_leaf = 10000)
          clf.fit(dev[colsSelect], y_dev)
          ks.ix[iter,:]=[depth, est, calculate_max_ks_rf(clf, dev[colsSelect], y_dev), calculate_max_ks_rf(clf, val[colsSelect], y_val)]
          print ks.ix[iter,:]
          iter = iter + 1
    #best_ks = max(ks.Val_KS[ks.Dev_KS/ks.Val_KS < 1.2])
    best_ks = max(ks.Val_KS)
    best_depth = min(ks.depth[ks.Val_KS == best_ks])
    best_ntree = min(ks.ntree[ks.Val_KS == best_ks])
    clf = RandomForestClassifier(criterion= 'entropy', n_estimators= int(best_ntree), class_weight  = 'auto', max_depth = int(best_depth), max_features = 'sqrt', oob_score  = True, 
                                n_jobs  = -1, random_state = 1, verbose  = 1, bootstrap = True, min_samples_leaf = 10000)
    #ks_5fold = getPredictions(dev, val, y_dev, y_val, location = fileName, classifier = clf, model = 'rf')
    clf.fit(dev, y_dev)
    featImp = pd.DataFrame([dev.columns.values,clf.feature_importances_ ]).transpose()
    featImp.columns = ['Feat' + str(best_depth) + str(best_ntree), 'Imp']
    featImp.to_csv(fileName + '_RF_Coeffs.csv')
    ks.to_csv(fileName + '_RF_KS_Scores.csv')
    print ('Random Forest Model over. Best depth = %f, Best nTree = %f and  Best KS Score = %f' %(best_depth,best_ntree, best_ks))

#Score L1 model
def ScoreL1(dev, val, y_dev, y_val,l1_gc_params,  fileName):
    '''
    Takes in all the files needed for scoring the model and file name (WITHOUT extension)
    First it drops the duplicate columns
    Does grid search over pre-defined set of parameters and indentifies the best model (highest Val KS where the Dev KS is not more than 20% of Val KS)
    Obtains the coefficients for this particular parameter and writes them to the file
    In the written file - one of the column name would have the best parameter value
    '''
    fileName = r'ModelScoring/' + fileName
    print ('Starting L1 Model')
    maxKs = 0
    all_ks_dev = {}
    all_ks_val = {}
    c_values = l1_gc_params
    for each_c in c_values:
        print "Model Building Started"
        LR_model = LogisticRegression(C= each_c, penalty = 'l1', tol = 0.01, class_weight = 'auto')
        LR_model.fit(dev, y_dev)
        print "Model Building Finished"
        all_ks_dev[each_c]=[calculate_max_ks(LR_model, dev, y_dev)]
        all_ks_val[each_c]=[calculate_max_ks(LR_model, val, y_val)]
        if each_c == c_values[0]:
            prediction = pd.DataFrame(LR_model.predict_proba(val)[:,1])
            prediction.columns = ['0']
        else:
            prediction[str(each_c)] = LR_model.predict_proba(val)[:,1]
        print all_ks_val
    ks_vals = pd.DataFrame([c_values, [all_ks_dev[x][0] for x in c_values], [all_ks_val[x][0] for x in c_values]]).transpose()
    ks_vals.columns = ['C_Value', 'Dev_KS','Val_KS']
    best_ks = max(ks_vals.Val_KS[ks_vals.Dev_KS/ks_vals.Val_KS < 1.2])
    #best_ks = max(ks_vals.Val_KS)
    best_c = min(ks_vals.C_Value[ks_vals.Val_KS == best_ks])
    LR_model = LogisticRegression(C= best_c, penalty = 'l1', tol = 0.01, class_weight = 'auto')
    #ks_5fold = getPredictions(dev, val, y_dev, y_val, location = fileName, classifier = LR_model, model = 'l1')
    LR_model.fit(dev, y_dev)
    modelCoefs = pd.DataFrame([dev.columns.values, LR_model.coef_[0].transpose()]).transpose()
    modelCoefs['absCoeffs'] = abs(modelCoefs[1])
    modelCoefs.columns = ['Features' + str(best_c), 'coeffs', 'absCoeffs']
    modelCoefs.to_csv(fileName + '_L1_Coeffs.csv')
    ks_vals.to_csv(fileName + '_L1_KS_Scores.csv')
    prediction.to_csv(fileName + '_L1_val_preds.csv')
    print ('L1 Model over. Best C value :%f and Best KS Score :%f' %(best_c,best_ks))

def ScoreRF(dev, val, y_dev, y_val,rf_gc_prams, fileName):
    '''
    Takes in all the files needed for scoring the model and file name (WITHOUT extension)
    First it drops the duplicate columns
    Does grid search over pre-defined set of parameters and indentifies the best model (highest Val KS where the Dev KS is not more than 20% of Val KS)
    Obtains the coefficients for this particular parameter and writes them to the file
    In the written file - one of the column name would have the best parameter value
    '''
    fileName = r'ModelScoring/' + fileName
    print ('Starting Random Forest Model')
    iter = 0
    maxKs = 0
    ks = pd.DataFrame([0.0,0.0,0.0,0.0,0.0,0.0,0.0]).transpose()
    ks.columns = ['depth','ntree','Class_weight','Features','min_sample_leaf','Dev_KS','Val_KS']
    colsSelect = dev.columns.values
    for depth in rf_gc_prams['depth']:
        for est in rf_gc_prams['est']:
            for c_class_weight in rf_gc_prams['c_class_weight']:
                for c_max_features in rf_gc_prams['c_max_features']:
                    for crit in rf_gc_prams['crit']:
                        for c_min_samples_leaf in rf_gc_prams['c_min_samples_leaf']:
                            all_ks_dev = {}
                            all_ks_val = {}
                            clf = RandomForestClassifier(criterion= crit, n_estimators=est, class_weight  = c_class_weight, max_depth = depth, max_features = c_max_features, oob_score  = False, 
                                                      n_jobs  = -1, random_state = 1, verbose  = 1, bootstrap = True, min_samples_leaf = c_min_samples_leaf)
                            clf.fit(dev[colsSelect], y_dev)
                            if iter == 0:
                                prediction = pd.DataFrame(clf.predict_proba(val)[:,1])
                                prediction.columns = ['0']
                            if iter != 0:
                                prediction[str(iter)] = clf.predict_proba(val)[:,1]
                            ks.ix[iter,:]=[depth, est,c_class_weight, c_max_features, c_min_samples_leaf, calculate_max_ks_rf(clf, dev[colsSelect], y_dev), calculate_max_ks_rf(clf, val[colsSelect], y_val)]
                            print ks.ix[iter,:]
                            iter = iter + 1
                            if iter % 25 == 0:
                                ks.to_csv(fileName + '_RF_KS_Scores.csv')
                                prediction.to_csv(fileName + '_RF_val_preds.csv')
    ks.to_csv(fileName + '_RF_KS_Scores.csv')
    prediction.to_csv(fileName + '_RF_val_preds.csv')

def ScoreXGB(xgtrain, xgtest, y_dev, y_val, xgb_gc_prams, fileName):
    '''
    Takes in all the files needed for scoring the model and file name (WITHOUT file extension)
    First it drops the duplicate columns
    Does grid search over pre-defined set of parameters and indentifies the best model (highest Val gs where the Dev gs is not more than 20% of Val gs)
    Obtains the coefficients for this particular parameter and writes them to the file
    In the written file - one of the column name would have the best parameter value
    '''
    fileName = r'ModelScoring/' + fileName
    eta_list = xgb_gc_prams['eta'] 
    max_depth_list = xgb_gc_prams['max_depth']
    rounds_list = xgb_gc_prams['rounds']
    col_sample = xgb_gc_prams['colsample_bytree']
    row_sample = xgb_gc_prams['subsample']
    p_gamma = xgb_gc_prams['gamma']
    c_weight = xgb_gc_prams['min_child_weight']
    c_aplha = xgb_gc_prams['alpha']
    print ('Starting XGBoost Model')
    gs = pd.DataFrame([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).transpose()
    gs.columns = ['Eta','MaxDepth','Rounds','Col_sample','Row_sample','Gamma','Weight','Alpha','Dev-ks','Val-ks']
    iter = 0
    for eta in eta_list:
        for max_depth in max_depth_list:
            for rounds in rounds_list:
                for colsample_bytree in col_sample:
                    for subsample in row_sample:
                        for gamma in p_gamma:
                            for min_child_weight in c_weight:
                                for alpha in c_aplha:
                                    ROUNDS = rounds
                                    gboost_params = { 
                                    "objective": xgb_gc_prams['objective'],
                                    "scale_pos_weight" : xgb_gc_prams['scale_pos_weight'],
                                    "booster": "gbtree",
                                    "eval_metric": "auc",
                                    "eta": eta, #0.01, # 0.06, #0.01,
                                    #"min_child_weight": 240,
                                    'gamma' : gamma,
                                    "subsample" : subsample,
                                    "alpha" : alpha,
                                    "colsample_bytree" : colsample_bytree,
                                    "max_depth" : max_depth,
                                    "min_child_weight" : min_child_weight,
                                    "silent" : 1
                                    }
                                    gs.ix[iter,:8] = [eta, max_depth, rounds,colsample_bytree,subsample,gamma,min_child_weight,alpha] 
                                    print "Iteration started #%d" %(iter)
                                    clf = xgb.train(gboost_params, xgtrain, ROUNDS)
                                    if iter == 0:
                                        prediction = pd.DataFrame(clf.predict(xgtest))
                                        prediction.columns = ['0']
                                    if iter != 0:
                                        prediction[str(iter)] = clf.predict(xgtest)
                                    gs.ix[iter,8:] = [calculate_max_ks_xgb(clf,xgtrain,y_dev), calculate_max_ks_xgb(clf,xgtest,y_val)] 
                                    print gs.ix[iter,7:]
                                    if iter % 25 == 0:
                                        gs.to_csv(fileName + '_XGB_KS_Scores.csv')
                                        prediction.to_csv(fileName + '_XGB_val_preds.csv')
                                    iter = iter + 1
    print ('XGBoost Model over')
    gs.to_csv(fileName + '_XGB_KS_Scores.csv')
    prediction.to_csv(fileName + '_XGB_val_preds.csv')


def getPredictions(x_dev, x_val, y_dev, y_val, classifier,location, model = 'l1'):
    '''
    Function would take a classifier and predict the result on a 5 fold cv
    Currently supports L1 and Random Forest models. 
    '''
    if model in ['l1', 'rf']:
        x = pd.DataFrame(x_dev.append(x_val))
        y = pd.DataFrame(y_dev.append(y_val))
        preds = pd.DataFrame(np.zeros(x.shape[0]), index = x.index, columns = ['Prediction'])
        np.random.seed(8)
        rowPick = np.random.choice(range(5),x.shape[0], replace = True)
        for i in range(5):
            valIndex = [x.index % 5 == i][:10]
            devIndex = [x.index % 5 != i][:10]
            classifier.fit(x.ix[devIndex[0]], y.ix[devIndex[0]])
            preds.ix[valIndex[0]] = classifier.predict_proba(x.ix[valIndex[0]])[:,1][:,None]
        ks = calculate_ks(y, preds)
        print ("The 5 fold KS score is %f" %(ks))
        preds.columns = str(ks)
        if model == 'l1':
            preds.to_csv(location+'_predictions_l1.csv')
        elif model == 'rf':
            preds.to_csv(location+'_predictions_rf.csv')
        return (ks)
############################################################################################ KS Calculation #################################################################################################
def calculate_max_ks(LR_model, Y_IDV, Y_DV):
    pridict_prob = LR_model.predict_proba(Y_IDV)    
    merged = numpy.dstack((pridict_prob[:,0], Y_DV.ix[:,0].astype('int')))
    merged = merged[0]
    mergesort = merged[numpy.argsort(merged[:,0])]   
    total_bad = numpy.unique(mergesort[:,1], return_counts=True)[1][0]
    total_good =  numpy.unique(mergesort[:,1], return_counts=True)[1][1]
    count0 =0
    count1 = 0
    max_ks =0
    for i,j in mergesort:
      if j == 1:
        count1 += 1
      elif j == 0:
        count0 += 1
      ks = (count1 / float(total_good) -  count0/float(total_bad)) * 100
      if ks > max_ks:
          max_ks = ks
    return max_ks

def calculate_max_ks_xgb(clf, xgtest, y_val):
    pridict_prob = clf.predict(xgtest)
    merged = numpy.dstack((pridict_prob, y_val.ix[:,0]))
    merged = merged[0]
    mergesort = merged[numpy.argsort(1 - merged[:,0])]   
    total_bad = numpy.unique(mergesort[:,1], return_counts=True)[1][0]
    total_good =  numpy.unique(mergesort[:,1], return_counts=True)[1][1]
    count0 =0
    count1 = 0
    max_ks =0
    for i,j in mergesort:
      if j == 1:
        count1 += 1
      elif j == 0:
        count0 += 1
      ks = (count1 / float(total_good) -  count0/float(total_bad)) * 100
      if ks > max_ks:
          max_ks = ks
    return max_ks

def calculate_max_ks_rf(LR_model, Y_IDV, Y_DV):
    pridict_prob = LR_model.predict_proba(Y_IDV)
    merged = numpy.dstack((pridict_prob[:,0], Y_DV.values.flatten().astype('float64')))
    merged = merged[0]
    mergesort = merged[numpy.argsort(merged[:,0])]   
    total_bad = numpy.unique(mergesort[:,1], return_counts=True)[1][0]
    total_good =  numpy.unique(mergesort[:,1], return_counts=True)[1][1]
    count0 =0
    count1 = 0
    max_ks =0
    for i,j in mergesort:
      if j == 1:
        count1 += 1
      elif j == 0:
        count0 += 1
      ks = (count1 / float(total_good) -  count0/float(total_bad)) * 100
      if ks > max_ks:
          max_ks = ks
    return max_ks

def calculate_ks(y, preds):
    merged = preds.join(y).sort(columns = 'Prediction', ascending  = False, inplace = False, axis = 0)
    goods = sum(merged['response_application'])
    bads = len(merged['response_application']) - goods
    max_ks = 0
    count0 = 0
    count1 = 0
    for i in merged.iterrows():
        if i[1][1] == 1:
            count1 += 1
        else:
            count0 += 1
        ks = (count1 / float(goods) -  count0/float(bads)) * 100
        if max_ks < ks:
            max_ks = ks
    return max_ks

def gains_xgb(clf, xgtest, y_val):
    pridict_prob = clf.predict(xgtest)
    merged = numpy.dstack((pridict_prob, y_val.ix[:,0]))
    merged = merged[0]
    mergesort = pd.DataFrame(merged[numpy.argsort(1 - merged[:,0])])
    mergesort['decile'] = pd.qcut(mergesort.ix[:,0],10, labels = range(10))
    mergesort.columns = ['prob','dv','decile']
    gains_table = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.min))
    gains_table.columns = ['min']
    gains_table['max'] = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.max))
    gains_table['total'] = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.size))
    gains_table['goods'] = pd.DataFrame(mergesort.groupby('decile')['dv'].agg(np.sum))
    gains_table['decile'] = pd.DataFrame(mergesort.groupby('decile')['decile'].agg(np.min))
    decile10_ks = ((gains_table.goods[gains_table.decile == 9]/sum(gains_table.goods)) - (gains_table.total - gains_table.goods)[gains_table.decile == 9] / sum(gains_table.total)) * 100
    return (gains_table, float(decile10_ks))

def gains_rf(LR_model, Y_IDV, Y_DV):
    pridict_prob = LR_model.predict_proba(Y_IDV)
    merged = numpy.dstack((pridict_prob[:,0], Y_DV.values.flatten().astype('float64')))
    merged = merged[0]
    merged[:,0] = 1 - merged[:,0]
    mergesort = pd.DataFrame(merged[numpy.argsort(1 - merged[:,0])])
    mergesort['decile'] = pd.qcut(mergesort.ix[:,0],10, labels = range(10))
    mergesort.columns = ['prob','dv','decile']
    gains_table = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.min))
    gains_table.columns = ['min']
    gains_table['max'] = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.max))
    gains_table['total'] = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.size))
    gains_table['goods'] = pd.DataFrame(mergesort.groupby('decile')['dv'].agg(np.sum))
    gains_table['decile'] = pd.DataFrame(mergesort.groupby('decile')['decile'].agg(np.min))
    decile10_ks = ((gains_table.goods[gains_table.decile == 9]/sum(gains_table.goods)) - (gains_table.total - gains_table.goods)[gains_table.decile == 9] / sum(gains_table.total)) * 100
    return (gains_table, float(decile10_ks))

def gains_l1(LR_model, Y_IDV, Y_DV):
    pridict_prob = LR_model.predict_proba(Y_IDV)    
    merged = numpy.dstack((pridict_prob[:,0], Y_DV.ix[:,0].astype('int')))
    merged = merged[0]
    merged[:,0] = 1 - merged[:,0]
    mergesort = pd.DataFrame(merged[numpy.argsort(1 - merged[:,0])])
    mergesort['decile'] = pd.qcut(mergesort.ix[:,0],10, labels = range(10))
    mergesort.columns = ['prob','dv','decile']
    gains_table = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.min))
    gains_table.columns = ['min']
    gains_table['max'] = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.max))
    gains_table['total'] = pd.DataFrame(mergesort.groupby('decile')['prob'].agg(np.size))
    gains_table['goods'] = pd.DataFrame(mergesort.groupby('decile')['dv'].agg(np.sum))
    gains_table['decile'] = pd.DataFrame(mergesort.groupby('decile')['decile'].agg(np.min))
    decile10_ks = ((gains_table.goods[gains_table.decile == 9]/sum(gains_table.goods)) - (gains_table.total - gains_table.goods)[gains_table.decile == 9] / sum(gains_table.total)) * 100
    return (gains_table, float(decile10_ks))
############################################################################################ WIP #################################################################################################


