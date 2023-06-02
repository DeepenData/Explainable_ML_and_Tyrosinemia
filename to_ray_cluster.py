from utils import *
import warnings
warnings.filterwarnings("ignore")
import time

if __name__ == '__main__':
    start_time = time.time()  # Record the start time

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    OPBG = [ 'age at diagnosis (months)',            
                'ntbc dosis mg/kg/day',
                'ntbc levels (dbs)',
                'sca (urine)',
                'methionine dbs',
                'tyrosine dbs',
                'phenylalanine dbs',
                'methionine (plasma)',
                'tyrosine (plasma)',
                'phenylalanine (plasma)',
                'inr',
                'pt (sec)',
                'bili total',
                'gpt',
                'got',
                'ggt',
                'alkaline phosphatase',
                'alfa-fetoprotein']
        
    df1 = pd.read_csv('data/tirosinemia.csv').loc[:,OPBG]
    df2 = pd.read_csv('data/tirosinemia_italia.csv').loc[:,OPBG]
    compare_dataframes(df1, df2)
    def extract_transform(path, keep_cols, cols_rows_with_missing_values, new_binary_target):
        _ , spark_df = SparkDataProcessor().process_data(
            data_csv_path= path,
            )

        return prepare_spark_df(
            spark_df=spark_df,
            keep_cols=keep_cols,
            cols_rows_with_missing_values=cols_rows_with_missing_values,
            new_binary_target = new_binary_target,
        )

    binary_target = 'Alpha-Fet'
    df_train = extract_transform("tirosinemia.csv", OPBG,  ['alfa-fetoprotein', 'sca (urine)', 'ntbc levels (dbs)'],binary_target)
    df_test  = extract_transform("tirosinemia_italia.csv", OPBG,  ['alfa-fetoprotein', 'sca (urine)', 'ntbc levels (dbs)'],binary_target)
    ray_df_train = ray.put(df_train)
    ray_df_test  = ray.put(df_test)
    features : list[str] = df_train.columns.tolist()
    features.remove(binary_target)
    
    studies = make_multiple_studies(
    ray_df_train, 
    features = features,
    targets = [binary_target],
    n_trials=10,
    Independent_testset = True,
    Independent_testset_df = ray_df_test
    )

    studies = ray.get(studies)
    ray.shutdown()
    import pickle

    with open('1a_hello_aws.pickle', 'wb') as handle:
        pickle.dump(studies, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print("Total time taken: {:.2f} seconds".format(elapsed_time))


    
    
