from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

#def group_df_TEST(target_column):
    #for i in target_column
    #for group_df_TEST in [df[Config.CLASS_COL], df[Config.CLASS_COL].str.cat(df[Config.CLASS_COL3], sep=' + ')]:
    #    get_input_data(place_holder_target_column) = df[group_df_TEST]

    # new code, start
def group_df_TEST():
   ygroups = [df[Config.CLASS_COL],[df[Config.CLASS_COL] + ' ' + df[Config.CLASS_COL3]], [df[Config.CLASS_COL] + ' ' + df[Config.CLASS_COL3] + ' ' + df[Config.CLASS_COL4]]]
   for i in ygroups:
       print(i)

    # new code, end
   # df = load_data()
   # df = preprocess_data(df)
   # x = [df[Config.CLASS_COL], [df[Config.CLASS_COL], df[Config.CLASS_COL].str.cat(df[Config.CLASS_COL3], sep=' + ')]]
   # y = [[df[Config.CLASS_COL], [df[Config.CLASS_COL], df[Config.CLASS_COL].str.cat(df[Config.CLASS_COL3], sep=' + ')]]]

    # y = pd.DataFrame({'Type 2': [df[Config.CLASS_COL]], 'Type 3': [df[Config.CLASS_COL3]]})

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
    # for name, group_df([df[Config.CLASS_COL], df[Config.CLASS_COL].str.cat(df[Config.CLASS_COL3], sep=' + ')]) in grouped_df:
    # for name, group_df_TEST(df[Config.CLASS_COL]) in grouped_df:
    # for name, group_df_TEST in grouped_df:
    # for name, group_df in range(len(y[0])):
    # for name, group_df in range(y.shape[1]):

        print(name)
        print(group_df)
        X, group_df = get_embeddings(group_df)
        # X, group_df = get_embeddings(group_df_TEST(get_input_data(i)))
        # X, group_df = get_embeddings([row[name, group_df] for row in y()])
        # X, group_df = get_embeddings(y[group_df])
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)
        #print(perform_modelling(data, group_df, name))

        # new
        #for group_df_TEST in [ df[Config.CLASS_COL], df[Config.CLASS_COL].str.cat(df[Config.CLASS_COL3], sep=' + ')]:
        #    get_input_data(place_holder_target_column) = df[group_df_TEST]

    #if __name__ == '__main__':
    #    df = load_data_group1(df[Config.CLASS_COL])
    #    df = preprocess_data(df)
    #    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    #    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    #    grouped_df = df.groupby(Config.GROUPED)
    #    for name, group_df in grouped_df:
    #    for name, group_df([df[Config.CLASS_COL], df[Config.CLASS_COL].str.cat(df[Config.CLASS_COL3], sep=' + ')]) in grouped_df:
    #    for name, group_df_TEST(df[Config.CLASS_COL]) in grouped_df:
    #    for name, group_df in grouped_df:
    #        print(name)
    #        # X, group_df = get_embeddings(group_df)
    #        X, group_df = get_embeddings(group_df)
    #        data = get_data_object(X, group_df)
    #        perform_modelling(data, group_df, name)
    #        print(group_df_TEST)
