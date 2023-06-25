import os
import json

import couchdb
from bayes_opt import BayesianOptimization
import xgboost as xgb
import tkinter.messagebox
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


class Model:
    def __init__(self):
        # Initialize CouchDB server
        self.couch = couchdb.Server('http://admin:1234@localhost:5984/')

        # Create or open the database for test_dataframes
        self.db_name_dataframes = "test_dataframes"
        if self.db_name_dataframes in self.couch:
            self.db_dataframes = self.couch[self.db_name_dataframes]
        else:
            self.db_dataframes = self.couch.create(self.db_name_dataframes)

        # Create or open the database for test_results
        self.db_name_results = "test_results"
        if self.db_name_results in self.couch:
            self.db_results = self.couch[self.db_name_results]
        else:
            self.db_results = self.couch.create(self.db_name_results)

        self.le = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def add_files(self, docs_class0, docs_class1):
        self.docs_class0 = docs_class0
        self.docs_class1 = docs_class1

    def preprocess_data(self, docs, apply_smote=False):
        self.df = self.preprocess_all(docs)
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column] = self.df[column].astype(str)
                self.df[column] = self.le.fit_transform(self.df[column])
        self.imputer.fit(self.df)
        self.df_imputed = self.imputer.transform(self.df)
        self.y = self.le.fit_transform(self.df['ADHD'])

        self.df_imputed[np.isnan(self.df_imputed)] = 0
        self.df_imputed[np.isinf(self.df_imputed)] = 0

        variances = np.var(self.df_imputed, axis=0)
        zero_variance_indices = np.where(variances == 0)[0]
        self.df_imputed = np.delete(self.df_imputed, zero_variance_indices, axis=1)

        if apply_smote:
            k_neighbors = min(3, len(np.where(self.y == 1)[0]))
            if k_neighbors < 1:
                raise ValueError("Not enough samples in the minority class for SMOTE.")
            smote = SMOTE(k_neighbors=k_neighbors)
            self.X_resampled, self.y_resampled = smote.fit_resample(self.df_imputed, self.y)
            print(f"x_train shape: {self.X_resampled.shape}, y_train shape: {self.y_resampled.shape}")
            return self.X_resampled, self.y_resampled
        else:
            print(f"x_test shape: {self.df_imputed.shape}, y_test shape: {self.y.shape}")
            return self.df_imputed, self.y

    def train_model(self, train_files_class0, train_files_class1):
        self.X_train, self.y_train = self.preprocess_data(self.docs_class0 + self.docs_class1, apply_smote=True)

        def bo_tune_xgb(max_depth, gamma, learning_rate, alpha, reg_lambda):
            params = {'max_depth': int(max_depth),
                      'gamma': gamma,
                      'learning_rate': learning_rate,
                      'alpha': alpha,
                      'reg_lambda': reg_lambda,
                      'subsample': 0.8,
                      'eta': 0.1,
                      'eval_metric': 'auc',
                      'objective': 'binary:logistic',
                      'seed': 42}

            # Initialize StratifiedKFold
            skf = StratifiedKFold(n_splits=5)

            auc_scores = []

            for train_index, test_index in skf.split(self.X_train, self.y_train):
                X_train_fold, X_test_fold = self.X_train[train_index], self.X_train[test_index]
                y_train_fold, y_test_fold = self.y_train[train_index], self.y_train[test_index]

                dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
                dtest_fold = xgb.DMatrix(X_test_fold, label=y_test_fold)

                xgb_classifier_fold = xgb.train(params, dtrain_fold, num_boost_round=70)

                y_pred_proba_fold = xgb_classifier_fold.predict(dtest_fold)
                y_pred_fold = np.where(y_pred_proba_fold > 0.5, 1, 0)  # Convert probabilities to binary labels

                auc_score_fold = metrics.roc_auc_score(y_test_fold, y_pred_fold)
                auc_scores.append(auc_score_fold)

            # Handle NaN values in the cross-validation scores
            auc_scores = np.nan_to_num(auc_scores)  # replace NaNs with 0

            return -1.0 * np.mean(auc_scores)

        xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (4, 12),
                                                    'gamma': (0, 1),
                                                    'learning_rate': (0, 1),
                                                    'alpha': (0, 1),
                                                    'reg_lambda': (0, 5)
                                                    })

        if self.X_train.shape[0] < 5:
            raise ValueError("Not enough samples for Bayesian Optimization with 8 initial points.")
        xgb_bo.maximize(n_iter=3, init_points=5)

        self.params = xgb_bo.max['params']
        self.params['max_depth'] = int(self.params['max_depth'])

        # Plot the predicted probabilities for each fold
        self.plot_kfold_cv(self.X_train, self.y_train)

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        xgb_classifier = xgb.train(self.params, dtrain, num_boost_round=250)

        # Save the model
        xgb_classifier.save_model("xgb_model.json")
        print("Model trained successfully.")

        # # Plot feature importance
        # xgb.plot_importance(xgb_classifier)
        # plt.show()

    def plot_kfold_cv(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits)
        fold = 0
        plt.figure(figsize=(10, 6))

        for train_index, test_index in skf.split(X, y):
            fold += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            xgb_classifier = xgb.train(self.params, dtrain, num_boost_round=70)
            y_pred_proba = xgb_classifier.predict(dtest)

            plt.plot(range(len(y_pred_proba)), y_pred_proba, label=f'Predicted Probabilities Fold {fold}')

        plt.xlabel('Sample Number within Each Fold')
        plt.ylabel('Predicted Probability')
        plt.title('Predicted probabilities for each fold')
        plt.legend()
        plt.show()

    def add_test_data(self, docs):
        self.X_test, self.y_test = self.preprocess_data(docs)
        print(f"x_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")  # Print the shapes of X_test
        # and y_test
        self.df_imputed = self.imputer.transform(self.df)  # Transform the data here
        self.y_test = self.le.transform(self.df['ADHD'])

        # Replace NaN and infinite values with 0
        self.df_imputed[np.isnan(self.df_imputed)] = 0
        self.df_imputed[np.isinf(self.df_imputed)] = 0

        variances = np.var(self.df_imputed, axis=0)
        zero_variance_indices = np.where(variances == 0)[0]
        self.df_imputed = np.delete(self.df_imputed, zero_variance_indices, axis=1)

        self.X_test = self.df_imputed  # Use the imputed data directly as X_test

    def add_train_data(self, docs_class0, docs_class1):
        if isinstance(docs_class0, tuple):
            docs_class0 = [docs_class0]
        if isinstance(docs_class1, tuple):
            docs_class1 = [docs_class1]

        class0_dataframes = [self.process_individual_data(doc, 0, 0) for doc in docs_class0]
        if docs_class1:
            class1_dataframes = [self.process_individual_data(doc, 1, 1) for doc in docs_class1]
            all_dataframes = class0_dataframes + class1_dataframes
        else:
            all_dataframes = class0_dataframes

        return pd.concat(all_dataframes, ignore_index=True)

    def load_json_files(self, folder, adhd, meds):
        file_dataframes = []  # Accumulate the processed dataframes
        for file in os.listdir(folder):
            if file.endswith(".json"):
                with open(os.path.join(folder, file), 'r', encoding='utf-8-sig') as f:
                    try:
                        file_data = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON file {file}: {e}")
                        continue
                file_dataframe = self.process_individual_data(file_data, adhd, meds)
                file_dataframes.append(file_dataframe)  # Append the processed dataframe to the list
        return pd.concat(file_dataframes, ignore_index=True)  # Concatenate all dataframes into one

    def process_individual_data(self, doc, adhd, meds):
        print("ADHD label: ", adhd)
        file_data = doc  # Use the document directly
        individual = file_data['Individual']

        # Initialize an empty DataFrame
        df = pd.DataFrame()

        # Extract the document ID
        doc_id = doc['_id']

        # Convert ADHD to string and concatenate with doc_id
        adhd_doc_id = str(adhd) + "_" + doc_id

        for i, session_type in enumerate(["SessionWithoutDisturbances", "SessionWithDisturbances"]):
            # Calculate mean delay
            pressed_and_should = individual["Type"][session_type]["PressedAndShould"]
            times_of_should_press = individual["Type"][session_type]["TimesOfShouldPress"]
            mean_delay = self.calculate_mean_delay(pressed_and_should, times_of_should_press)

            # Calculate average vector size
            head_rotations = individual["Type"][session_type]["HeadRotation"]
            avg_vector_size = sum(
                np.linalg.norm(np.array([rotation['x'], rotation['y'], rotation['z']])) for rotation in
                head_rotations) / len(head_rotations)

            # Create a new DataFrame for the row
            new_row = pd.DataFrame({
                'ADHD': [adhd],
                'Medication': [meds],
                'Response Accuracy': [len(pressed_and_should)],
                'Commission Error': [len(individual["Type"][session_type]["PressedAndShouldNot"])],
                'Omission Error': [len(individual["Type"][session_type]["NotPressedAndShould"])],
                'Mean Delay': [mean_delay],
                'Average Vector Size': [avg_vector_size],  # Add the average vector size here
                'Session Type': [i],
                'Filename': [adhd_doc_id],
                'Name': [individual['Name']]
            })

            # Append new row to the DataFrame
            df = pd.concat([df, new_row], ignore_index=True)

        return df

    def calculate_mean_delay(self, pressed_and_should, times_of_should_press):
        delays = []
        for time in times_of_should_press:
            if time % 5 == 0:  # Check if the time is divisible by 5
                # Find the values in pressed_and_should that are within the 5 seconds interval after 'time'
                values_within_interval = [value for value in pressed_and_should if time <= value <= time + 5]
                if values_within_interval:  # If there are any values within the interval
                    closest_value = min(values_within_interval)  # Get the closest value to 'time'
                    delay = closest_value - time  # Calculate the delay
                    delays.append(delay)
        if delays:  # If there are any delays
            mean_delay = sum(delays) / len(delays)  # Calculate the mean delay
        else:
            mean_delay = 5  # If there are no delays, set the mean delay to 0
        print("Mean delay: ", mean_delay)
        return mean_delay

    def flatten_json(self, nested_json):
        out = {}

        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    flatten(x[a], name + a + '_')
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + '_')
                    i += 1
            else:
                out[name[:-1]] = x

        flatten(nested_json)
        return out

    def preprocess_all(self, docs):
        data = []
        for doc in docs:
            if isinstance(doc, str) and doc.isdigit():  # If doc is a string of digits
                # Fetch the document by its ID
                doc_content = self.db_results[doc]
            elif isinstance(doc, dict):  # If doc is a dictionary
                # Use the document directly
                doc_content = doc
            else:
                print(f"Unexpected type {type(doc)} encountered in docs. Expected a string of digits or a dictionary.")
                continue

            # Process the document content
            adhd = doc_content.get('ADHD')  # Use the get method to avoid KeyError
            if adhd is not None:  # Proceed only if ADHD key exists
                individual_data = self.process_individual_data(doc_content, adhd, meds=0)
                data.append(individual_data)

        df = pd.concat(data, ignore_index=True)
        df.to_csv('dataframe.csv', index=True)

        # Save to CouchDB
        self.save_to_couchdb(df)

        print(df)
        return df

    def save_to_couchdb(self, df):
        # Convert the dataframe to a dictionary
        df_dict = df.to_dict()

        # Save the dictionary to the test_dataframes database
        self.db_dataframes.save(df_dict)

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        unique_preds = np.unique(self.y_pred)
        if len(unique_preds) == 1:  # If the model is predicting only one class
            if unique_preds[0] == 0:  # If the predicted class is "NO ADHD"
                disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NO ADHD"])
            else:  # If the predicted class is "ADHD"
                disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ADHD"])
        else:
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NO ADHD", "ADHD"])
        fig, ax = plt.subplots()  # Create a new Figure with an axes
        disp.plot(ax=ax)  # Plot the confusion matrix on the axes

        return fig  # Return the Figure object

    def test_model(self):
        # Make predictions
        self.y_pred, self.y_pred_proba = self.model_predict(self.X_test)  # Store the returned values here
        # Calculate the accuracy and print the classification report
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Accuracy: {:.2f}".format(self.accuracy * 100))
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, zero_division=1))

        # Always plot predictions
        predictions_plot = self.plot_predictions(self.y_pred, self.y_test)

        # Check if there's only one class in the test data or predictions
        if len(np.unique(self.y_test)) == 1 or len(np.unique(self.y_pred)) == 1:
            print("Cannot plot confusion matrix because there's only one class in predictions or test data.")
            print(f"Predicted class for the single test instance: {self.y_pred[0]}")
            return predictions_plot, None
        else:
            # Assume model is an instance of the Model class
            confusion_matrix_plot = self.plot_confusion_matrix()
            return predictions_plot, confusion_matrix_plot

    def model_predict(self, X_test):
        loaded_xgb_classifier = xgb.Booster()
        try:
            loaded_xgb_classifier.load_model("xgb_model.json")
        except Exception as e:
            print(e)
            exit(-1)
        finally:
            if X_test.shape[1] > 0:  # If there's at least one feature left
                dtest = xgb.DMatrix(X_test)
                y_pred_proba = loaded_xgb_classifier.predict(dtest)
                print("Predicted probabilities:", y_pred_proba)
                y_pred = np.where(y_pred_proba > 0.5, 1, 0)  # Convert probabilities to binary labels
                return y_pred, y_pred_proba  # Return both the binary labels and the probabilities
            else:
                # If there's no features left, return a default prediction
                # Here, I'm returning an array of zeros with the same length as the number of samples in X_test
                return np.zeros(X_test.shape[0]), np.zeros(X_test.shape[0])

    def plot_predictions(self, y_pred, y_test):
        print(f"y_pred: {y_pred}, y_test: {y_test}")
        if y_pred.size == 0 or y_test.size == 0:
            print("y_pred or y_test is empty.")
            return None

        if np.isnan(y_pred).any() or np.isnan(y_test).any():
            print("y_pred or y_test contains NaN values.")
            return None

        if np.isinf(y_pred).any() or np.isinf(y_test).any():
            print("y_pred or y_test contains infinite values.")
            return None

        if y_pred.shape != y_test.shape:
            print("y_pred and y_test do not have the same shape.")
            return None

        y_pred = y_pred.flatten()

        fig = plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual Values')  # plot without sorting
        plt.plot(y_pred, label='Predictions')  # plot without sorting
        plt.xlabel('Sample Number')  # change 'Sorted Sample Number' to 'Sample Number'
        plt.ylabel('Value')
        plt.title(
            'Predictions vs Actual values')  # change 'Sorted Predictions vs Actual values' to 'Predictions vs Actual values'
        plt.legend()
        plt.show()  # Show the plot

        return fig  # Return the Figure object

    def predict_single(self, doc):
        # Preprocess the data
        X_single, _ = self.preprocess_data([doc])

        # Load the model
        loaded_xgb_classifier = xgb.Booster()
        loaded_xgb_classifier.load_model("xgb_model.json")

        # Create DMatrix
        dtest = xgb.DMatrix(X_single)

        # Predict
        y_pred_proba = loaded_xgb_classifier.predict(dtest)
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)  # Convert probabilities to binary labels

        return y_pred, y_pred_proba


def get_json_files(library):
    json_files = [file for file in os.listdir(library) if file.endswith(".json")]
    return json_files


def add_noise(value, noise_level):
    noise = np.random.uniform(low=-noise_level, high=noise_level)
    return value + noise
