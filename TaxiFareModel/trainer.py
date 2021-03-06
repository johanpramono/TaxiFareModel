import numpy as np
import pandas as pd
from google.cloud import storage

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder, DistanceToCenter, DfOptimizer
from TaxiFareModel.utils import compute_rmse, df_optimized

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

import joblib

from TaxiFareModel.params import *

class Trainer():
    def __init__(self, X, y, model_name="linear", model=LinearRegression(), *args, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME
        self.model_name = model_name
        self.model = model

    def set_pipeline(self, dist_to_center=False):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler()),
            ('df_optimize', DfOptimizer())
        ])
        
        dist_center_pipe = Pipeline([
            ('dist_center_trans', DistanceToCenter()),
            ('stdscaler', StandardScaler()),
            ('df_optimize', DfOptimizer())
        ])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
            ('df_optimize', DfOptimizer())
        ])
        
        if dist_to_center:
            preproc_pipe = ColumnTransformer([
                ('distance_to_center', dist_center_pipe, ["pickup_latitude", "pickup_longitude"]),
                ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        else:
            preproc_pipe = ColumnTransformer([
                ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        
        pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('model', self.model)
        ])
        
        self.pipeline = pipeline
        
    def cross_validate(self):
        self.set_pipeline()
        cv_results = cross_validate(self.pipeline, self.X, self.y, cv=5, n_jobs=-1, 
                                    scoring={"r2" : "r2", "rmse": "neg_root_mean_squared_error"})
        metrics = {"r2" : cv_results["test_r2"].mean(), 
                   "rmse" : (-1*cv_results["test_rmse"].mean())}
        return metrics
        
    def run(self, dist_to_center):
        """set and train the pipeline"""
        self.dist_to_center = dist_to_center
        self.set_pipeline(self.dist_to_center)
        self.pipeline.fit(self.X, self.y)
        
        

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_param("model", self.model_name + str(int(self.dist_to_center)))
        self.mlflow_log_metric("rmse", rmse)
        self.mlflow_log_metric("dtc", int(self.dist_to_center))
        
        return compute_rmse(y_pred, y_test)
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
    def save_model(self, filename="model"):
        joblib.dump(self.pipeline, filename + '.joblib')
        print("saved model.joblib locally")
        
        self.upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")
    
    def upload_model_to_gcp(self):
        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('model.joblib')       
    
    
if __name__ == "__main__":
    # get data
    df = get_data()

    # clean and optimize data
    df = clean_data(df)
    df = df_optimized(df)
    
    # set X and y
    y = df.pop("fare_amount")
    X = df
    
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # train
    # for name, model in {"linear" : LinearRegression(), "ridge" : Ridge(), "lasso" : Lasso(), 
    #                     "elasticnet" : ElasticNet(), "random forest" : RandomForestRegressor()}.items():
    
    for name, model in {"random forest" : RandomForestRegressor()}.items():    
        
        """cross_validate"""
        # metrics_dict = trainer.cross_validate()
        # trainer.mlflow_log_param("model", name)
        # for metric, val in metrics_dict.items():
        #     trainer.mlflow_log_metric(metric, val)
        # print(f"Metrics for {name}: {metrics_dict}")
        
        """fitting"""
        for dtc in [True]:
            trainer = Trainer(X_train, y_train, model_name=name, model=model)
            trainer.run(dist_to_center=dtc)
        
            # evaluate
            rmse = trainer.evaluate(X_test, y_test)
            print(f"RMSE for {name}, dtc = {dtc}: {rmse}")

            trainer.save_model("model")
            print(f"model {name} is saved")
            
    # save model
    # trainer.save_model("model")
    
    # print(f"RMSE: {rmse}")
