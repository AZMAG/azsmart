"""
Use this to run/estimate models for real estate prices and rents.

TODO: look into replacing this w/ urbansim_template.

"""
from __future__ import print_function
import numpy as np
import pandas as pd
import orca

from urbansim.models.regression import RegressionModel

#from smartpy_sim.defaults import *


class PriceModelWrapper(object):
    """
    Use this to create and execute hedonic price
    models specified in a config dictionary.

    """

    def __init__(self, config):

        # global properties
        self.config_directory = config['config_directory']
        self.price_per_unit_column = config['price_per_unit_column']
        self.total_price_column = config['total_price_column']
        self.segment_column = config['segment_column']

        # loop through the models
        cols = set([
            self.price_per_unit_column,
            self.total_price_column,
            self.segment_column
        ])

        models = {}
        unit_cols = {}

        for k, v in config['models'].items():

            # open the yaml and create a regression model
            cfg = self.config_directory + '//' + v['config']
            model = RegressionModel.from_yaml(str_or_buffer=cfg)
            models[k] = model
            unit_cols[k] = v['units_column']

            # update the columns used
            cols = cols.union(set(model.columns_used()))

        self.models = models
        self.columns = list(cols)
        self.unit_cols = unit_cols

    def columns_used(self):
        """
        Returns the columns used across all price models

        """
        return self.columns

    def __call__(self, df, inplace=True):
        """
        Executes the regressions and updates the price columns.

        Parameters:
        -----------
        df: pandas.DataFrame
            Buildings data frame to compute prices for.
        inplace: bool, default True
            Indicates whether or not to copy the price series'.

        Returns:
        --------
        price_per_unit: pandas.Series
            Price per unit results, aligned to input data frame.
        total_price: pandas.Series
            Total price results, aligned to the input data frame.
        """

        # get price columns, retain existing values so that we can
        # preserve prices for building types we're not predicting
        price_per_unit = df[self.price_per_unit_column]
        total_price = df[self.total_price_column]
        if not inplace:
            price_per_unit = price_per_unit.copy()
            total_price = total_price.copy()

        # loop through all the  models
        for k, v in self.models.items():

            # make the per unit estimation
            # results = v.predict(df)
            # temporary -- remove rows with nulls
            # TODO: handle this before
            results = v.predict(
                df.dropna(subset=v.columns_used())
            )
            price_per_unit.loc[results.index] = results

            # get the total value
            units = df[self.unit_cols[k]]
            total_price.loc[results.index] = results * units.loc[results.index]

        return price_per_unit, total_price


@orca.injectable()
def price_model_config(config_root):
    """
    Specific the configuration for the price mode, bascially this is
    the yaml files and segments.

    """
    config_dir = '{}//Estimation//MCPC//REPM'.format(config_root)

    return {

        "config_directory": config_dir,
        "price_per_unit_column": "average_value_per_unit",
        "total_price_column": "total_fcv",
        "segment_column": "building_type_name",

        "models": {
            "single_family": {
                "segment": 'rsf',
                "config": "single_family_price.yaml",
                "units_column": "residential_units"
            },
            "multi_family": {
                "segment": 'rmf',
                "config": "multi_family_price.yaml",
                "units_column": "residential_units"
            },
            "retail": {
                "segment": 'retl',
                "config": "retail_price.yaml",
                "units_column": "non_residential_sqft"
            },
            "warehouse": {
                "segment": 'ware',
                "config": "warehouse_price.yaml",
                "units_column": "non_residential_sqft"
            },
            "industrial": {
                "segment": 'ind',
                "config": "ind_price.yaml",
                "units_column": "non_residential_sqft"
            },
            "office": {
                "segment": 'off',
                "config": "office_price.yaml",
                "units_column": "non_residential_sqft"
            },
            "med": {
                "segment": 'med',
                "config": "med_price.yaml",
                "units_column": "non_residential_sqft"
            },
            "hotel": {
                "segment": 'hot',
                "config": "hotel_price.yaml",
                "units_column": "non_residential_sqft"
            }
        }
    }


# initializes the price models
@orca.injectable(cache=True)
def price_model(price_model_config):
    return PriceModelWrapper(price_model_config)


# executes all the price models for a given year
@orca.step()
def price_models_step(year, price_model, buildings):
    """
    Executes the price models. This will update values on the buildings table.

    """
    bldgs_df = buildings.to_frame(price_model.columns_used())
    price_per_unit, total_price = price_model(bldgs_df)
    buildings.update_col(price_model.price_per_unit_column, price_per_unit)
    buildings.update_col(price_model.total_price_column, total_price)
