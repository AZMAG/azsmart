"""
...
"""
from __future__ import print_function, division

from collections import OrderedDict
import inspect

import numpy as np
import pandas as pd
from patsy import dmatrix
from prettytable import PrettyTable

#from . import util
#from ..utils import yamlio, choice
#from ..urbanchoice import mnl
from urbansim.models import util
from urbansim.utils import yamlio
from urbansim.urbanchoice import mnl
from . import choice


def get_mnl_utilities(data, model_expression, coeff):
    """
    Calculates MNL utilities for the provided interaction dataset.

    Parameters:
    -----------
    data: pandas.DataFrame
        Table containing interaction data.
    model_expression: str
        Patsy string defining the model specification.
    coeff: pandas.Series
        Coefficients to apply.

    Returns:
    --------
    pandas.Series with exponentiated utilities.

    """
    model_design = dmatrix(model_expression, data=data, return_type='dataframe')
    coeff = coeff.reindex(model_design.columns)

    coeff = np.reshape(np.array(coeff.values), (1, len(coeff)))
    model_data = np.transpose(model_design.values)
    utils = np.dot(coeff, model_data)
    exp_utils = np.exp(utils)

    return pd.Series(exp_utils.ravel(), data.index)


class MnlChoiceModel(object):
    """
    TODO: add doc strings
    """
    def __init__(self,
                 name,
                 parent=None,
                 model_expression=None,
                 choosers_segmentation_col=None,
                 alternatives_choice_col=None,

                 # fit options
                 choosers_fit_size=None,
                 choosers_fit_filters=None,
                 alternatives_fit_size=None,
                 alternatives_fit_filters=None,
                 alternatives_fit_sampling_weights_col=None,

                 # prediction options
                 alternatives_capacity_col=None,
                 choosers_predict_filters=None,
                 alternatives_predict_size=None,
                 alternatives_predict_filters=None,
                 alternatives_predict_sampling_weights_col=None,
                 predict_sampling_segmentation_col=None,
                 predict_sampling_within_percent=None,
                 predict_sampling_within_segments=None,
                 predict_max_iterations=100,
                 predict_sequential=False,

                 # fit results
                 fit_parameters=None,
                 log_likelihoods=None):

        # grab all the provided inputs
        local_args = locals()

        self._name = name
        self._parent = parent
        self._sub_models = OrderedDict()

        # if fit parameters is a dictionary, convert to data frame
        if fit_parameters is not None:
            if not isinstance(fit_parameters, pd.DataFrame):
                local_args['fit_parameters'] = pd.DataFrame(fit_parameters)

        # for most properties, prefix with '___'
        # these are properties we will potentially allow sub-models to inherit
        # from their parent
        for arg, value in local_args.items():
            if arg not in [self, 'name', 'parent']:
                self.__dict__['___{}'.format(arg)] = value

    def _sub_model_ignore(self):
        """
        List of properties not allowed to be overriden by a sub-model.

        """
        return [
            'parent',
            'choosers_segmentation_col',
            'alternatives_choice_col',
            'alternatives_capacity_col',
            'predict_max_iterations',
            'predict_sequential'
        ]

    def __getattr__(self, key):
        """
        Custom property accessor.

        """

        # if this is not managed, just return the property
        managed = [p for p in self.__dict__ if p.startswith('___')]
        m_key = '___{}'.format(key)
        if m_key not in managed:
            if key not in self.__dict__:
                raise AttributeError('{} not found'.format(key))
            return self.__dict__[key]

        # if there is no parent (this not a sub-model)
        # just return the managed property
        local_val = self.__dict__[m_key]
        parent = self.__dict__['_parent']
        if parent is None:
            return local_val

        # where necessary, inherit from the parent, 2 cases
        # 1 - properties not allowed to be overriden by a child
        # 2 - the child value is not defined
        parent_val = parent.__dict__[m_key]
        if key in self._sub_model_ignore() or local_val is None:
            return parent_val

        # finally, allow the sub-model to override the parent
        return local_val

    def __setattr__(self, key, value):
        """
        Custom property setter.

        """
        managed = [p for p in self.__dict__ if p.startswith('___')]
        m_key = '___{}'.format(key)
        if m_key in managed:
            self.__dict__[m_key] = value
        else:
            self.__dict__[key] = value

    @property
    def name(self):
        """
        Name of the model/sub-model.

        """
        return self._name

    @property
    def sub_models(self):
        """
        OrderedDict of sub-models.

        """
        return self._sub_models

    @property
    def parent(self):
        """
        Reference to the parent model.

        """
        return self._parent

    @property
    def is_submodel(self):
        """
        Indicates if this is a sub-model.

        """
        return self._parent is not None

    @property
    def has_submodels(self):
        """
        Indicates if this model contains sub-model definitions.

        """
        return len(self.sub_models) > 0

    @property
    def str_model_expression(self):
        """
        Model expression as a string suitable for use with patsy/statsmodels.

        """
        if self.model_expression is None:
            return None

        return util.str_model_expression(
            self.model_expression, add_constant=False)

    def add_sub_model(self, model):
        """
        Adds a sub-model to the parent/main model.

        ?? Do we need to deep copy this or add as an option?

        Parameters:
        -----------
        model: MnlChoiceModel
            Instantiated choice model to add.

        """
        self._sub_models[model.name] = model
        model._parent = self

    @classmethod
    def from_yaml(cls, yaml_str=None, str_or_buffer=None):
        """
        Create a MnlChoiceModel instance from a saved YAML configuration.
        Arguments are mutally exclusive.

        Parameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        Returns
        -------
        MnlChoiceModel

        """
        cfg = yamlio.yaml_to_dict(yaml_str, str_or_buffer, ordered=True)
        return MnlChoiceModel.from_dict(**cfg)

    @classmethod
    def from_dict(cls, **cfg):
        """
        Create a MnlChoiceModel instance from a saved dictionary.

        Parameters
        ----------
        **cfg: dict-like or keyword values

        Returns
        -------
        MnlChoiceModel

        """
        # create an instance of the model from the dictionary
        init_args = {a: cfg[a] for a in cfg if a != 'sub_models'}
        model = cls(**init_args)

        # add sub-models
        if 'sub_models' in cfg:
            for segment, sub_model_cfg in cfg['sub_models'].items():
                # model.add_sub_model(segment, **sub_model_cfg)
                model.add_sub_model(
                    MnlChoiceModel(segment, **sub_model_cfg))

        return model

    def to_dict(self):
        """
        Return a OrderedDict respresentation of an MnlChoiceModel
        instance.

        """

        # get the model parameters/argument
        # remember some properties cannot be overriden by sub-models
        ignore = ['self', 'parent', 'name']
        if self.is_submodel:
            ignore += self._sub_model_ignore()
        model_args = [a for a in inspect.getargspec(self.__init__).args if a not in ignore]

        # grab the internally managed property values
        # remember, these properteis are prefixed with '___'
        d = OrderedDict()
        if not self.is_submodel:
            d['name'] = self.name
        for model_arg in model_args:
            prop = self.__dict__['___{}'.format(model_arg)]
            if prop is not None:
                if isinstance(prop, pd.DataFrame):
                    # need to convert data frame objects
                    prop = yamlio.frame_to_yaml_safe(prop, True)
                d[model_arg] = prop

        # add submodels
        if self.has_submodels:
            d['sub_models'] = {k: v.to_dict() for k, v in self.sub_models.items()}

        return d

    def to_yaml(self, str_or_buffer=None):
        """
        Save a model respresentation to YAML.

        Parameters
        ----------
        str_or_buffer : str or file like, optional
            By default a YAML string is returned. If a string is
            given here the YAML will be written to that file.
            If an object with a ``.write`` method is given the
            YAML will be written to that object.

        Returns
        -------
        j : str
            YAML is string if `str_or_buffer` is not given.

        """
        return yamlio.convert_to_yaml(self.to_dict(), str_or_buffer)

    def interaction_columns_used(self, segment=None):
        """
        Returns all columns used by the model expression.

        Parameters:
        ----------
        segment: value, optional, default None
            Optionally limit the columns to those used by a specific sub-model.

        Returns:
        --------
        List of column names

        """
        if segment is not None:
            return self.sub_models[segment].interaction_columns_used()

        # columns for the current model
        self_cols = util.columns_in_formula(self.model_expression)

        # strip _alt or _chooser from column names, these are
        # for cases where the same column exists on both choosers and alts
        self_cols = [x.replace('_chooser', '') for x in self_cols]
        self_cols = [x.replace('_alt', '') for x in self_cols]
        self_cols = set(self_cols)

        # columns for all sub-models
        sm_cols = []
        if self.has_submodels:
            sm_cols = [sm.interaction_columns_used() for sm in self.sub_models.values()]

        return list(self_cols.union(*sm_cols))

    def choosers_columns_used(self, segment=None, choosers=None, fit=True, predict=True):
        """
        Columns from the choosers table used in the model.

        Parameters:
        ----------
        segment: value, optional, default None
            Optionally limit the columns to those used by a specific sub-model.
        choosers: pandas.DataFrame or table-like optional, default None
            If provided, the result will include columns from used by
            the model expression (i.e. all model columns).
            If not provided, this is limited to columns used by filters.
        fit: bool, optional, default True
            If False, columns only needed for fitting are ignored.
        predict: bool, optional, default True
            If False, columns only needed for predicting are ignored.

        Returns:
        --------
        List of column names

        """
        if segment is not None:
            return self.sub_models[segment].choosers_columns_used(None, choosers, fit, predict)

        # get filter cols in the current model
        self_cols = set([self.choosers_segmentation_col])

        if fit:
            self_cols = self_cols.union(
                set(util.columns_in_filters(self.choosers_fit_filters)))

        if predict:
            self_cols.add(self.predict_sampling_segmentation_col)
            self_cols = self_cols.union(
                set(util.columns_in_filters(self.choosers_predict_filters)))

        # get expression columns for the current model
        if choosers is not None:
            exp_cols_all = self.interaction_columns_used()
            exp_cols = [c for c in exp_cols_all if c in choosers.columns]
            self_cols = self_cols.union(set(exp_cols))

        # get sub-model columns
        sm_cols = set()
        if self.has_submodels:
            sm_cols = [sm.choosers_columns_used(
                None, choosers, fit, predict)for sm in self.sub_models.values()]

        all_cols = self_cols.union(*sm_cols)
        return [c for c in all_cols if c is not None]

    def alternatives_columns_used(self, segment=None, alternatives=None, fit=True, predict=True):
        """
        Columns from the alternatives table used in the model.

        Parameters:
        ----------
        segment: value, optional, default None
            Optionally limit the columns to those used by a specific sub-model.
        alternatives: pandas.DataFrame or table-like optional, default None
            If provided, the result will include columns from used by
            the model expression (i.e. all model columns).
            If not provided, this is limited to columns used by filters.
        fit: bool, optional, default True
            If False, columns only needed for fitting are ignored.
        predict: bool, optional, default True
            If False, columns only needed for predicting are ignored.

        Returns:
        --------
        List of column names

        """
        if segment is not None:
            return self.sub_models[segment].alternatives_columns_used(
                None, alternatives, fit, predict)

        # get filters in the current model
        self_cols = set()
        if fit:
            self_cols.add(self.alternatives_fit_sampling_weights_col)
            self_cols = self_cols.union(
                set(util.columns_in_filters(self.alternatives_fit_filters)))

        if predict:
            self_cols.add(self.predict_sampling_segmentation_col)
            self_cols.add(self.alternatives_predict_sampling_weights_col)
            self_cols.add(self.alternatives_capacity_col)
            self_cols = self_cols.union(
                set(util.columns_in_filters(self.alternatives_predict_filters)))

        # get expression columns for the current model
        if alternatives is not None:
            exp_cols_all = self.interaction_columns_used()
            exp_cols = [c for c in exp_cols_all if c in alternatives.columns]

            self_cols = self_cols.union(set(exp_cols))

        # get filters in sub-models
        sm_cols = []
        if self.has_submodels:
            sm_cols = [sm.alternatives_columns_used(
                None, alternatives, fit, predict) for sm in self.sub_models.values()]

        all_cols = self_cols.union(*sm_cols)
        return [c for c in all_cols if c is not None]

    def columns_used(self, segment=None, fit=True, predict=True):
        """
        Columns from any table used in the model. May come from either
        the choosers or alternatives tables.

        Parameters:
        ----------
        segment: value, optional, default None
            The segment of the sub-model to get columns for.
        fit: bool, optional, default True
            If False, columns only needed for fitting are ignored.
        predict: bool, optional, default True
            If False, columns only needed for predicting are ignored.

        """
        return list(set().union(*[
            self.choosers_columns_used(segment, fit=fit, predict=predict),
            self.alternatives_columns_used(segment, fit=fit, predict=predict),
            self.interaction_columns_used(segment)
        ]))

    def report_fit(self):
        """
        Print a report of the fit results.

        """
        if self.fit_parameters is None:
            print('Model not yet fit.')
            return

        print('Null Log-liklihood: {0:.3f}'.format(
            self.log_likelihoods['null']))
        print('Log-liklihood at convergence: {0:.3f}'.format(
            self.log_likelihoods['convergence']))
        print('Log-liklihood Ratio: {0:.3f}\n'.format(
            self.log_likelihoods['ratio']))

        tbl = PrettyTable(
            ['Component', ])
        tbl = PrettyTable()

        tbl.add_column('Component', self.fit_parameters.index.values)
        for col in ('Coefficient', 'Std. Error', 'T-Score'):
            tbl.add_column(col, self.fit_parameters[col].values)

        tbl.align['Component'] = 'l'
        tbl.float_format = '.3'

        print(tbl)

    def fit(self, choosers, alternatives, current_choice):
        """
        Fit and save model parameters based on given data. Will update the
        `fit_parameters` and `log_likelihoods` properties on the model or
        sub-models.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing,
            e.g. buildings.
        current_choice : pandas.Series or any
            A Series describing the `alternatives` currently chosen
            by the `choosers`. Should have an index matching `choosers`
            and values matching the index of `alternatives`.
            If a non-Series is given it should be a column in `choosers`.

        """

        # make sure alternative IDs are on the index
        if self.alternatives_choice_col is not None:
            alternatives = alternatives.set_index(self.alternatives_choice_col, drop=False)

        # apply upper/main level filters
        if self.is_submodel:
            c_filters = self.parent.choosers_fit_filters
            a_filters = self.parent.alternatives_fit_filters
        else:
            c_filters = self.choosers_fit_filters
            a_filters = self.alternatives_fit_filters
        choosers = util.apply_filter_query(choosers, c_filters)
        alternatives = util.apply_filter_query(alternatives, a_filters)

        # if chooser segmentation is defined, but there are no sub-models, add a sub-model
        # for each unique value
        if self.choosers_segmentation_col is not None and not self.has_submodels:
            for s in choosers[self.choosers_segmentation_col].unique():
                if not isinstance(s, str):
                    if isinstance(s, unicode):
                        s = str(s)
                    else:
                        # this can be problematic, todo: look for a better approach
                        s = np.asscalar(s)

                self.add_sub_model(MnlChoiceModel(s))

        # inner function to handle the fit for a given model
        def fit_model(key, model, choosers, alternatives, current_choice):
            print('fitting {}...'.format(key))

            # apply sub-model filters
            if model.is_submodel:
                if model.choosers_segmentation_col is not None:
                    f = "{} == '{}'" if isinstance(key, str) else '{} == {}'
                    f = f.format(model.choosers_segmentation_col, key)
                    choosers = util.apply_filter_query(choosers, f)

                alternatives = util.apply_filter_query(alternatives, model.alternatives_fit_filters)

            # get observed choicess
            if isinstance(current_choice, pd.Series):
                current_choice = current_choice.loc[choosers.index]
            else:
                current_choice = choosers[current_choice]

            # just keep alternatives who have chosen an available alternative
            in_alts = current_choice.isin(alternatives.index)
            choosers = choosers[in_alts]
            current_choice = current_choice[in_alts]

            # just keep interaction columns
            choosers = choosers[model.choosers_columns_used(None, choosers, False, False)]
            alternatives = alternatives[model.alternatives_columns_used(
                None, alternatives, False, False)]

            # sample choosers for estimation
            num_choosers = model.choosers_fit_size

            if num_choosers is not None:
                if num_choosers < 1:
                    # the parameter is expressed as a percentage of the available choosers
                    num_choosers = 1 + int(len(choosers) * num_choosers)
                num_choosers = min(num_choosers, len(choosers))
                idx = np.random.choice(choosers.index, num_choosers, replace=False)
                choosers = choosers.loc[idx]
                current_choice = current_choice.loc[idx]

            # get interaction data
            sample_size = model.alternatives_fit_size
            print ('alts to sample: {}'.format(sample_size))
            print ('num choosers: {}'.format(num_choosers))

            sampling_weights = model.alternatives_fit_sampling_weights_col
            interact, sample_size, chosen = choice.get_interaction_data_for_estimation(
                choosers, alternatives, current_choice, sample_size, sampling_weights)

            # get the design matrix
            model_design = dmatrix(
                model.str_model_expression, data=interact, return_type='dataframe')

            # estimate and report
            # TODO: make the estimation method a callback provided as a model argument
            log_likelihoods, fit_parameters = mnl.mnl_estimate(
                model_design.values, chosen, sample_size)
            fit_parameters.index = model_design.columns

            model.fit_data = interact
            model.fit_chosen = chosen
            model.log_likelihoods = log_likelihoods
            model.fit_parameters = fit_parameters

            model.report_fit()
            # logger.debug('finish: fit DCM model {}'.format(name))

        # get the collection of models to fit
        models = {None: self}
        if self.has_submodels:
            models = self.sub_models

        for key, model in models.items():
            fit_model(key, model, choosers, alternatives, current_choice)

    @property
    def fitted(self):
        """
        True if model is ready for prediction.

        """
        if self.has_submodels:
            return all([m.fit_parameters is not None for m in self.sub_models.values()])
        else:
            return self.fit_parameters is not None

    def assert_fitted(self):
        """
        Raises `RuntimeError` if the model is not ready for prediction.

        """
        if not self.fitted:
            raise RuntimeError('Model has not been fit.')

    def predict(self, choosers, alternatives, debug=False):
        """
        Make choice predictions using fitted results.

        Parameters:
        -----------
        choosers: pandas.DataFrame
            Agents making choices.
        alternatives: pandas.DataFrame
            Choice set to choose from.
        debug: optional, default False
            If True, diagnostic information about the alternatives,
            utilities and probabilities will be returned.


        Returns:
        -------
        choices: pandas.Series

        capacities: pandas.Series

        verbose: pandas.DataFrame

        """

        # make sure alternative IDs are on the index
        if self.alternatives_choice_col is not None:
            alternatives = alternatives.set_index(self.alternatives_choice_col, drop=False)

        # apply upper/main level filters
        if self.is_submodel:
            c_filters = self.parent.choosers_predict_filters
            a_filters = self.parent.alternatives_predict_filters
        else:
            c_filters = self.choosers_predict_filters
            a_filters = self.alternatives_predict_filters
        choosers = util.apply_filter_query(choosers, c_filters)
        alternatives = util.apply_filter_query(alternatives, a_filters)

        # if sub-modeled, only keep choosers in a valid segment
        if self.has_submodels:

            keep = choosers[self.choosers_segmentation_col].isin(self.sub_models)
            choosers = choosers[keep]

        # get the alternative capacities
        capacities = None
        constrained = self.alternatives_capacity_col is not None
        if constrained:
            capacities = choice.get_capacities(
                alternatives, self.alternatives_capacity_col)

        # get the collection of models to predict
        models = {None: self}
        if self.has_submodels:
            models = self.sub_models

        # if running in a constrained AND sequential manner, remove capacities
        # after each sub-model; if not running sequentially leave the removal of
        # capacities to the constrained choice method
        sequential = self.predict_sequential

        if constrained and sequential:
            remove_cap = True
        else:
            remove_cap = False

        # callback for making constrained choices across all models
        def main_callback(choosers, alternatives, verbose, capacities, as_callback=True):

            choices_all = []
            verbose_all = []

            for key, model in models.items():

                # need to do this if we don't have sub-models?
                curr_choosers = choosers
                curr_alts = alternatives

                # apply sub-model filters
                if model.is_submodel:
                    if model.choosers_segmentation_col is not None:
                        f = "{} == '{}'" if isinstance(key, str) else '{} == {}'
                        f = f.format(model.choosers_segmentation_col, key)
                        curr_choosers = util.apply_filter_query(choosers, f)

                    curr_alts = util.apply_filter_query(
                        alternatives, model.alternatives_predict_filters)

                # reindex the capacities to match the alternatives
                curr_cap = None
                if constrained:
                    curr_cap = capacities.reindex(alternatives.index)

                # define the weights callback (generates utilities)
                def weights_cb(interaction_data):
                    return get_mnl_utilities(
                        interaction_data,
                        model.str_model_expression,
                        model.fit_parameters['Coefficient']
                    )

                # make the choice
                choice_cb = choice.IndividualChoiceCallback(
                    weights_cb,
                    model.predict_sampling_segmentation_col,
                    model.alternatives_predict_size,
                    model.alternatives_predict_sampling_weights_col,
                    model.predict_sampling_within_percent,
                    model.predict_sampling_within_segments
                )
                if remove_cap:
                    c, capacities, v = choice.constrained_choice(
                        curr_choosers,
                        curr_alts,
                        choice_cb,
                        capacity=curr_cap,
                        verbose=verbose,
                        max_iterations=self.predict_max_iterations
                    )
                else:
                    c, v = choice_cb(curr_choosers, curr_alts, verbose)

                if c is not None:
                    choices_all.append(c)

                if v is not None and verbose:
                    verbose_all.append(v)

            # combine results across models
            choices_concat = pd.concat(choices_all) if len(choices_all) > 0 else None
            verbose_concat = pd.concat(verbose_all) if len(verbose_all) > 0 else None

            if as_callback:
                return choices_concat, verbose_concat
            else:
                return choices_concat, capacities, verbose_concat

        # main execution
        if sequential or not constrained:
            # run each sub-model in order and allow it to fully complete
            choices, cap, pdf = main_callback(
                choosers, alternatives, debug, capacities, as_callback=False)
            choices = choices.reindex(choosers.index)
        else:
            # allow choosers across models to compete for the same alternatives
            choices, cap, pdf = choice.constrained_choice(
                choosers,
                alternatives,
                main_callback,
                capacity=capacities,
                max_iterations=self.predict_max_iterations,
                verbose=debug
            )

        # now do something else?
        self.sim_pdf = pdf
        return choices, cap, pdf
