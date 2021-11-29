"""
Contains utilities for making choices.

"""
from __future__ import print_function, division
import math
import numpy as np
import pandas as pd

# from .misc import fidx
from smartpy_core.wrangling import broadcast as fidx

##################
# GENERAL UTILS
##################


def seeded_call(seed, func, *args, **kwargs):
    """
    Executes a function with the provided numpy seed.
    Reverts the numpy PRNG back to the previous state
    after the function call. Allows for reproducing results
    for functions with random dependencies.

    Parameters:
    -----------
    seed: numeric
        The seed to provide the PRNG.
    func: callable
        The function to execute.
    *args, **kwargs:
        Ordered and named arguments to pass to func.

    Returns:
    --------
    The results of the provided function, given the provided
    arguments.

    """
    old_state = np.random.get_state()
    np.random.seed(seed)
    results = func(*args, **kwargs)
    np.random.set_state(old_state)
    return results


def get_probs(weights):
    """
    Returns probabilities for a series of weights. Null
    values are treated as 0s.

    Parameters:
    -----------
    weights: pandas.Series
        Series to get probabilities for.

    Returns:
    --------
    pandas.Series

    """
    w = weights.fillna(0)

    w_sum = w.sum()
    if w_sum == 0:
        probs = pd.Series(np.ones(len(w)) / len(w), index=w.index)
    else:
        probs = w / w_sum

    return probs


def get_segmented_probs(df, w_col, segment_cols):
    """
    Converts a series of weights into probabilities across multiple
    segments. Null values are treated as 0s. Segments containing
    all nulls and/or 0s will have equal probabilities.

    Parameters:
    -----------
    df: pandas.DataFrame
        Data frame containing weights.
    w_col: str
        Name of column containing weights.
    segment_cols: str, list of str, series, index, ...
        Defines segments to generate probabilties for.

    Returns:
    --------
    pandas.Series with probabilties

    """

    # get probabilities
    w_sums = fidx(df.groupby(segment_cols)[w_col].sum(), df, segment_cols)
    probs = df[w_col] / w_sums

    # handle nulls
    w_sums = w_sums.fillna(0)
    probs = probs.fillna(0)

    # handle cases where all weights in a segment are 0
    z_sums = w_sums == 0
    if z_sums.any():
        w_cnts = fidx(df.groupby(segment_cols).size(), df, segment_cols)
        probs[z_sums] = 1 / w_cnts[z_sums]

    return probs


def randomize_probs(p):
    """
    Randomizes probabilities to be used in weighted sorted sampling.
    The probabilities will no longer sum to 1.

    Parameters:
    -----------
    p: pandas.Series
        Series containing probablilties to randomize.

    Returns:
    --------
    pandas.Series

    """
    return np.power(np.random.rand(len(p)), 1.0 / p)


def segmented_sample_no_replace(amounts, data, segment_cols, w_col=None):
    """
    Returns samples without replacement in a segmented manner (i.e. each
    segment has a different amount to sample).

    Parameters:
    -----------
    amounts: pandas.Series
        Amounts to sample. Should be indexed by the segment.
    data: pandas.DataFrame
        Data to sample from.
    segment_cols: str or list of str
        Columns defining segments on the data. Should
        match the index of the amounts.
    w_col: str, optional, default None
        If provided, defines sampling weights.
        These will be converted to probabilities for each segment.
        If None the sample is random.

    Returns:
    --------
    numpy array with the indexes of the chosen rows.

    """
    if not isinstance(segment_cols, list):
        segment_cols = [segment_cols]

    # sort based on the weights
    if w_col is None:
        # totally random
        ran_p = np.random.rand(len(data))
    else:
        # apply weights to get randomized probabilities
        probs = get_segmented_probs(data, w_col, segment_cols)
        ran_p = randomize_probs(probs)

    data = data.copy()
    data['ran_p'] = ran_p
    data.sort_values('ran_p', ascending=False, inplace=True)

    # choose items whose relative ordering is smaller than the amount needed
    amounts = fidx(amounts, data, segment_cols)
    cc = data.groupby(segment_cols).cumcount() + 1
    to_sample = cc <= amounts

    return data[to_sample].index.values


def get_capacities(df=None, capacities=None):
    """
    Returns capacity information from the provided data.

    Parameters:
    -----------
    df: pandas.DataFrame, optional, default None
        Data frame containing capacity information.
    capacities str or pd.Series, optional, default None
        Either a series containing capacity information or
        a column name in the data frame.
        If None, will assume each row in the data frame has a
        single unit of capacity.

    Returns:
    --------
    pandas.Series

    """

    if capacities is None:
        return pd.Series(np.ones(len(df)), index=df.index)

    if isinstance(capacities, pd.Series):
        return capacities.copy()

    return df[capacities].copy()


#################################
# AGGREGATE CHOICE / PREDICTION
#################################


def simple_aggregate_choice(choosers, alternatives):
    """
    Chooses among alternatives based on the provided weights without regard
    for alternative capacity constraints (i.e. with replacement).
    Choosers are randomly assigned to the chosen alternatives.

    Parameters:
    -----------
    choosers: pandas.DataFrame or pandas.Series
        Agents to make choices.
    alternatives: pandas.Series
        Choiceset. Index represents the alternative ID, the
        values to the weights.

    Returns:
    --------
    choices: pandas.DataFrame
       Data frame of chosen indexes, aligned to the choosers.
       Column containing choices is called `alternative_id`
    probs: pandas.DataFrame
        DataFrame of probabilites, aligned to the alternatives.
        Column containing probabilities is called `prob`.

    """
    probs = get_probs(alternatives)
    choices = pd.Series(
        np.random.choice(
            probs.index.values, size=len(choosers), p=probs, replace=True),
        index=choosers.index
    )

    return choices.to_frame('alternative_id'), probs.to_frame('prob')


def constrained_aggregate_choice(choosers, alt_weights, alt_capacities=None,
                                 normalize_probs=True):
    """
    Chooses among alternatives based on the provided weights, while respecting
    alternative capacities. Choosers are randomly assigned to the chosen alternatives.

    To mimimic the existing behavior of `urbansim.models.dcm.unit_choice`, run this
    withouth providing `alt_capacities`.

    Parameters:
    -----------
    choosers: pandas.DataFrame or pandas.Series
        Agents to make choices.
    alt_weights: pandas.Series
        Weights for making the choice, indexed by the alternative ID.
    alt_capacities: pandas.Series, optional, default None
        Capacities for constraining the choice, indexed by the alternative ID.
        If None, all alternatives assumed to have a capacity of 1.
    normalize_probs: bool, optional, default True
        Only applicable if `alt_capacities` is provided.
        If True, unit-level probabilites are normalized so that their sum matches
        the alternative-level probabilities when computed independently, thereby
        removing the influence of capacities on the choice.
        If False, the unit-level probabilities will NOT sum to the alternative-level
        probabilities, thus allowing the capacity distributions to influence the choice.

    Returns:
    --------
    choices: pandas.DataFrame
       Data frame of chosen indexes, aligned to the choosers.
       Column containing choices is called `alternative_id`
    probs: pandas.DataFrame
        DataFrame of probabilites, aligned to the alternatives.
        Column containing probabilities is called `prob`.

    """
    if alt_capacities is None:
        # if no capacities provided, assume each row is 1 unit
        unit_probs = get_probs(alt_weights)
        alt_probs = unit_probs
    else:
        # filter out alternatives without capacity
        gtz = (alt_capacities > 0) & (alt_weights > 0)
        cap = alt_capacities[gtz]
        w = alt_weights[gtz]

        # if NOT normalizing unit level probabilities
        # adjust weights based on capacities
        # note: this is equivalent to repeating the weights based on the capacity
        #       and then calaculating probabilities from this.
        if not normalize_probs:
            w = w * cap

        # calaculate probabilities
        probs = get_probs(w)
        alt_probs = probs.reindex(alt_weights.index).fillna(0)
        unit_probs = probs / cap

        # explode alternatives to so we have 1 row for each unit of capacity
        unit_probs = unit_probs.repeat(cap.astype(int))

    if len(choosers) >= len(unit_probs):
        # if we have more choosers than capacity, randomly pick among choosers
        idx = np.random.choice(
            choosers.index.values, size=len(unit_probs), replace=False)
        choices = pd.Series(index=choosers.index)
        choices.loc[idx] = unit_probs.index.values

    else:
        choices = pd.Series(
            np.random.choice(
                unit_probs.index.values, size=len(choosers), p=unit_probs, replace=False),
            index=choosers.index
        )

    return choices.to_frame('alternative_id'), alt_probs.to_frame('prob')


def aggregate_choice(choosers,
                     alternatives,
                     weights,
                     constrained,
                     segment_col=None,
                     capacities=None,
                     normalize_probs=True):
    """
    General method for making aggregate choices from a set of alternatives
    and assigning them to agents. Supports both constrained and un-constrained cases
    as well as segmented choices.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Agents to make choices.
    alternatives: pandas.DataFrame
        Alternatives to choose from.
    weights: str or pandas.Series or callable
        Weights used to guide the choice:
            If str, this references a column on the alterantives data frame.
            If pandas.Series, should be indexed to the alternatives data frame.
            If callable, should returns a series of weights for alterantives. The
            callable should accept the alterantives as an input argument.
    constrained: bool
        If True, the choices will respect alternative capacities.
    segment_col: str, optional, default None
        Column on both used choosers and alterantives used to segment
    capacities: pandas.Series, optional, default None
        Capacities for constraining the choice, indexed by the alternative ID.
        If None, all alternatives assumed to have a capacity of 1.
    normalize_probs: bool, optional, default True
        Only applicable if `alt_capacities` is provided.
        If True, unit-level probabilites are normalized so that their sum matches
        the alternative-level probabilities when computed independently, thereby
        removing the influence of capacities on the choice.
        If False, the unit-level probabilities will NOT sum to the alternative-level
        probabilities, thus allowing the capacity distributions to influence the choice.

    Returns:
    --------
    choices: pandas.DataFrame
       Data frame of chosen indexes, aligned to the choosers.
       Column containing choices is called `alternative_id`
    probs: pandas.DataFrame
        DataFrame of probabilites, aligned to the alternatives.
        Column containing probabilities is called `prob`.

    """

    def make_the_choice(choosers, alternatives):

        # get the weights
        if isinstance(weights, pd.Series):
            w = weights.loc[alternatives.index]
        elif isinstance(weights, str):
            w = alternatives[weights]
        elif callable(weights):
            w = weights(alternatives)

        # make the choice
        if constrained:
            return constrained_aggregate_choice(
                choosers, w, capacities, normalize_probs)
        else:
            return simple_aggregate_choice(choosers, w)

    # if no segmentation column is provided, just make a single aggregate choice
    if segment_col is None:
        return make_the_choice(choosers, alternatives)

    # otherwise make a choice for each segment
    choices_all = []
    probs_all = []

    for segment in choosers[segment_col].unique():

        # get current choosers and alternatives
        curr_choosers = choosers[choosers[segment_col] == segment]
        curr_alts = alternatives[alternatives[segment_col] == segment]

        # get choices and probabilities for the current segment
        curr_choices, curr_probs = make_the_choice(curr_choosers, curr_alts)
        choices_all.append(curr_choices)
        probs_all.append(curr_probs)

    # combine results across segments
    choices_concat = None
    probs_concat = None

    if len(choices_all) > 0:
        choices_concat = pd.concat(choices_all).reindex(choosers.index)
        probs_concat = pd.concat(probs_all).reindex(alternatives.index)

    return choices_concat, probs_concat


###############################
# SAMPLING / INTERACTION DATA
###############################


def sample_alts_2d(alternatives, choosers, sample_size=None, observed=None,
                   sampling_weights=None, max_iter=100):
    """
    Samples alternatives for each chooser for use in estimation or prediction.

    Parameters:
    -----------
    alternatives: pandas.DataFrame or Series
        Alternatives to sample. The resulting sample contains the index values.
        Multi-index is NOT supported.
    choosers: pandas.DataFrame or Series
        Choosers to sample alternatives for.
    sample_size: int, optional, default None
        If provided defines the number of alternatives to sample for each chooser.
        If None, the same alternatives will be provided to each chooser.
    observed: pandas.Series, optional, default None
        Series of observed chosen alternative IDs.
        If provided, the observed choices will be included in the sample and an
        additional array will be returned indicating locations of the
        observed choices in the sample. Should be aligned to choosers.
    sampling_weights: pandas.Series, optional, default None
        Weights used to guide sampling. Should be aligned to alternatives.
    max_iter: int, optional, default 100
        Number of iterations used to attempt to remove row-wise duplicates.

    Returns:
    --------
    sample: numpy array
        Contains the index values of the sampled alternatives.
        If sample size is provide, shape (num_choosers, sample_size)
        If no sample size, shape of (num_choosers, num alts)

    sample_size: int
        Number of samples for each agent (i.e. columns).
        Equals sample_size if provided.
        Equals number of alternatives if no sample size provided.

    chosen: numpy array (only returned if `observed` arg provided)
        Same shape as sample array.
        1s indicate the sample is the observed, 0s otherwise.
        Each row should sum to 1.

    """
    alt_ids = alternatives.index.values
    num_alts = len(alt_ids)
    num_choosers = len(choosers)

    if observed is not None:
        observed_arr = observed.values.reshape(num_choosers, 1)

    # handle sample weights
    sampling_probs = None
    if sampling_weights is not None:
        sampling_probs = get_probs(sampling_weights).values

    # if not sampling, just use same alternatives for each chooser
    if sample_size is None or sample_size >= num_alts:
        sample_size = num_alts
        sample = np.tile(alt_ids, num_choosers).reshape(num_choosers, num_alts)

    else:
        # perfrom the initial sample
        sample = np.random.choice(
            alt_ids, size=num_choosers * sample_size, p=sampling_probs)
        sample = sample.reshape(num_choosers, sample_size)

        # if observations provided, add these to the sample
        if observed is not None:
            sample = np.hstack([
                observed_arr,
                sample
            ])

        # refine sample to eliminate row level duplicates
        curr_iter = 0
        while curr_iter < max_iter:

            # sort the sample IDs
            sample.sort()

            # get duplicates
            is_dup = np.hstack([
                np.full((num_choosers), False, dtype=bool).reshape(num_choosers, 1),
                sample[:, 1:] == sample[:, :-1]
            ])

            num_dups = np.sum(is_dup)
            if num_dups == 0:
                break

            # resample
            new_samples = np.random.choice(alt_ids, size=num_dups, p=sampling_probs)
            sample[is_dup] = new_samples
            curr_iter += 1

    if observed is None:
        return sample, int(sample.shape[1])

    # identify the sample positions of observed choices
    chosen = None
    chosen = 1 * (observed_arr == sample)
    return sample, int(sample.shape[1]), chosen


def sample_2d_no_compete(alternatives, choosers, sample_size, capacities=None):
    """
    Samples alternatives in a manner that guarantees that capacities will
    be respected: the maximum # of times a given alternative can appear
    in the sample (across choosers) is equal to its capacity.

    Use this for the last iteration of a constrained choice, this allows for
    locating all remaining choosers in the last iteration (capacities permitting).

    Parameters:
    -----------
    alternatives: pandas.DataFrame or Series
        Alternatives to sample. The resulting sample contains the index values.
        Multi-index is NOT supported.
    choosers: pandas.DataFrame or Series
        Choosers to sample alternatives for.
    sample_size: int, optional, default None
        Defines the number of alternatives to sample for each chooser.
    capacities: pandas.Series, optional, default None
        Specifies capacities for alternatives. If omitted, each alternative
        assumed to have a capacity of 1.

    Returns:
    --------
    sample: numpy array
        Contains the index values of the sampled alternatives,
        with shape (num_choosers, sample_size). If there is not enough capacity,
        the shape will be (alternative total capacity, 1).

    sample_size: int
        Number of samples for each agent (i.e. columns).
        Adjusted to acount for capacities where necessary.
        Returns -1 if there is not enough capacity to support all choosers.

    """
    num_choosers = len(choosers)

    # ensure we have 1 row for each unit of capacity
    if capacities is None:
        num_alts = len(alternatives)
        capacities = pd.Series(np.ones(num_alts), index=alternatives.index)
    else:
        idx_explode = alternatives.index.repeat(capacities)
        alternatives = alternatives.reindex(idx_explode)
        num_alts = len(alternatives)

    # make sure we have enough
    if num_alts <= num_choosers:
        # if not enough, just return all alternatives
        return alternatives.index.values.reshape(num_alts, 1), -1

    # make the sample
    sample_size = min(sample_size, num_alts // num_choosers)
    num_samples = num_choosers * sample_size
    sample = np.random.choice(alternatives.index.values, size=num_samples, replace=False)
    sample = sample.reshape(num_choosers, sample_size)

    return sample, sample_size


def link_sample(choosers, alternatives, sample, sample_size):
    """
    Links sampled alternative IDs with chooser and alternative
    attributes. Column names that exist in both choosers and alternatives
    are suffixed with `_chooser' for columns from the chooser table
    and `_alt` for columns from the alternatives table.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    sample: numpy array
        Sampled alternative IDs.
    sample_size: int
        Number of samples per agent.

    Returns:
    --------
    interaction_data: pandas.DataFrame
        Data frame with 1 row for each chooser and sampled alternative. Index is a
        multi-index with level 0 containing the chooser IDs and level 1 containing
        the alternative IDs.

    """
    # align samples
    sampled_alts = alternatives.reindex(sample.ravel())
    sampled_alts['chooser_id'] = choosers.index.repeat(sample_size).values
    sampled_alts['alternative_id'] = sampled_alts.index.values

    # join
    interaction_data = pd.merge(
        choosers,
        sampled_alts,
        left_index=True,
        right_on='chooser_id',
        suffixes=('_chooser', '_alt')

    )
    interaction_data.set_index(['chooser_id', 'alternative_id'], inplace=True, drop=True)

    return interaction_data


def get_interaction_data_for_estimation(choosers, alternatives, observed,
                                        sample_size=None, sampling_weights=None,
                                        max_sampling_iter=100):
    """
    Returns an interaction data necessary for estimation.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    observed: pandas.Series
        Observed choices of choosers
    sample_size: int, optional, default None
        If provided defines the number of alternatives to sample for each chooser.
        If None, the same alternatives will be provided to each chooser.
    sampling_weights: pandas.Series, optional, default None
        Weights used to influence sampling of alternatives
    max_sampling_iter: int, optional, default 100
        Number of iterations used to attempt to remove row-wise duplicates.

    """
    if not alternatives.index.is_unique:
        raise ValueError('Alternatives have duplicate IDs in index')

    if not choosers.index.is_unique:
        raise ValueError('Choosers have duplicate IDs in index')

    sample, sample_size, chosen = sample_alts_2d(
        alternatives, choosers, sample_size, observed, sampling_weights, max_sampling_iter)

    interaction_data = link_sample(choosers, alternatives, sample, sample_size)
    return interaction_data, sample_size, chosen


def get_interaction_data_for_predict(choosers, alternatives, sample_size=None,
                                     compete=True, capacities=None,
                                     sampling_weights=None, max_sampling_iter=100):
    """
    Returns an interaction dataset necessary for prediction.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    sample_size: int, optional default None
        If provided defines the number of alternatives to sample for each chooser.
        If None, the same alternatives will be provided to each chooser.
    compete: bool, optional, default True
        If True, sampled alternatives and choices can be shared across multiple choosers.
        If False, this will generate a non-overlapping choiceset.
    capacities: pandas.Series, optional, default None
        Only applicable if `compete` is False.
        Specifies alternative capacities for sampling.
    sampling_weights: pandas.Series, optional, default None
        Weights used to influence sampling of alternatives
    max_sampling_iter: int, optional, default 100
        Number of iterations used to attempt to remove row-wise duplicates.

    Returns:
    -------
    interaction_data: pandas.DataFrame
        Resulting interaction dataset. Contains a multi-index with chooser
        IDs as the 1st level and alternative IDs as the 2nd.
    sample_size: int
        Number of samples per chooser.
    num_choosers: int
        Number of choosers in the interaction dataset.

    """
    if not alternatives.index.is_unique:
        raise ValueError('Alternatives have duplicate IDs in index')

    if not choosers.index.is_unique:
        raise ValueError('Choosers have duplicate IDs in index')

    num_choosers = len(choosers)

    if compete:
        # use for most cases
        sample, sample_size = sample_alts_2d(
            alternatives, choosers, sample_size,
            sampling_weights=sampling_weights, max_iter=max_sampling_iter)
    else:
        # used for last iteration of contrained choice
        sample, sample_size = sample_2d_no_compete(
            alternatives, choosers, sample_size, capacities)

        if sample_size == -1:
            # not enough capacity, randomly select choosers
            sample_size = 1
            num_choosers = len(sample)
            c_idx = np.random.choice(choosers.index.values, size=num_choosers, replace=False)
            choosers = choosers.loc[c_idx]

    interaction_data = link_sample(choosers, alternatives, sample, sample_size)
    return interaction_data, sample_size, num_choosers


def get_sampling_weights(alternatives, sampling_weights_col=None):
    """
    Returns sampling weights for the alternative set.

    Parameters:
    -----------
    alternatives: pandas.DataFrame
        Data frame of alternatives to sample from.
    sampling_weights_col: str
        Name of column containing sampling weights.

    Returns:
    --------
    pandas.Series, None if `sampling_weights_col` is None

    """
    if sampling_weights_col is None:
        return None
    return alternatives[sampling_weights_col]


class DefaultSampler(object):
    """
    Used to generate interaction datasets for individual choice predictions. Intended to
    be used as a callback to pass to other choice methods.

    Parameters:
    ----------
    sample_size: int, optional, default None
        Determines the number of alternatives to sample for each choosers, if
        None all alterantives will be used.
    sampling_weights_col: str, optional default None
        Defines the sampling weight column on the alterantives to influence
        sampled alternatives.
    max_iter: int, optional, default 100
        Maximum number of iterations to employ when removing row-level duplicates.

    """
    def __init__(self, sample_size=None, sampling_weights_col=None, max_iter=100):
        self.sample_size = sample_size
        self.sampling_weights_col = sampling_weights_col
        self.max_iter = max_iter

    def __call__(self, choosers, alternatives, *args, **kwargs):
        """
        Returns an interaction datset for the provided choosers and alterantives.

        Parameters:
        -----------
        choosers: pandas.DataFrame
            Agents making choices.
        alternatives: pandas.DataFrame
            Rows to choose from.

        Returns:
        --------
        interaction_data: pandas.DataFrame
            Resulting interaction dataset. Contains a multi-index with chooser
            IDs as the 1st level and alternative IDs as the 2nd.
        sample_size: int
            Number of samples per chooser.
        num_choosers: int
            Number of choosers in the interaction dataset.

        """
        # handle sampling weights
        sampling_weights = get_sampling_weights(alternatives, self.sampling_weights_col)

        return get_interaction_data_for_predict(
            choosers,
            alternatives,
            sample_size=self.sample_size,
            sampling_weights=sampling_weights
        )


def get_sample_for_segment(choosers, alternatives, segment, segment_col,
                           sample_size=None, sampling_weights_col=None,
                           sampling_within_percent=1):
    """
    Generates samples within a single segment.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    segment: value
        The value used to define the current chooser set.
    segment_col: str
        Column name on both choosers and alternatives
        defining the segmentation.
    sample_size: int, optional, default None
        The number of alterantives to sample per chooser. If None,
        all alterantives will be used.
    sampling_weight_col: str, optional default None
        Column used on alterantives to influence sampling.
    sampling_within_percent: float, optional, default 1
        Defines the share of sampled alternatives that should be obtained
        from within the same segment as the choosers.

    Returns:
    --------
    interaction_data: pandas.DataFrame
        Resulting interaction dataset. Contains a multi-index with chooser
        IDs as the 1st level and alternative IDs as the 2nd.
    sample_size: int
        Number of samples per chooser.
    num_choosers: int
        Number of choosers in the interaction dataset.

    """

    # get choosers and alternatives in the segment
    curr_choosers = choosers[choosers[segment_col] == segment]
    num_choosers = len(curr_choosers)
    alt_is_in = alternatives[segment_col] == segment
    curr_alts = alternatives[alt_is_in]

    # define the sampling method, in the future this might be an argument
    sampler_2d = sample_alts_2d

    # sample alternative IDs
    if sample_size is None or sampling_within_percent >= 1:
        # sample exlusively within the segment
        sample, size = sampler_2d(
            curr_alts, curr_choosers, sample_size,
            sampling_weights=get_sampling_weights(curr_alts, sampling_weights_col))
    else:
        # first sample within
        sample_in, sample_in_size = sampler_2d(
            curr_alts,
            curr_choosers,
            int(math.ceil(sample_size * sampling_within_percent)),
            sampling_weights=get_sampling_weights(curr_alts, sampling_weights_col)
        )

        # sample outside
        out_alts = alternatives[~alt_is_in]
        sample_out, sample_out_size = sampler_2d(
            out_alts,
            curr_choosers,
            sample_size - sample_in_size,
            sampling_weights=get_sampling_weights(out_alts, sampling_weights_col)
        )

        # combine samples
        sample = np.hstack([sample_in, sample_out])
        size = sample.shape[1]

    # link sample with chooser and alternative attributes
    data = link_sample(curr_choosers, alternatives, sample, size)
    return data, size, num_choosers


#################################
# INDIVIDUAL CHOICE / PREDICTION
#################################


def individual_choice(choosers, alternatives,
                      interaction_callback, weights_callback, verbose=False):
    """
    Makes individual-level choices between a set of choosers and alterantives
    based on a provided weighting scheme.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    interaction_callback: callable(choosers, alternatives)
        Callable that will return an interaction dataset, the sample size used
        and the number of choosers used.
    weights_callback: callable(interaction_data)
        Callable that will return a pandas.Series of choice weights for the
        sampled alternatives.
    verbose: bool, optional, default False
        If True, returns a data frame containing the IDs, weights and probabilities
        of all sampled alternatives.
        If False, None is returned.

    Returns:
    --------
    choices: pandas.DataFrame
        Data frame of choice results. The chooser ID is the index, columns are
    sample: pandas.DataFrame
        Data frame of sample with weights and probabilities.

    """

    # get sampled interaction data from the callback
    interaction_data, sample_size, num_choosers = interaction_callback(choosers, alternatives)
    chooser_idx = interaction_data.index.get_level_values(0).values
    alt_idx = interaction_data.index.get_level_values(1).values

    # return None if no choosers or alterantives
    if num_choosers == 0 or sample_size == 0:
        return None, None

    # get probabilities for the sampled alternatives as 2d numpy array
    w = weights_callback(interaction_data)
    w = w.values.reshape(num_choosers, sample_size)
    probs = w / w.sum(axis=1, keepdims=True)

    # make choices for each agent
    cs = np.cumsum(probs, axis=1)
    r = np.random.rand(num_choosers).reshape(num_choosers, 1)
    chosen_rel_idx = np.argmax(r < cs, axis=1)
    chosen_abs_idx = chosen_rel_idx + (np.arange(num_choosers) * sample_size)

    curr_choices = pd.DataFrame(
        {
            'alternative_id': alt_idx[chosen_abs_idx],
            'prob': probs.ravel()[chosen_abs_idx],
            'w': w.ravel()[chosen_abs_idx],
        },
        index=pd.Index(chooser_idx[chosen_abs_idx])
    )

    # return the results
    curr_samples = None
    if verbose:
        curr_samples = pd.DataFrame(
            {
                'alternative_id': alt_idx,
                'prob': probs.ravel(),
                'w': w.ravel()
            },
            index=pd.Index(chooser_idx)
        )

    return curr_choices, curr_samples


def segmented_individual_choice(choosers,
                                alternatives,
                                weights_callback,
                                segment_col=None,
                                sample_size=None,
                                sampling_weights_col=None,
                                sampling_within_percent=1,
                                sampling_segments={},
                                verbose=False):
    """
    Makes individual-level choices between a set of choosers and alternatives
    in a segmented manner: there will be independent samples and choices conducted
    for each segment.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    weights_callback: callable(interaction_data)
        Callable that will return a pandas.Series of choice weights for the
        sampled alternatives.
    segment_col: str, optional, default None
        Column used to define segment for choosers and alternatives. If
        None a single choice among will be conducted.
    sample_size: int, optional, default None
        The number of alterantives to sample per chooser. If None,
        all alterantives will be used.
    sampling_weights_col: str, optional default None
        Column used on alternatives to influence sampling.
    sampling_within_percent: float, optional, default 1
        Defines the share of sampled alternatives that should be obtained
        from within the same segment as the choosers.
    sampling_segments: dict, optional, default None
        If provided, allows segment level specifcation of the share of
        alternatives that should be sampled from within the same
        segment as the choosers.
    verbose: bool, optional, default False
        If True, returns a data frame containing the IDs, weights and probabilities
        of all sampled alternatives.
        If False, None is returned.

    Returns:
    --------
    choices: pandas.DataFrame
        Data frame of choice results. The chooser ID is the index, columns are
    sample: pandas.DataFrame
        Data frame of sample with weights and probabilities.

    """
    choices_all = []
    verbose_all = []

    # scott couple of fixes -- todo do something better
    if sampling_segments is None:
        sampling_segments = {}

    if sampling_within_percent is None:
        sampling_within_percent = 1

    # if no segmentation column is provided, just make an individual choice
    if segment_col is None:
        sampler = DefaultSampler(sample_size, sampling_weights_col)
        return individual_choice(
            choosers, alternatives, sampler, weights_callback, verbose)

    # define an interaction callback that will be passed to the individual choice method
    def interaction_callback(choosers, alternatives):
        return get_sample_for_segment(
            choosers,
            alternatives,
            segment,
            segment_col,
            sample_size,
            sampling_weights_col,
            curr_within_pct
        )

    for segment in choosers[segment_col].unique():

        # get the current within segment sampling percentage
        curr_within_pct = sampling_segments.get(segment, sampling_within_percent)

        # make choices for the current segment
        curr_choices, curr_sample = individual_choice(
            choosers, alternatives, interaction_callback,
            weights_callback, verbose)

        if curr_choices is not None:
            choices_all.append(curr_choices)

        if curr_sample is not None:
            verbose_all.append(curr_sample)

    # combine results across segments
    choices_concat = None
    verbose_concat = None

    if len(choices_all) > 0:
        choices_concat = pd.concat(choices_all).reindex(choosers.index)
        if verbose:
            verbose_concat = pd.concat(verbose_all)

    return choices_concat, verbose_concat


class IndividualChoiceCallback(object):
    """
    Wraps calls to `segmented_individual_choice` so it can be used
    as a choice callback with the signature:

        choice_callback(choosers, alternatives, verbose, capacities)

    To run a non-segmented individual choice just pass None as the `segment_col`.

    """
    def __init__(self,
                 weights_callback,
                 segment_col=None,
                 sample_size=None,
                 sampling_weights_col=None,
                 sampling_within_percent=1,
                 sampling_segments={}):

        self.weights_callback = weights_callback
        self.segment_col = segment_col
        self.sample_size = sample_size
        self.sampling_weights_col = sampling_weights_col
        self.sampling_within_percent = sampling_within_percent
        self.sampling_segments = sampling_segments

    def __call__(self, choosers, alternatives, verbose=False, capacities=None):

        return segmented_individual_choice(
            choosers,
            alternatives,
            self.weights_callback,
            self.segment_col,
            self.sample_size,
            self.sampling_weights_col,
            self.sampling_within_percent,
            self.sampling_segments,
            verbose)


#######################
# CONSTRAINED CHOICE
#######################


def lottery_overfill_callback(choices, capacities, choice_col='alternative_id'):
    """
    Handles overfilled alternatives by randomly choosing.

    """
    return segmented_sample_no_replace(capacities, choices, choice_col)


def choiceprobs_overfill_callback(choices, capacities, choice_col='alternative_id'):
    """
    Handles overfilled alternatives by preferring agents
    whose choices have higher probabilities

    """
    return segmented_sample_no_replace(capacities, choices, choice_col, 'prob')


def constrained_choice(choosers,
                       alternatives,
                       choice_callback,
                       overfilled_callback=lottery_overfill_callback,
                       capacity=None,
                       max_iterations=50,
                       verbose=False):
    """
    Generic implementation of constrained choice framework.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    choice_callback: func or callable class
        Callable used to make choices for a given iteration.
        The callback should accept the arguments:
            - choosers:
            - alternatives:
            - verbose:
            - capacities:
        The callback should return the alternative IDs of the choices and
        a verbose result.
    overfilled_callback: func or callable classs, optional, default lottery_choice
        Use to determine which choosers keep their choice in the event that that an
        alternative's capacity is exceeded.
        The callback should accept the arguments: `choices`, `capacities`, `choice_col`.
        The callback should return the indexes of the choosers who get to keep their choice.
        By default this is random.
    capacity: str or pd.Series, optional, default None
        Column on the alternatives data frame or series to provide capacities.
        If None, all alternatives assumed to have a capacity of 1.
    max_iterations: integer, optional, default 50
        Number of iterations to apply.
    verbose: bool, optional, default False
        If true, an additional data frame is returned containing
        additional information about the choices.

    Returns:
    --------
    choices: pandas.Series
        Series of chosen alternative IDs, indexed to the agents.

    capacity: pandas.Series
        Series containing updated capacities after making choices. Indexed to alternatives.

    samples: pandas.DataFrame
        If verbose is True, return additonal info provided by the choice callback.

    """

    # initialize the choice results w/ null values
    # choices = pd.Series(dtype=alternatives.index.dtype, index=choosers.index)
    # note: previously this would create a series w/ nulls, now these are populated with zeros
    choices = pd.Series(index=choosers.index)

    # get alternative capacities
    capacity = get_capacities(alternatives, capacity)

    # use this hold verbose results at each iteration
    verbose_dfs = []

    for i in range(1, max_iterations + 1):

        # filter out choosers who have already chosen
        curr_choosers = choosers[choices.isnull()]
        num_choosers = len(curr_choosers)
        if num_choosers == 0:
            break

        # filter out unavailable alternatives
        has_cap = capacity > 0
        curr_cap = capacity[has_cap]
        if len(curr_cap) == 0:
            break
        curr_alts = alternatives[has_cap]

        #  choices for the current iteration
        curr_choices, curr_v = choice_callback(curr_choosers, curr_alts, verbose, curr_cap)
        if curr_choices is None:
            # TODO: log this
            break

        if verbose:
            curr_v['iter'] = i
            verbose_dfs.append(curr_v)

        # for over-filled alternatives, determine which agents keep their choices
        chosen_idx = overfilled_callback(curr_choices, curr_cap, 'alternative_id')
        chosen = curr_choices.loc[chosen_idx, 'alternative_id']
        choices.loc[chosen_idx] = chosen

        # update capacities
        capacity -= chosen.value_counts().reindex(capacity.index).fillna(0)

    # return the results
    verbose_all = None
    if verbose:
        verbose_all = pd.concat(verbose_dfs)

    return choices, capacity, verbose_all
