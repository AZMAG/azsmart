"""
Contains various aggregations to support neighborhood and accessibility queries.

"""

import orca

from smartpy_core.wrangling import *
from azsmart.framework import *


@orca.injectable()
def model_variables(price_model, hlcm, elcm, wah_lcm, developer_access_cols):
    """
    Returns a set of the columns/variables used by various models.

    """
    return set().union(*[
        price_model.columns_used(),
        hlcm.columns_used(),
        elcm.columns_used(),
        wah_lcm.columns_used(),
        developer_access_cols
    ])


def parse_access_variable(v):
    """
    Parses the accessibility arguments from a variable anme.

    Should be structued as <variable_name>_<dir>_within<travel_time>_mode.

    For example: `sb_jobs_sector92_to_within20_OpAuto`:
        - variable: sb_jobs_sector92
        - direction: travel towards the zone
        - travel time: 20 minutes
        - mode: OpAuto

    Returns a dictionary of the parsed arguments.

    If the variable name does match the needed pattern a None is returned.

    """
    try:
        if '_within' not in v:
            return None

        # parse into left and right of `_within`
        split1 = v.split('_within')
        if len(split1) != 2:
            return None

        # parse out the variable and direction from the left
        left = split1[0]
        if left.endswith('_to'):
            var_name = left.split('_to')[0]
            to = True
        elif left.endswith('_from'):
            var_name = left.split('_from')[0]
            to = False
        else:
            return None

        # parse out the travel time and mode from the right
        split2 = split1[1].split('_')
        travel_time = int(split2[0])
        travel_mode = split2[1]

        return {
            'variable': var_name,
            'time': travel_time,
            'mode': travel_mode,
            'to': to
        }

    except Exception:
        return None


def organize_variables(model_variables):
    """
    Takes the list of variables to compute and puts them
    into threee groups: zonal, hex_nbrs and access. For
    access this will also parse out the required arguments.

    """
    zonal_vars = []
    hex_nbr_vars = []
    access_vars = {}

    for mv in model_variables:

        if mv.startswith('zonal_'):
            # add to the list of zonal variables to compute
            zonal_vars.append(mv)

        elif mv.endswith('_hex_nbr'):
            # add to the list of hex neightbors to compute
            hex_nbr_vars.append(mv)

        else:
            # skims accessibility variable -- need to store parameters
            access_args = parse_access_variable(mv)
            if access_args is None:
                continue
            access_vars[mv] = access_args

    return zonal_vars, hex_nbr_vars, access_vars


def get_year_bin(year, year_bins):
    """
    Returns the bin containing the given year. Intended for small lists.

    Parameters:
    -----------
    year: int
        The current simulation year.
    year_bins: list
        List of years.

    Returns:
    --------
    The year bin that contains the provided year.

    """
    year_bins = sorted(year_bins)

    first_bin = year_bins[0]
    if year is None or year <= first_bin:
        return first_bin

    idx = -1
    for y_bin in year_bins:
        if year < y_bin:
            break
        idx += 1

    return year_bins[idx]


def get_skim_col(year, skims, col):
    """
    Pulls the column for the given year and mode.

    """
    years = np.sort([c[-4:] for c in skims.columns if c.startswith(col)])
    curr_year = min(
        np.searchsorted(years, year),
        len(years) - 1
    )
    return skims['{}_{}'.format(col, years[curr_year])]


@orca.column('taz_skims')
def am_lov_time(year, taz_skims):
    return get_skim_col(year, taz_skims.local, 'am_lov_time')


@orca.column('taz_skims')
def md_auto_time(year, taz_skims):
    return get_skim_col(year, taz_skims.local, 'md_auto_time')


@orca.column('taz_skims')
def am_hov_time(year, taz_skims):
    return get_skim_col(year, taz_skims.local, 'am_hov_time')


def nbr_sums(zones, nbrs, nbr_src_col, nbr_nbr_col, agg_df, agg_cols, fk_col, separate=False):
    """
    Computes neighborhood sums.

    TODO: generalize this to support multiple agg types.

    Parameters:
    ----------
    zones: pandas.DataFrame
        Zones to compute. Results will be aligned to this.
    nbrs: pandas.DataFrame
        Defines neighborhood realtionships.
    nbr_src_col: str
        Column in the neighbors table with the target zone.
    nbr_nbr_col: str:
        Column in the neighbors table with the neighboring zone.
    agg_df: pandas.DataFrame
        Data frame containing data to aggregate.
    agg_cols: list of str
        Columns in the aggregation data frame to aggregate.
    fk_col: str
        Name of the column in the aggregation data frame which
        serves as the foreing key to the zones (e.g. section_id, hex_id)
    separate: bool, optional default False
        If True returns a tuple w/ 1st item containing the target zone sums
            and the 2nd item the corresponding neighrbood sums (w/ out the target -- annulus)
        If False a single data frame is returned that sums the target and its neighrborhood.

    Returns:
    --------
    pandas.DataFrame

    """
    # sum by zones
    zone_sums = agg_df.groupby(fk_col)[agg_cols].sum().reindex(zones.index).fillna(0)

    # join to neighbors
    m = pd.merge(
        zone_sums,
        nbrs,
        left_index=True,
        right_on=nbr_nbr_col
    )

    # aggregate back to zones
    nbr_sums_res = m.groupby(nbr_src_col)[agg_cols].sum().reindex(zones.index).fillna(0)
    if not separate:
        return zone_sums + nbr_sums_res
    else:
        return zone_sums, nbr_sums_res


def travel_time_agg(within, impedance, to, zonal_counts, agg_func=sum, *agg_args, **agg_kwargs):
    """
    Computes an aggregation based on a travel time.

    TODO: potentially decay the counts based on the time.

    Parameters:
    -----------
    within: numeric
        Maximum impedance to consider, usually expressed
        in time (minutes).
    impedance: pandas.Series
        Series containing travel time impedances. Should have a multi-index
        with the first level the the origin zone and the second the destination
        zone.
    to: bool
        Indicates the travel time direction.
        True indicates movements towards the zone.
        False indicates movements away from (leaving) the zone.
    zonal_counts: pandas.DataFrame
        Data frame containing zonal opportunities, e.g. number of jobs.
        Should be indexed by zone id (i.e. taz).
    agg_func: str or callable, optional, default sum
        If provided then this will be used to aggregate results. If None then
        counts will be generated for all columns in the data frame.
    *agg_args:
        Optional possitional arguments to pass to agg_func.
    *agg_kwargs:
        Optional keyword arguments to pass to agg_func.

    """

    if to:
        # towards the zone
        # indicates how easy it is for agents in other zones
        # to reach this zone
        join_level = 0
        agg_level = 1
    else:
        # away from the zone
        # indicates how easy it is for agents in the zone to reach
        # opportunities in other zones
        join_level = 1
        agg_level = 0

    is_within = impedance[impedance <= within]
    zone_reidx = zonal_counts.reindex(is_within.index, level=join_level)
    a = zone_reidx.groupby(level=agg_level).agg(agg_func, *agg_args, **agg_kwargs)
    # return a.reindex(zonal_counts.index)
    return a


@orca.step()
def calc_access_variables(access_variables):
    """
    Forces the calculation/re-calculation of zonal
    and accessibility variables.

    TODO: test this further.

    """
    pass


@orca.table(cache=True, cache_scope='iteration')
def access_variables(model_variables, taz_skims, tazes,
                     pers_by_zone, hh_by_zone, jobs_by_zone,
                     buildings_by_zone, enrollment_by_zone):
    """
    Returns a table of calculated accessibility measures needed
    to satify the provided variables.

    Note: this assumes the hex stuff has already been broadcasted, so we're
    not handling this here.

    TODO: this is still pretty darn clunky and cumbersome, find a better
    way to do this.

    """

    # organize variables
    zonal_vars, hex_nbr_vars, access_vars = organize_variables(model_variables)

    # get columns/variables provided by each table
    tabs = {
        'pers_by_zone': pers_by_zone,
        'hh_by_zone': hh_by_zone,
        'jobs_by_zone': jobs_by_zone,
        'buildings_by_zone': buildings_by_zone,
        'enrollment_by_zone': enrollment_by_zone
    }

    def get_table(col):
        """
        Get the table containing the provided column.

        """
        for tab_name, tab in tabs.items():
            if col in tab.columns:
                return tab_name, tab
        return None, None

    # for zonal columns create the needed broadcasts to buildings
    for zv in zonal_vars:
        # grab the series and register a buildings brodcast for it
        tab_name, tab = get_table(zv)
        make_broadcast_injectable(tab_name, 'buildings', zv, 'taz')

    # mapping between skims columns and mode text in the variable name
    modes = {
        'PkSov': 'am_lov_time',
        'OpAuto': 'md_auto_time',
        'Hov': 'am_hov_time'  # this doesn't appear to be in use
    }

    # for rates/densities we need to compute this as the sum of
    # of two accessibility variables -- for now define these here
    #   This is a dictionary of tuples. The key
    #   is the variable name of interest, the 1st value in the
    #   tuple is the numerator, the second the denominator.
    post_agg_defs = {
        # person-level
        'age_mean': ('age', 'persons'),
        'edu_mean': ('education', 'persons'),
        'adults_edu_mean': ('adults_edu', 'res_adults'),

        # household-level
        'avg_hh_size': ('persons', 'res_hh'),
        'pct_income_group1': ('income_group_1', 'res_hh'),
        'pct_income_group2': ('income_group_1', 'res_hh'),
        'pct_income_group3': ('income_group_2', 'res_hh')
    }

    # run access queries
    bad_access_vars = []
    access_results = {}

    # define an inner method to do a single access query
    def run_access_query(base_name, time, mode, to):
        # get the zonal series
        curr_tab_name, curr_tab = get_table('zonal_' + base_name)
        if curr_tab_name is None:
            # if we couldnt find the variable
            bad_access_vars.append(var_name)
            return None

        # run the query
        return travel_time_agg(
            time,
            taz_skims[modes[mode]],
            to,
            curr_tab['zonal_' + base_name]
        )

    # look through the variables
    for var_name, var_args in access_vars.items():
        base_name = var_args['variable']
        time = var_args['time']
        mode = var_args['mode']
        to = var_args['to']

        # peform the query
        if base_name not in post_agg_defs:
            # simpler case a single query
            t = run_access_query(base_name, time, mode, to)
            if t is None:
                continue
        else:
            # run separate queries and then compute
            num_name = post_agg_defs[base_name][0]
            denom_name = post_agg_defs[base_name][1]
            num_q = run_access_query(num_name, time, mode, to)
            denom_q = run_access_query(denom_name, time, mode, to)
            if num_q is None or denom_q is None:
                continue
            t = num_q / denom_q

        # register w/ orca
        inj_name = '___access__{}'.format(var_name)
        orca.add_injectable(inj_name, t)
        make_series_broadcast_injectable(
            inj_name, 'buildings', var_name, 'taz', fill_with=0)

        # temp
        access_results[var_name] = t

    # add some injectables for inspection later
    orca.add_injectable('zonal_vars', zonal_vars)
    orca.add_injectable('hex_vars', hex_nbr_vars)
    orca.add_injectable('access_vars', access_vars)
    orca.add_injectable('bad_access', bad_access_vars)

    return pd.concat(access_results, axis=1).fillna(0)


#########################
# persons
#########################


def pers_post_agg(s):
    """
    Steps to apply after aggregating. Mostly computing rates, etc.

    """

    # compute means
    s['age_mean'] = fill_nulls(s['age'] / s['persons'])
    s['edu_mean'] = fill_nulls(s['education'] / s['persons'])
    s['adults_edu_mean'] = fill_nulls(s['adult_edu'] / s['is_adult'])

    # clean up and format
    # s.drop(['education', 'age'], axis=1, inplace=True)
    s.rename(
        columns={
            'is_adult': 'res_adults',
            'is_kid': 'kids',
            'is_worker': 'workers',
            'adult_edu': 'adults_edu'
        },
        inplace=True
    )
    return s


@orca.table(cache=True, cache_scope='iteration')
def pers_by_zone(persons, tazes):
    """
    Person-level taz aggregations.

    """

    # columns of interest
    agg_cols = ['age', 'education', 'bachelors_degree', 'grad_degree',
                'hs_grad', 'adult_edu', 'persons', 'is_adult', 'is_kid', 'is_worker']
    pers_df = persons.to_frame(['taz'] + agg_cols)
    s = pers_df.groupby('taz')[agg_cols].sum().reindex(tazes.index).fillna(0)
    s = pers_post_agg(s)
    rename_columns(s, prefix='zonal_')
    return s


@orca.table(cache=True, cache_scope='iteration')
def pers_hex_nbrs(persons, hex_1mi, hex_neighbors):
    """
    Person-level hex neighborhood aggregations.

    """

    # columns of interest
    agg_cols = ['age', 'education', 'bachelors_degree', 'grad_degree',
                'hs_grad', 'adult_edu', 'persons', 'is_adult', 'is_kid', 'is_worker']

    # get data frames
    hex_1 = hex_1mi.local
    hex_nbrs = hex_neighbors.local
    pers_df = persons.to_frame(['hex_id'] + agg_cols)

    # get neighbor sums
    s = nbr_sums(
        hex_1, hex_nbrs, 'src_hex_id', 'nbr_hex_id', pers_df, agg_cols, 'hex_id')
    s = pers_post_agg(s)
    rename_columns(s, prefix='', suffix='_hex_nbr')

    return s


# broadcasts
pers_agg_cols = [
    'bachelors_degree',
    'grad_degree',
    'hs_grad',
    'adults_edu',
    'persons',
    'res_adults',
    'kids',
    'age_mean',
    'edu_mean',
    'adults_edu_mean',
    'workers'
]


for c in pers_agg_cols:
    make_broadcast_injectable('pers_hex_nbrs', 'buildings', c + '_hex_nbr', 'hex_id')
    # make_broadcast_injectable('pers_by_zone', 'buildings', 'zonal_' + c, 'taz')
    # make_reindex_injectable('pers_by_zone', 'tazes', 'zonal_' + c)


#########################
# households
#########################


hh_agg_cols = [
    'res_hh',
    'income1',
    'income2',
    'income3',
    'avg_hhsize',
    'pct_hh_income1',
    'pct_hh_income2',
    'pct_hh_income3'
]


@orca.table(cache=True, cache_scope='iteration')
def hh_hex_nbrs(households, hex_1mi, hex_neighbors):
    """
    Household-level hex neighbor aggregations.

    """

    # columns of interest
    agg_cols = [
        'res_hh', 'persons',
        'income1', 'income2', 'income3'
    ]

    # get data frames
    hex_1 = hex_1mi.local
    hex_nbrs = hex_neighbors.local
    hh_df = households.to_frame(['hex_id'] + agg_cols)

    # get nbr sums
    s = nbr_sums(
        hex_1, hex_nbrs, 'src_hex_id', 'nbr_hex_id', hh_df,
        ['res_hh', 'persons', 'income1', 'income2', 'income3'], 'hex_id')

    # means, rates, etc.
    s['avg_hhsize'] = fill_nulls(s['persons'] / s['res_hh'])
    s['pct_hh_income1'] = fill_nulls(s['income1'] / s['res_hh'])
    s['pct_hh_income2'] = fill_nulls(s['income2'] / s['res_hh'])
    s['pct_hh_income3'] = fill_nulls(s['income3'] / s['res_hh'])
    rename_columns(s, prefix='', suffix='_hex_nbr')

    return s


# register the broadcasts
for c in hh_agg_cols:
    make_broadcast_injectable('hh_hex_nbrs', 'buildings', c + '_hex_nbr', 'hex_id')
    # make_broadcast_injectable('hh_by_zone', 'buildings', 'zonal_' + c, 'taz')
    # make_reindex_injectable('hh_by_zone', 'tazes', 'zonal_' + c)


@orca.table(cache=True, cache_scope='iteration')
def hh_by_zone(households, tazes):
    """
    Household-level taz aggregations.

    I'm ommitting some of the calcls since
    they don't seem to be in the needed variables.

    """
    taz_df = tazes.local
    hh_df = households.to_frame([
        'taz', 'raz', 'mpa',
        'res_hh', 'persons',
        'income', 'income1', 'income2', 'income3'
    ])

    # simple aggregation
    simple_agg_cols = [
        'res_hh', 'persons',
        'income1', 'income2', 'income3'
    ]
    simple_agg = hh_df.groupby('taz')[simple_agg_cols].sum()
    simple_agg.rename(columns={
        'income1': 'income_group_1',
        'income2': 'income_group_2',
        'income3': 'income_group_3',
        'persons': 'res_pop_hh'
    }, inplace=True)

    # hierarchy aggregates
    med_inc = hierarchy_aggregate(
        taz_df[['raz', 'mpa']], hh_df, 'income', ['taz', 'raz', 'mpa'], agg_f='median')
    med_inc.name = 'median_income'

    avg_pph = hierarchy_aggregate(
        taz_df[['raz', 'mpa']], hh_df, 'persons', ['taz', 'raz', 'mpa'], agg_f='mean')
    avg_pph.name = 'avg_pph'

    # concat
    final = pd.concat([
        simple_agg,
        med_inc,
        avg_pph
    ], axis=1).fillna(0)
    rename_columns(final, prefix='zonal_')
    return final


#########################
# jobs
#########################


# subset of modeling sectors to compute hex aggregations for
job_sectors_for_hex = {
    'retail': 'retl_sb_jobs_cnt',
    'office': 'off_sb_jobs_cnt',
    'medical': 'med_sb_jobs_cnt',
    'manufacturing': 'manufact_sb_jobs_cnt',
    'warehouse and transport': 'waretrans_sb_jobs_cnt'
}


@orca.table(cache=True, cache_scope='iteration')
def jobs_hex_nbrs(jobs, hex_1mi, hex_neighbors):
    """
    Job-level hex neighbor aggregations.

    Right now this is much sparser than then the zonal agg, attempting
    to keep the # of calcs to what is minimally required. Just keep this to
    overall jobs and site-based jobs by a few modeling sectors.

    """

    hex_1 = hex_1mi.local
    hex_nbrs = hex_neighbors.local
    jobs_df = jobs.to_frame(['hex_id', 'job_class', 'model_sector'])

    is_sb = jobs_df['job_class'] == 'site based'
    jobs_df['sb_jobs_cnt'] = is_sb.astype(int)
    jobs_df['wah_jobs_cnt'] = (jobs_df['job_class'] == 'work at home').astype(int)

    # compute by modeling sectors
    for k, v in job_sectors_for_hex .items():
        jobs_df[v] = (is_sb & (jobs_df['model_sector'] == k)).astype(int)

    agg_cols = ['sb_jobs_cnt', 'wah_jobs_cnt'] + list(job_sectors_for_hex.values())

    # get the neighbor sums
    s = nbr_sums(
        hex_1, hex_nbrs, 'src_hex_id', 'nbr_hex_id', jobs_df, agg_cols, 'hex_id')

    rename_columns(s, suffix='_hex_nbr')
    return s


# broadcasts
for c in ['sb_jobs_cnt', 'wah_jobs_cnt'] + list(job_sectors_for_hex.values()):
    make_broadcast_injectable('jobs_hex_nbrs', 'buildings', c + '_hex_nbr', 'hex_id')


@orca.table(cache=True, cache_scope='iteration')
def jobs_by_zone(jobs, tazes):
    """
    All job related zonal aggregations

    """

    jobs_df = jobs.to_frame(['job_class', 'model_sector', 'mag_naics', 'building_type_name', 'taz', 'raz', 'mpa'])

    # wah jobs
    wah_df = jobs_df[jobs_df.job_class == 'work at home']
    wah_total = wah_df.groupby('taz').size()
    wah_total.name = 'wah_jobs'

    # construction jobs
    constr_df = jobs_df[jobs_df.job_class == 'construction']
    constr_total = constr_df.groupby('taz').size()
    constr_total.name = 'constr_on_constr_jobs'

    # non-site jobs
    nsb_df = jobs_df[jobs_df.job_class == 'non site based']
    nsb_total = nsb_df.groupby('taz').size()
    nsb_total.name = 'nsb_jobs'

    # just site-based jobs
    sb_jobs_df = jobs_df[jobs_df.job_class == 'site based']

    # total sb job counts
    sb_total = sb_jobs_df.groupby('taz').size()
    sb_total.name = 'sb_jobs'

    # sb job pivots
    gen_piv = get_2d_pivot(sb_jobs_df, 'taz', 'model_sector', '', '_jobs')
    sector_piv = get_2d_pivot(sb_jobs_df, 'taz', 'mag_naics', 'sb_jobs_sector')
    bldg_type_piv = get_2d_pivot(sb_jobs_df, 'taz', 'building_type_name', 'sb_jobs_in_')

    # densities
    gen_dens_piv = gen_piv.div(tazes.acres, axis=0).fillna(0)
    rename_columns(gen_dens_piv, prefix='', suffix='_per_acre')

    final = pd.concat([
        pd.DataFrame(sb_total),
        pd.DataFrame(wah_total),
        pd.DataFrame(constr_total),
        pd.DataFrame(nsb_total),
        gen_piv,
        sector_piv,
        bldg_type_piv,
        gen_dens_piv
    ], axis=1).fillna(0)

    final.columns = [c.replace(' ', '_') for c in final.columns]
    rename_columns(final, prefix='zonal_')
    return final

"""
for jobs_taz_col in orca.get_table('jobs_by_zone').columns:
    for curr_tab in ['buildings']:
        make_broadcast_injectable('jobs_by_zone', curr_tab, jobs_taz_col, 'taz')
"""


########################
# buildings
########################


bldg_hex_agg_cols = [
    'residential_units', 'vac_res_units',
    'job_spaces', 'vac_job_spaces']


@orca.table(cache=True, cache_scope='iteration')
def buildings_hex_nbrs(buildings, hex_1mi, hex_neighbors):
    """
    Building-level hex neighbor aggregations.

    Keep this pretty sparse for now. Most building
    variables are zonal/taz.

    """

    hex_1 = hex_1mi.local
    hex_nbrs = hex_neighbors.local
    bldgs_df = buildings.to_frame(['hex_id'] + bldg_hex_agg_cols)

    s = nbr_sums(
        hex_1, hex_nbrs, 'src_hex_id', 'nbr_hex_id', bldgs_df, bldg_hex_agg_cols, 'hex_id')
    rename_columns(s, suffix='_hex_nbr')
    return s


# register the broadcasts
for c in bldg_hex_agg_cols:
    make_broadcast_injectable('buildings_hex_nbrs', 'buildings', c + '_hex_nbr', 'hex_id')


@orca.table(cache=True, cache_scope='iteration')
def buildings_by_zone(buildings, tazes):
    """
    Building-level taz aggregations.

    """

    taz_df = tazes.local
    buildings_df = buildings.to_frame([
        'building_type_name', 'taz', 'raz', 'mpa', 'total_fcv',
        'residential_units', 'vac_res_units', 'residential_sqft',
        'non_residential_sqft', 'sqft_per_job', 'sb_edu_jobs', 'site_based_jobs',
        'vac_job_spaces', 'job_spaces', 'total_sqft', 'building_age'
    ])

    # residential
    res = buildings_df[np.in1d(buildings_df.building_type_name, ('rsf', 'mh', 'rmf'))]

    # residential unit sums
    res_units_piv = get_2d_pivot(res, 'taz', 'building_type_name', 'res_units_', sum_col='residential_units')
    res_units_piv.loc[:, 'res_units_agg'] = res_units_piv.sum(axis=1)
    res_units_piv.loc[:, 'res_units_per_acre'] = res_units_piv.res_units_agg / taz_df.acres

    # vacant residetial unit sums
    vac_res_units_piv = get_2d_pivot(res, 'taz', 'building_type_name', 'vac_res_units_', sum_col='vac_res_units')
    vac_res_units_piv.loc[:, 'vac_res_units'] = vac_res_units_piv.sum(axis=1)

    # residential average zonal values, with scale-up
    res_value_sum = hierarchy_pivot(
        taz_df,
        res,
        'total_fcv',
        ['taz', 'raz', 'mpa'],
        'building_type_name',
        agg_f='sum',
    )
    res_units_sum = hierarchy_pivot(
        taz_df,
        res,
        'residential_units',
        ['taz', 'raz', 'mpa'],
        'building_type_name',
        agg_f='sum',
    )
    avg_res_value = res_value_sum / res_units_sum
    rename_columns(avg_res_value, 'avg_', '_value')

    # non residential
    non_res = buildings_df[np.in1d(buildings_df.building_type_name, ('rsf', 'mh', 'rmf'), invert=True)]

    job_space_piv = get_2d_pivot(non_res, 'taz', 'building_type_name', 'job_spaces_in_', sum_col='job_spaces')
    job_space_piv.loc[:, 'job_spaces'] = job_space_piv.sum(axis=1)

    vac_job_space_piv = get_2d_pivot(
        non_res, 'taz', 'building_type_name', 'vac_job_spaces_in_', sum_col='vac_job_spaces')
    vac_job_space_piv.loc[:, 'vac_job_spaces'] = vac_job_space_piv.sum(axis=1)

    nonres_units_piv = get_2d_pivot(
        non_res, 'taz', 'building_type_name', 'nonres_sqft_in_', sum_col='non_residential_sqft')

    # non-residential average zonal values, with scale-up
    non_res_value_sum = hierarchy_pivot(
        taz_df,
        non_res,
        'total_fcv',
        ['taz', 'raz', 'mpa'],
        'building_type_name',
        agg_f='sum',
    )
    non_res_units_sum = hierarchy_pivot(
        taz_df,
        non_res,
        'non_residential_sqft',
        ['taz', 'raz', 'mpa'],
        'building_type_name',
        agg_f='sum',
    )
    avg_non_res_value = non_res_value_sum / non_res_units_sum
    rename_columns(avg_non_res_value, 'avg_', '_value')

    vac_rate_res_units = fill_nulls(vac_res_units_piv.vac_res_units / res_units_piv.res_units_agg)
    vac_rate_res_units.name = 'res_units_vac_rate'

    vac_rate_res_units_rsf = fill_nulls(vac_res_units_piv.vac_res_units_rsf / res_units_piv.res_units_rsf)
    vac_rate_res_units_rsf.name = 'res_units_rsf_vac_rate'

    taz_bldg_sqft = fill_nulls(buildings_df[['taz', 'residential_sqft', 'non_residential_sqft']].groupby('taz').sum())
    taz_bldg_sqft_per_landarea = fill_nulls(
        (taz_bldg_sqft.residential_sqft + taz_bldg_sqft.non_residential_sqft) / (taz_df.acres))
    taz_bldg_sqft_per_landarea.name = 'bldg_sqft_per_landarea'

    # newer residential units built  in last 5 years
    new_res_units = get_2d_pivot(
    buildings_df.query("building_age <= 5 and building_type_name in ('rsf', 'rmf')"),
        'taz',
        'building_type_name',
        sum_col='residential_units',
        suffix='_res_units_in_last5'
    )
    new_res_units['res_units_in_last5'] = new_res_units.sum(axis=1)

    final = pd.concat([
        res_units_piv, vac_res_units_piv, avg_res_value,
        job_space_piv, vac_job_space_piv, avg_non_res_value, nonres_units_piv,
        pd.DataFrame(vac_rate_res_units), pd.DataFrame(vac_rate_res_units_rsf),
        pd.DataFrame(taz_bldg_sqft_per_landarea),
        pd.DataFrame(buildings_df.groupby('taz')['total_sqft'].sum()),
        new_res_units
    ], axis=1).fillna(0)

    rename_columns(final, prefix='zonal_')
    return final


########################
# school enrollment
########################


@orca.table(cache=True, cache_scope='iteration')
def enrollment_by_zone(parcels, tazes):

    par_df = parcels.to_frame(['taz', 'k12_enrollment', 'posths_enrollment'])
    s = par_df.groupby('taz').sum().reindex(tazes.index).fillna(0)
    s.rename(columns={
        'k12_enrollment': 'zonal_k12',
        'posths_enrollment': 'zonal_posths'
    }, inplace=True)
    return s


################################
# all zones
################################

@orca.table(cache=True, cache_scope='iteration')
def zonal_aggregations(buildings_by_zone, pers_by_zone, hh_by_zone, jobs_by_zone):
    """
    Aggregate the various zonal aggregation tables into a single source.

    This massive (~3k columns, do we really need to do this or can we broadcast
    individual columns to tazes/zones?)

    """
    return pd.concat([
        buildings_by_zone.local,
        pers_by_zone.local,
        hh_by_zone.local,
        jobs_by_zone.local

    ], axis=1)


################################
# plss-level
################################


@orca.table(cache=True, cache_scope='iteration')
def section_nbr_sums(sections, section_neighbors, buildings):
    plss = sections.local
    plss_nbrs = section_neighbors.local
    bldgs = buildings.to_frame(['total_hh', 'residential_units', 'year_built', 'res_units_in_last5', 'section_id'])
    n_sums = nbr_sums(
        plss, plss_nbrs, 'src_section_id', 'nbr_section_id',
        bldgs,
        ['residential_units', 'res_units_in_last5', 'total_hh'],
        'section_id'
    )
    n_sums['occ_rate'] = fill_nulls(n_sums['total_hh'] / n_sums['residential_units'], 0)

    return n_sums
