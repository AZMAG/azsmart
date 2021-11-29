from __future__ import division

import numpy as np
import pandas as pd
import orca

from smartpy_core.wrangling import *
from azsmart.access_vars import nbr_sums


@orca.injectable()
def developer_access_cols():
    """
    Return the list of zonal/access/neighborhood
    columns that need to be computed for developer.

    """
    return set([
        # single family
        # 'persons_to_within15_OpAuto',
        # 'persons_to_within45_OpAuto',
        'persons_to_within30_OpAuto',
        'sb_jobs_to_within23_OpAuto',

        # multi-family
        'persons_to_within20_OpAuto',
        'res_units_rmf_from_within15_OpAuto',
        'vac_res_units_rmf_from_within15_OpAuto',
        'rmf_res_units_in_last5_from_within15_OpAuto',

        # retail
        'persons_to_within15_OpAuto',
        'sb_jobs_to_within15_OpAuto',
        'sb_jobs_in_retl_to_within15_OpAuto',
        'job_spaces_in_retl_to_within15_OpAuto',

        # office
        'workers_to_within20_OpAuto',
        'sb_jobs_in_off_to_within20_OpAuto',
        'sb_jobs_in_off_to_within15_OpAuto',
        'job_spaces_in_off_to_within15_OpAuto',

        # industrial & warehouse
        'workers_to_within20_OpAuto',
        'sb_jobs_in_ind_from_within20_OpAuto',
        'sb_jobs_in_ware_from_within20_OpAuto',
        'sb_jobs_in_ind_from_within15_OpAuto',
        'sb_jobs_in_ware_from_within15_OpAuto',
        'job_spaces_in_ind_from_within15_OpAuto',
        'job_spaces_in_ware_from_within15_OpAuto',

        # medical
        'persons_to_within20_OpAuto',
        'sb_jobs_in_med_from_within20_OpAuto',
        'job_spaces_in_med_from_within20_OpAuto',

        # hotel
        'sb_jobs_in_hot_to_within15_OpAuto',
        'sb_jobs_in_off_to_within15_OpAuto',
    ])


@orca.column('flu_space', cache=True, cache_scope='iteration')
def max_annual_pct(base_year, year, flu_space, taz_development_constraints):
    """
    Defines the maximum % of units that could be build from a project in a given year.

    Note: right now I'm hard-coding these and adding an additional class for extremely
    larger areas/devs.

    """
    f = flu_space.to_frame(['building_type', 'mag_lu', 'mpa', 'residential_units', 'non_residential_sqft', 'taz'])
    s = pd.Series(index=f.index.copy())

    # note: for some reason np reports a warning when doing the categorize,
    #       suppress this for now
    with np.errstate(invalid='ignore'):

        # rsf
        rsf_proj = f.query("building_type in ('rsf', 'mh')")
        rsf_max = categorize(
            rsf_proj['residential_units'],
            # [np.nan, 50, 100, 200, 500, 1000, np.nan],
            # [1, .5, .29, .20, .15, .01]
            [np.nan, 50, 100, 200, 500, 5000, np.nan],
            [1, .5, .29, .20, .15, .05]
        ).astype(float)

        # for rual/wildcat areas reduce so these grow slower
        not_wildcat = ['PH', 'CH', 'GI', 'ME', 'PE', 'SC', 'QC']
        is_rural = (rsf_proj['mag_lu'].isin([110, 120])) & (~rsf_proj['mpa'].isin(not_wildcat))
        rsf_max.loc[is_rural] = .01

        # for casa grande -- take all the densities down
        is_cg = f['mpa'] == 'CG'
        rsf_max.loc[is_cg] *= 0.75

        # remove capacities for areas with taz constraints
        taz_constrain_start = broadcast(taz_development_constraints['start_year'], rsf_proj['taz']).fillna(base_year)
        to_remove = taz_constrain_start > year
        rsf_max.loc[to_remove] = 0

        s.loc[rsf_max.index] = rsf_max

        # rmf
        rmf_proj = f.query("building_type == 'rmf'")
        rmf_max = categorize(
            rmf_proj['residential_units'],
            [np.nan, 100, 200, 500, 1000, np.nan],
            [1, .75, .5, .35, .15]
        ).astype(float)
        s.loc[rmf_max.index] = rmf_max

        # non-residential
        # TODO: potentiallly segment these further
        com_proj = f.query('non_residential_sqft > 0')
        com_max = categorize(
            com_proj['non_residential_sqft'],
            [np.nan, 100000, 500000, 1000000, 5000000, np.nan],
            [1, .5, .25, .10, .01]
        ).astype(float)
        s.loc[com_max.index] = com_max

    return s


@orca.column('flu_space')
def remaining_units(flu_space):
    """
    Remaining units that can be built.

    """
    return (flu_space['total_units'] - flu_space['built_units']).clip(0)


@orca.column('flu_space')
def max_annual_units(year, flu_space):
    """
    Max units a project can build in a year.

    """
    remain = flu_space['remaining_units']
    max_pct = flu_space['max_annual_pct']
    return stochastic_round(remain * max_pct)


@orca.column('flu_space')
def remaining_land_area(flu_space):
    """
    Estimation of the remaining land area available based
    on the % of units built.

    """
    pct_done = fill_nulls(flu_space['built_units'] / flu_space['total_units'], 1)
    bad = (pct_done < 0) | (pct_done > 1)
    pct_done[bad] = 1
    return (1 - pct_done) * flu_space['land_area']



@orca.column('flu_space', cache=True, cache_scope='iteration')
def rsf_score(year, flu_space, sections, section_neighbors, buildings,
              access_variables):
    """
    Scores/weights for single family development.

    """

    # get projects
    projs = flu_space.to_frame(['building_type', 'project_type',
                                'total_units', 'built_units', 'permitted', 'section_id', 'taz', 'mpa'])
    sf_projs = projs.query("building_type in ('rsf')").copy()  # do we want to add mobile home?

    # the number of projs in each section
    #sf_projs['num_section_projs'] = sf_projs.groupby(
    #    'section_id').size().reindex(sf_projs.index).fillna(0)

    # get section and neighborhood sums
    plss = sections.local
    plss_nbrs = section_neighbors.local
    bldgs = buildings.to_frame(['total_hh', 'residential_units', 'year_built', 'sf_res_units_in_last5', 'section_id'])
    s_sums, n_sums = nbr_sums(
        plss, plss_nbrs, 'src_section_id', 'nbr_section_id',
        bldgs,
        ['residential_units', 'sf_res_units_in_last5', 'total_hh'],
        'section_id',
        separate=True
    )
    s_sums['occ_rate'] = fill_nulls(s_sums['total_hh'] / s_sums['residential_units'], 0)
    n_sums['occ_rate'] = fill_nulls(n_sums['total_hh'] / n_sums['residential_units'], 0)

    # broadcast these to the projects
    s_sums = broadcast(s_sums, sf_projs['section_id'])
    n_sums = broadcast(n_sums, sf_projs['section_id'])

    # convert to rankings
    s_ranks = s_sums.rank(pct=True)
    n_ranks = n_sums.rank(pct=True)

    #  NOT DOING THIS RIGHT NOW, but it might make sense to?  -- discount state land
    # sf_projs['state_trust_red'] = (sf_projs['exlu_dev_state'] == 'State Trust Developable').astype(int)

    sf_projs['n_occ_rank'] = n_ranks['occ_rate']
    sf_projs['n_new_rank'] = n_ranks['sf_res_units_in_last5']
    sf_projs['n_hh_rank'] = n_ranks['total_hh']

    sf_projs['s_occ_rank'] = s_ranks['occ_rate']
    sf_projs['s_new_rank'] = s_ranks['sf_res_units_in_last5']
    sf_projs['s_hh_rank'] = s_ranks['total_hh']

    if year <= 2024:
        # in the 1st 5 years, closely follow recent development trends
        sf_projs['score'] = \
            (3 * sf_projs['s_new_rank']) + \
            (2 * sf_projs['s_hh_rank']) + \
            (2 * sf_projs['n_new_rank']) + \
            sf_projs['s_hh_rank']

    else:
        # after 2025 consider recent development but also sub-regional access
        # pers_win = 'persons_to_within15_OpAuto'
        pers_win = 'persons_to_within30_OpAuto'
        jobs_win = 'sb_jobs_to_within23_OpAuto'

        # broadcast access variables
        av = access_variables.local[[pers_win, jobs_win]]
        av = broadcast(av, sf_projs['taz']).fillna(0)

        # get an overall access score
        access_score = (av[pers_win] + (0.5 * av[jobs_win])).rank(pct=True)
        growth_score = (n_ranks['sf_res_units_in_last5'] + s_ranks['sf_res_units_in_last5']).rank(pct=True)

        # combine w/ recent growth
        growth_w = .75

        if year > 2030:
            growth_w = .65
        if year > 2040:
            growth_w = 0.5

        sf_projs['score'] = access_score + (growth_score * growth_w)

        is_go = sf_projs['mpa'] == 'GO'
        sf_projs.loc[is_go, 'score'] *= 1.25

    # let's try and normalize based on the # of projects
    # todo: should we do this after computing probabilities?
    # sf_projs['score'] /= sf_projs['num_section_projs']

    return sf_projs['score'].reindex(projs.index)


@orca.column('flu_space', cache=True, cache_scope='iteration')
def rmf_score(flu_space, access_variables):
    """
    Scores/weights for multi family development.

    """
    # access cols
    prox_rmf = 'res_units_rmf_from_within15_OpAuto'
    prox_vac_rmf = 'vac_res_units_rmf_from_within15_OpAuto'
    prox_new_rmf = 'rmf_res_units_in_last5_from_within15_OpAuto'
    prox_pers = 'persons_to_within20_OpAuto'

    # get projects
    projs = flu_space.to_frame(['building_type', 'taz', 'freeway_dist', 'lr_dist'])
    rmf_projs = projs.query("building_type in ('rmf')").copy()

    # broadcast access variables
    av = access_variables.local[[prox_rmf, prox_vac_rmf, prox_pers, prox_new_rmf]]
    av = broadcast(av, rmf_projs['taz']).fillna(0)
    av['rmf_occ'] = 1 - fill_nulls(av[prox_vac_rmf] / av[prox_rmf])

    # create the score
    occ_rank = av['rmf_occ'].rank(pct=True)
    pers_rank = av[prox_pers].rank(pct=True)
    new_rmf_rank = av[prox_new_rmf].rank(pct=True)

    rmf_projs['lr_score'] = 0
    rmf_projs.loc[rmf_projs['lr_dist'] <= 5280, 'lr_score'] = .25
    rmf_projs.loc[rmf_projs['lr_dist'] <= 2640, 'lr_score'] = .5
    rmf_projs.loc[rmf_projs['lr_dist'] <= 1320, 'lr_score'] = 1

    rmf_projs['fwy_score'] = 0
    rmf_projs.loc[rmf_projs['freeway_dist'] <= 5280 * 3, 'fwy_score'] = .65
    rmf_projs.loc[rmf_projs['freeway_dist'] <= 5280 * 2, 'fwy_score'] = .75
    rmf_projs.loc[rmf_projs['freeway_dist'] <= 5280 * 1, 'fwy_score'] = 1

    rmf_projs['score'] = \
        new_rmf_rank + \
        pers_rank +    \
        occ_rank +     \
        rmf_projs['lr_score'] + \
        rmf_projs['fwy_score']

    return rmf_projs['score'].reindex(projs.index)


@orca.column('flu_space', cache=True, cache_scope='iteration')
def retail_score(flu_space, access_variables):
    """
    Scores/weights for retail development.

    """
    # access columns
    prox_pers = 'persons_to_within15_OpAuto'
    prox_jobs = 'sb_jobs_to_within15_OpAuto'
    prox_retail_jobs = 'sb_jobs_in_retl_to_within15_OpAuto'
    prox_retail_space = 'job_spaces_in_retl_to_within15_OpAuto'

    # get projects
    projs = flu_space.to_frame(['building_type', 'taz', 'freeway_dist'])
    retail_projs = projs.query("building_type in ('retl')").copy()

    # broadcast access variables
    av = access_variables.local[[
        prox_pers,
        prox_jobs,
        prox_retail_jobs,
        prox_retail_space
    ]]
    av = broadcast(av, retail_projs['taz']).fillna(0)
    av['occ'] = fill_nulls(av[prox_retail_jobs] / av[prox_retail_space])

    # create the score
    demand = (av[prox_pers] + av[prox_jobs]) - av[prox_retail_jobs]
    demand_ratio = fill_nulls(demand / av[prox_retail_jobs])
    ratio_rank = demand_ratio.rank(pct=True)
    fwy_inv_dist = 1 / np.power(retail_projs['freeway_dist'] + 1, 2)
    fwy_rank = fwy_inv_dist.rank(pct=True)
    score = ratio_rank + (fwy_rank / 4)

    no_demand = (av[prox_pers] + av[prox_jobs]) < 500
    score[no_demand] = 0

    return score.reindex(projs.index)


@orca.column('flu_space', cache=True, cache_scope='iteration')
def office_score(flu_space, access_variables):
    """
    Scores/weights for office development.

    """
    # access columns
    prox_workers = 'workers_to_within20_OpAuto'
    prox_off_jobs_20 = 'sb_jobs_in_off_to_within20_OpAuto'
    prox_off_jobs = 'sb_jobs_in_off_to_within15_OpAuto'
    prox_off_job_spaces = 'job_spaces_in_off_to_within15_OpAuto'

    # get projects
    projs = flu_space.to_frame(['building_type', 'taz', 'freeway_dist'])
    off_projs = projs.query("building_type in ('off')").copy()

    # broadcast access variables
    av = access_variables.local[[
        prox_workers,
        prox_off_jobs_20,
        prox_off_jobs,
        prox_off_job_spaces
    ]]
    av = broadcast(av, off_projs['taz']).fillna(0)
    av['occ'] = fill_nulls(av[prox_off_jobs] / av[prox_off_job_spaces])

    # freeway influence distance
    fwy_influence_factor = 0.3
    fwy_inv_dist = 1 / np.power(off_projs['freeway_dist'] + 1, 2)
    fwy_rank = fwy_inv_dist.rank(pct=True)
    fwy_weight = fwy_rank * fwy_influence_factor

    # compute an office jobs to workers balance shortage rank
    shortage = av[prox_workers] / (1 + av[prox_off_jobs_20])
    shortage_rank = shortage.rank(pct=True)

    # compute an agglomeration rank
    occ_rank = (av[prox_off_jobs] / av[prox_off_job_spaces]).rank(pct=True)
    local_rank = av[prox_off_jobs].rank(pct=True)
    agglomeration_rank = (occ_rank + (local_rank / 2)).rank(pct=True)

    # compute the final score as the max of aggolomeration and shortage
    is_aglomerate = agglomeration_rank > shortage_rank
    agglomeration_rank[~is_aglomerate] = 0
    shortage_rank[is_aglomerate] = 0
    combined = (agglomeration_rank + shortage_rank).rank(pct=True)

    score = combined + fwy_weight
    return score.reindex(projs.index)


@orca.column('flu_space', cache=True, cache_scope='iteration')
def ind_score(flu_space, access_variables):
    """
    Scores/weights for industrial development.

    """
    # access columns
    prox_workers = 'workers_to_within20_OpAuto'
    prox_ind_jobs_20 = 'sb_jobs_in_ind_from_within20_OpAuto'
    prox_ware_jobs_20 = 'sb_jobs_in_ware_from_within20_OpAuto'
    prox_ind_jobs = 'sb_jobs_in_ind_from_within15_OpAuto'
    prox_ware_jobs = 'sb_jobs_in_ware_from_within15_OpAuto'
    prox_ind_job_spaces = 'job_spaces_in_ind_from_within15_OpAuto'
    prox_ware_job_spaces = 'job_spaces_in_ware_from_within15_OpAuto'

    # get projects
    projs = flu_space.to_frame(['building_type', 'taz', 'freeway_dist', 'rail_dist', 'airport_dist'])
    ind_projs = projs.query("building_type in ('ind', 'ware')").copy()

    # broadcast access variables
    av = access_variables.local[[
        prox_workers,
        prox_ind_jobs_20,
        prox_ware_jobs_20,
        prox_ind_jobs,
        prox_ware_jobs,
        prox_ind_job_spaces,
        prox_ware_job_spaces,
    ]]
    av = broadcast(av, ind_projs['taz']).fillna(0)

    # combine industrial and warehouse together and get an occupancy
    prox_jobs_20 = av[prox_ind_jobs_20] + av[prox_ware_jobs_20]
    prox_jobs = av[prox_ind_jobs] + av[prox_ware_jobs]
    prox_job_spaces = av[prox_ind_job_spaces] + av[prox_ware_job_spaces]

    # get freeway distance influence
    fwy_influence_factor = 0.3
    fwy_inv_dist = 1 / np.power(ind_projs['freeway_dist'] + 1, 2)
    fwy_rank = fwy_inv_dist.rank(pct=True)
    fwy_weight = fwy_rank * fwy_influence_factor

    # get rail distance influence
    rail_influence_factor = 0.25
    rail_inv_dist = 1 / np.power(ind_projs['rail_dist'] + 1, 2)
    rail_rank = rail_inv_dist.rank(pct=True)
    rail_weight = rail_rank * rail_influence_factor

    # get airport distance influence
    air_influence = 0.3
    air_inv_dist = 1 / np.power(ind_projs['airport_dist'] + 1, 2)
    air_rank = air_inv_dist.rank(pct=True)
    air_weight = air_rank * air_influence

    # compute our jobs balance shortage
    shortage = av[prox_workers] / (1 + prox_jobs_20)
    shortage_rank = shortage.rank(pct=True)

    # compute an agglomeration rank
    occ_rank = (prox_jobs / prox_job_spaces).rank(pct=True)
    local_rank = prox_jobs.rank(pct=True)
    agglomeration_rank = (occ_rank + (local_rank / 2)).rank(pct=True)

    # compute the final score as the max of aggolomeration and shortage
    is_aglomerate = agglomeration_rank > shortage_rank
    agglomeration_rank[~is_aglomerate] = 0
    shortage_rank[is_aglomerate] = 0
    combined = (agglomeration_rank + shortage_rank).rank(pct=True)

    score = combined + fwy_weight + rail_weight + air_weight
    return score.reindex(projs.index)


@orca.column('flu_space', cache=True, cache_scope='iteration')
def med_score(flu_space, access_variables, year):
    """
    Scoring/weighting for medical project.

    Note: for this, let's let them also go into office/retail?

    """
    # access columns
    prox_pers = 'persons_to_within20_OpAuto'
    prox_med_jobs = 'sb_jobs_in_med_from_within20_OpAuto'
    prox_med_job_spaces = 'job_spaces_in_med_from_within20_OpAuto'

    # get projects
    projs = flu_space.to_frame(['building_type', 'taz', 'freeway_dist'])
    med_projs = projs.query("building_type in ('med', 'off', 'retl', 'gq')").copy()

    # broadcast access variables
    av = access_variables.local[[
        prox_pers,
        prox_med_jobs,
        prox_med_job_spaces
    ]]
    av = broadcast(av, med_projs['taz']).fillna(0)
    occ = fill_nulls(av[prox_med_jobs] / av[prox_med_job_spaces])

    # compute our score
    occupancy_rank = occ.rank(pct=True)
    agglomeration_rank = av[prox_med_jobs].rank(pct=True)
    pop_rank = av[prox_pers].rank(pct=True)

    fwy_inv_dist = 1 / np.power(med_projs['freeway_dist'] + 1, 2)
    fwy_rank = fwy_inv_dist.rank(pct=True)

    # if the project is a medical project give it a slight bump?
    is_med = (med_projs['building_type'] == 'med').astype(int)
    if year < 2026:

        score = \
            fill_nulls(agglomeration_rank) +     \
            fill_nulls(occupancy_rank / 2.0) + \
            fill_nulls(pop_rank) +       \
            fill_nulls(fwy_rank / 4.0) +       \
            (is_med) #orig divided by 5, #orig divided by 2 for pop_rank
    else:

        score = \
            fill_nulls(agglomeration_rank / 2.0) +     \
            fill_nulls(occupancy_rank / 2.0) + \
            fill_nulls(pop_rank) +       \
            fill_nulls(fwy_rank / 4.0) +       \
            (is_med) #orig divided by 5, #orig divided by 2 for pop_rank, orig agglomeration_rank no division

    return score.reindex(projs.index)


@orca.column('flu_space', cache=True, cache_scope='iteration')
def hotel_score(flu_space, access_variables, year):
    """
    Scoring for hotel projects.

    """
    # access cols
    prox_hot_jobs = 'sb_jobs_in_hot_to_within15_OpAuto'
    prox_off_jobs = 'sb_jobs_in_off_to_within15_OpAuto'

    # get projects
    projs = flu_space.to_frame(['building_type', 'taz', 'freeway_dist'])
    hotel_projs = projs.query("building_type in ('hot')").copy()

    # broadcast access variables
    av = access_variables.local[[
        prox_hot_jobs,
        prox_off_jobs
    ]]
    av = broadcast(av, hotel_projs['taz']).fillna(0)

    fwy_inv_dist = 1 / np.power(hotel_projs['freeway_dist'] + 1, 2)
    fwy_rank = fwy_inv_dist.rank(pct=True)
    hot_rank = av[prox_hot_jobs].rank(pct=True)
    off_rank = av[prox_off_jobs].rank(pct=True)

    score = hot_rank + (off_rank * .75) + (fwy_rank * 0.75)
    return score.reindex(projs.index)


@orca.table()
def job_space_demand(year, jobs, buildings):
    """
    Computes demand for new non-residential space. This
    is a data frame with rows for builidng types and
    columns for counties.

    Note:
        these are job spaces not non-residential sqft

    """
    j = jobs.to_frame(['job_class', 'building_id', 'src_building_id', 'mag_naics'])
    # TODO: change this to also look at the year added field also
    #       when transition is up so we don't built space
    #       for re-locators
    new = j.query("building_id == -1 and job_class == 'site based'").copy()
    new['county'] = broadcast(buildings['county'], new['src_building_id'])
    new['building_type_name'] = broadcast(buildings['building_type_name'], new['src_building_id'])

    # Change the demand of off and retl building spaces by medical model sector to med building spaces
    new.loc[(((new.building_type_name == 'off') | (new.building_type_name == 'retl')) & (new.mag_naics == '62')), 'building_type_name'] = 'med'

    return get_2d_pivot(new, 'building_type_name', 'county')


@orca.table()
def residential_demand(year, county_du_controls):
    """
    Demand for residential units. Building types
    are rows, counties are columns.

    """
    # need to pull out the controls for the current year and reformat
    c = county_du_controls.local.loc[year]
    return pd.DataFrame([
        {'building_type_name': 'rsf', 'MC': c['mc_rsf'], 'PC': c['pc_rsf']},
        {'building_type_name': 'rmf', 'MC': c['mc_rmf'], 'PC': c['pc_rmf']},
    ]).set_index('building_type_name')


@orca.table()
def msa_job_space_demand(job_space_demand):
    """
    Job space demand aggregated to the MSA.

    """
    df = job_space_demand.local
    return df.fillna(0).sum(axis=1).to_frame('msa')


@orca.table()
def msa_residential_demand(residential_demand):
    """
    Residential demand aggregated to the MSA.

    """
    df = residential_demand.local
    return df.fillna(0).sum(axis=1).to_frame('msa')


def cum_choose(amount, amounts, match_exact=False):
    """
    Choose the top n items in a list that satify a
    cumulative amount. Assumes the amounts have been
    sorted to reflect item preferences.

    amount: int
        Number to choose
    amounts: pandas.Series
        Amounts available
    match_exact: bool, default False
        If True, the amount of the last chosen item
        will be trimmed to match the remaining needed.

    """
    cs = amounts.cumsum()

    if match_exact:
        remaining = amount - cs
        remaining[cs <= amount] = 0
        chosen_amounts = amounts + remaining
        return chosen_amounts[chosen_amounts > 0]
    else:
        return amounts[cs <= amount]


def choose_projects(projs, score_col, amount_col, demand, county=None, top_pct=0.5):
    """
    Choose projects for a given building type and county.

    """

    # get availavle projects
    if  county is None:
        q = "{} > 0 and {} > 0".format(score_col, amount_col)
    else:
        q = "county == '{}' and {} > 0 and {} > 0".format(county, score_col, amount_col)
    curr_projs = projs.query(q)

    # filter out the bottom feeders
    total_available = curr_projs[amount_col].sum()

    if total_available > demand / top_pct:
        amount_to_sample = round(total_available * top_pct)
        s = curr_projs[[score_col, amount_col]].sort_values(score_col, ascending=False)
        choice_set = curr_projs.loc[cum_choose(amount_to_sample, s[amount_col]).index]
    else:
        choice_set = curr_projs.copy()

    # sort based on probabilties
    exp_score = np.exp(choice_set[score_col])
    prob = exp_score / exp_score.sum()
    shuffle = np.random.choice(
        choice_set.index.values, p=prob.values, size=len(choice_set), replace=False)

    # make the choice
    amounts = choice_set.loc[shuffle, amount_col]
    return cum_choose(demand, amounts, True)


@orca.injectable()
def developer_space_types():
    """
    Configuration of space/buildings types we will be developing.

    """
    return [
        {
            'type': 'rsf',
            'score_col': 'rsf_score',
            'is_residential': True,
            'active_weight': 100,
            'permit_weight': 10,
            'rescomp_weight': 500
        },
        {
            'type': 'rmf',
            'score_col': 'rmf_score',
            'is_residential': True,
            'active_weight': 100,
            'permit_weight': 10,
            'rescomp_weight': 500
        },
        {
            'type': 'retl',
            'score_col': 'retail_score',
            'is_residential': False,
            'active_weight': 100,
            'permit_weight': 10,
            'rescomp_weight': 500
        },
        {
            'type': 'off',
            'score_col': 'office_score',
            'is_residential': False,
            'active_weight': 100,
            'permit_weight': 10,
            'rescomp_weight': 500
        },
        {
            'type': 'ind',
            'score_col': 'ind_score',
            'is_residential': False,
            'active_weight': 100,
            'permit_weight': 10,
            'rescomp_weight': 500
        },
        {
            'type': 'ware',
            'score_col': 'ind_score',
            'is_residential': False,
            'active_weight': 100,
            'permit_weight': 10,
            'rescomp_weight': 500
        },
        {
            'type': 'med',
            'score_col': 'med_score',
            'is_residential': False,
            'active_weight': 100,
            'permit_weight': 10,
            'rescomp_weight': 500
        },
        {
            'type': 'hot',
            'score_col': 'hotel_score',
            'is_residential': False,
            'active_weight': 100,
            'permit_weight': 10,
            'rescomp_weight': 500
        }
    ]


@orca.step()
def msa_developer(year,
                  flu_space,
                  buildings,
                  developer_space_types,
                  msa_residential_demand,
                  msa_job_space_demand):
    """
    Step for devloping new built space (adds buildings) w/ controls specified
    at the MSA level.

    """
    run_developer(
        year, flu_space, buildings, developer_space_types,
        msa_residential_demand, msa_job_space_demand, 'msa'
    )


@orca.step()
def developer(year,
              flu_space,
              buildings,
              developer_space_types,
              residential_demand,
              job_space_demand):
    """
    Step for devloping new built space (adds buildings) w/ controls specified
    at the county level.

    """
    run_developer(
        year, flu_space, buildings, developer_space_types,
        residential_demand, job_space_demand, ['MC', 'PC']
    )


def run_developer(year,
                  flu_space,
                  buildings,
                  developer_space_types,
                  residential_demand,
                  job_space_demand,
                  demand_cols):
    """
    Executes the developer.

    """

    # msa vs. county level
    by_county=True
    if not isinstance(demand_cols, list):
        by_county = False
        demand_cols = [demand_cols]

    # combine demand into a single table
    demand = pd.concat([residential_demand.local, job_space_demand.local])
    orca.add_injectable('_demand', demand)

    # get a list of the scoring columns
    score_cols = [c['score_col'] for c in developer_space_types]

    # get flu space/projects
    extra_cols = ['county', 'taz', 'section', 'max_annual_units'] + score_cols
    projs = flu_space.to_frame(flu_space.local_columns + extra_cols)

    # for non-residential projects, convert the max available amount to job spaces
    is_res = projs['building_type'].isin(['rsf', 'rmf', 'mh'])
    projs['amount'] = projs['max_annual_units']
    projs.loc[~is_res, 'amount'] = np.round(fill_nulls(
        projs.loc[~is_res, 'max_annual_units'] / projs.loc[~is_res, 'sqft_per_job']))

    # just keep projects available in the current year
    projs = projs.query('start_year <= {}'.format(year)).copy()

    # in the first few years, let's limit this to active developments
    # note: not enough in Pinal -- so just try weighting the active devs higher
    #if year <= 2020:
    #    projs = projs.query('is_active == 1').copy()

    # choose projects by space type and county
    choices_all = []

    for st in developer_space_types:
        # get space-type props
        curr_st = st['type']
        score_col = st['score_col']
        is_residential = st['is_residential']
        active_w = st['active_weight']
        permit_w = st['permit_weight']
        rescomp_w = st['rescomp_weight']

        # update score based on rescomps, active status, permits
        projs[score_col + '_final'] = \
            projs[score_col] + \
            (projs['is_active'] * active_w) + \
            (projs['permitted'] * permit_w) + \
            (projs['recent_rescomp'] * rescomp_w)

        # let's persistently add both the orginal and final score so we can evaluate later if needed
        flu_space.local.loc[projs.index, '__pre_score_{}'.format(curr_st)] = projs[score_col]
        flu_space.local.loc[projs.index, '__final_score_{}'.format(curr_st)] = projs[score_col + '_final']

        for county in demand_cols:

            # get the demand
            curr_demand = demand.loc[curr_st][county]
            if curr_demand <= 0:
                continue

            # choose projects to build
            if by_county:
                curr_choices = choose_projects(projs, score_col + '_final', 'amount', curr_demand, county)
            else:
                curr_choices = choose_projects(projs, score_col + '_final', 'amount', curr_demand)

            # check if we matched the demand
            if curr_choices.sum() != curr_demand:
                print('')
                print ('****** DEMAND NOT MATCHED!! **************')
                print (county)
                print (curr_st)
                print ('demand: {}'.format(curr_demand))
                print ( 'choice sum: {}'.format(curr_choices.sum()))
                print ('******************************************')
                continue

            if curr_choices.sum() == 0:
                continue

            # remove the chosen space from available amounts
            projs.loc[curr_choices.index, 'amount'] -= curr_choices

            # format choices so we can convert to buildings later
            curr_choices_df = curr_choices.to_frame('amount')
            curr_choices_df['building_type_name'] = curr_st
            curr_choices_df['is_residential'] = is_residential
            choices_all.append(curr_choices_df)

    choices_all = pd.concat(choices_all)

    # convert choices to buildings
    for c in ['sqft_per_job', 'sqft_per_res', 'parcel_id', 'project_id']:
        choices_all[c] = projs.loc[choices_all.index, c]

    choices_all['project_idx'] = choices_all.index.values
    choices_all['year_built'] = year
    choices_all['residential_units'] = 0
    choices_all['residential_sqft'] = 0
    choices_all['non_residential_sqft'] = 0

    is_res = choices_all['is_residential']
    choices_all.loc[is_res, 'residential_units'] = choices_all.loc[is_res, 'amount']
    choices_all.loc[is_res, 'residential_sqft'] = choices_all.loc[is_res, 'amount'] * choices_all.loc[is_res, 'sqft_per_res']
    choices_all.loc[~is_res, 'non_residential_sqft'] = choices_all.loc[~is_res, 'amount'] * choices_all.loc[~is_res, 'sqft_per_job']

    for c in ['transient_hh_in_hh', 'transient_hh_in_hotels',
              'transient_pop_in_hh', 'transient_pop_in_hotels']:
        choices_all[c] = 0

    # not sure what to do with this avg val per unit and total fcv
    # the price model should handle this, but set to 1 so we don't wind up having issues with nulls
    choices_all['total_fcv'] = 1
    choices_all['average_value_per_unit'] = 1

    # append to the buildings table
    old_bldgs = buildings.local
    start_id = old_bldgs.index.max() + 1
    choices_all.reset_index(drop=True, inplace=True)
    choices_all.index = choices_all.index + start_id
    choices_all.index.name = 'building_id'
    choices_all.drop(['amount', 'is_residential', 'sqft_per_res'], inplace=True, axis=1)
    bldgs_new = pd.concat([old_bldgs, choices_all])
    assert bldgs_new.index.is_unique
    orca.add_table('buildings', bldgs_new)

    # update the project status to reflect additional built units
    # do we want to make un-active projects active? for now, No
    tot_unit = choices_all['residential_units'] + choices_all['non_residential_sqft']
    tot_unit_sums = tot_unit.groupby(choices_all['project_idx']).sum()
    flu_space.local.loc[tot_unit_sums.index, 'built_units'] += tot_unit_sums





    # do I want to clear the cache? for now don't, just let the iteration clear it

    # for debugging / checking
    # orca.add_injectable('__projs', projs)
    # orca.add_injectable('__choices', choices_all)

    # TODO: for redevelopment remove existing space
