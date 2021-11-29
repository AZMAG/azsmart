"""
Model/steps for demographics/households.

"""

import numpy as np
import orca

from urbansim.models.transition import *
from smartpy_core.allocation import get_simple_allocation, get_segmented_allocation
from smartpy_core.wrangling import stochastic_round, broadcast
from azsmart.utils.dcm2 import *


##################################
# household population transition
##################################


def run_hh_transition(year, controls_wrap, hh_wrap, pers_wrap, by_county=True):
    """
    Executes the transition.

    """
    controls = controls_wrap.local

    if by_county:
        hh = hh_wrap.to_frame(hh_wrap.local_columns + ['county'])
    else:
        hh = hh_wrap.local

    pers = pers_wrap.local

    # run the transition
    t = TabularTotalsTransition(controls, 'hh_population', 'persons')
    tm = TransitionModel(t)
    linked = {'persons': (pers, 'household_id')}
    updated, added, updated_links = tm.transition(hh, year, linked)

    # flag new households/persons
    # for new households retain the building id they transitioned from
    updated.loc[added, 'year_added'] = year
    updated['src_building_id'] = updated['building_id']
    updated.loc[added, 'building_id'] = -1

    if 'county' in updated.columns:
        updated.drop('county', axis=1, inplace=True)

    # update the environment
    orca.add_table('households', updated)
    orca.add_table('persons', updated_links['persons'])


@orca.step()
def county_hh_transition(year, county_hh_pop_controls, households, persons):
    """
    Transition households using county-level controls

    """
    run_hh_transition(year, county_hh_pop_controls, households, persons, True)


@orca.step()
def msa_hh_transition(year, msa_hh_pop_controls, households, persons):
    """
    Transition households using MSA-level controls

    """
    run_hh_transition(year, msa_hh_pop_controls, households, persons, False)


##################################
# seasonal population transition
##################################


def run_seasonal_transition(year, controls_wrap, seas_hh_wrap, by_county=True):
    """
    Executes a transition model for seasonal households.

    """
    controls = controls_wrap.local
    if by_county:
        seas_hh = seas_hh_wrap.to_frame(seas_hh_wrap.local_columns + ['county'])
    else:
        seas_hh = seas_hh_wrap.local

    # run the transition
    t = TabularTotalsTransition(controls, 'seasonal_population', 'persons')
    tm = TransitionModel(t)
    updated, added, _ = tm.transition(seas_hh, year)

    # flag new households/persons
    updated.loc[added, 'year_added'] = year
    updated['src_building_id'] = updated['building_id']
    updated.loc[added, 'building_id'] = -1

    #if by_county:
        # updated.rename(columns={'county': 'src_county'}, inplace=True)
    if 'county' in updated.columns:
        updated.drop('county', axis=1, inplace=True)

    # update the environment
    orca.add_table('seasonal_households', updated)


@orca.step()
def county_seasonal_transition(year, county_seasonal_controls, seasonal_households):
    """
    Transition seasonal households uisng county-level controls.

    """
    run_seasonal_transition(year, county_seasonal_controls, seasonal_households, True)


@orca.step()
def msa_seasonal_transition(year, msa_seasonal_controls, seasonal_households):
    """
    Transition seasonal households using MSA-level controls.

    """
    run_seasonal_transition(year, msa_seasonal_controls, seasonal_households)


##################################
# group quarters transition
##################################


def run_gq_transition(year, controls_wrap, gq_wrap, gq_events_wrap, bldgs_wrap, by_county=True,):
    """
    Executes a group quarters transtion.

    Note: Because we just grow GQ in place right now,  just keep the
    same `building_id`  as was transition from and this takes care
    of location choice as well.

    **THIS ONE ADDS A DORM FOR OTTAWA UNIV in Surprise in 2022.****

    """
    controls = controls_wrap.local
    if by_county:
        gq = gq_wrap.to_frame(gq_wrap.local_columns + ['county'])
    else:
        gq = gq_wrap.local

    # run the transtion
    t = TabularTotalsTransition(controls, 'gq_population')
    tm = TransitionModel(t)
    updated, added, _ = tm.transition(gq, year)

    # handle events
    events = gq_events_wrap.local.query("year == {}".format(year))

    # loop through the events
    if len(events) > 0:
        bldgs = bldgs_wrap.local
        start_id = bldgs.index.max() + 1
        gq_to_sample = updated.copy()
        sampled_agents = []
        new_bldgs = {}

        for idx, row in events.iterrows():
            start_id += 1

            # create a new building
            new_bldgs[start_id] = {
                'building_type_name': 'gq',
                'parcel_id': row['parcel_id'],
                'year_built': year,
                'project_id': 'gq event',
                'project_idx': np.nan,

                # just fill in some defaults
                'non_residential_sqft': 0,
                'residential_sqft': 0,
                'residential_units': 0,
                'total_fcv': 1,
                'transient_hh_in_hh': 0,
                'transient_hh_in_hotels': 0,
                'transient_pop_in_hh': 0,
                'transient_pop_in_hotels': 0,
                'sqft_per_job': 0,
                'average_value_per_unit': 0
            }

            # randomly sample agents to fill the buildings
            if by_county:
                target_agents = gq_to_sample.query(
                    "gq_type == '{}' and county == '{}'".format(row['gq_type'], row['county']))
            else:
                target_agents = gq_to_sample.query(
                    "gq_type == '{}'".format(row['gq_type']))

            sampled = np.random.choice(
                target_agents.index.values,
                size=row['pop'],
                replace=False
            )
            gq_to_sample.drop(sampled, inplace=True)
            sampled_agents.append(
                pd.Series(np.repeat(start_id, len(sampled)), index=pd.Index(sampled)))

        # apply changes to buildings
        new_bldgs = pd.DataFrame.from_dict(new_bldgs, orient='index')
        new_bldgs.index.name = 'building_id'
        new_bldgs['project_idx'] = np.nan

        orca.add_table(
            'buildings',
            pd.concat([bldgs, new_bldgs[bldgs.columns]])
        )

        # apply changes to agents
        sampled_agents = pd.concat(sampled_agents)
        updated.loc[sampled_agents.index, 'building_id'] = sampled_agents

    # update the environment
    updated.loc[added, 'year_added'] = year
    if by_county:
        updated.drop('county', axis=1, inplace=True)
    orca.add_table('gq_persons', updated)


@orca.step()
def county_gq_transition(year, county_gq_controls, gq_persons, gq_events, buildings):
    """
    Transtion group quarters population by county and type.

    """
    run_gq_transition(year, county_gq_controls, gq_persons, gq_events, buildings)


@orca.step()
def msa_gq_transition(year, msa_gq_controls, gq_persons, gq_events, buildings):
    """
    Transtion group quarters population at the MSA-level by type.

    """
    run_gq_transition(year, msa_gq_controls, gq_persons, gq_events, buildings, False)


##########################
# transient allocation
##########################

orca.add_injectable('transient_pph', 2.8)


def run_transient_in_hotels(year, controls, buildings, transient_pph, by_county=True):
    """
    Allocate transient households and population to hotels based on hotel
    square footage.

    """
    c = controls.local.loc[(year, 'transient_hh_in_hotels')]

    if by_county:
        bldgs = buildings.to_frame(['building_type_name', 'non_residential_sqft', 'county'])
        hot = bldgs.query("building_type_name == 'hot'")
        trans_hh = get_segmented_allocation(
            c['transient_hh'],
            hot['non_residential_sqft'],
            hot['county']
        )
    else:
        bldgs = buildings.local[['building_type_name', 'non_residential_sqft']]
        hot = bldgs.query("building_type_name == 'hot'")
        trans_hh = get_simple_allocation(c['transient_hh'], hot['non_residential_sqft'])

    trans_hh = trans_hh.reindex(bldgs.index).fillna(0)
    trans_pers = stochastic_round(trans_hh * transient_pph)
    buildings.update_col('transient_hh_in_hotels', trans_hh)
    buildings.update_col('transient_pop_in_hotels', trans_pers)


def run_transient_in_hh(year, controls, buildings, households, transient_pph, by_county=True):
    """
    Allocate transient households and population to households.

    *** TODO: bring in air bnb data to inform this distribution and allow transients to
    go into vacant and seasonal units. ****

    """
    c = controls.local.loc[(year, 'transient_hh_in_hh')]

    if by_county:
        hh = households.to_frame(['building_id', 'county'])
        sampled = []
        for idx, val in c['transient_hh'].items():
            in_county = hh['county'] == idx
            curr_hh = hh[in_county]
            sampled.append(np.random.choice(curr_hh['building_id'].values, val, replace=False))
        sampled = np.concatenate(sampled)

    else:
        sampled = np.random.choice(households['building_id'].values, c['transient_hh'], replace=False)

    trans_hh = pd.Series(sampled).value_counts().reindex(buildings.index).fillna(0)
    trans_pers = stochastic_round(trans_hh * transient_pph)
    buildings.update_col('transient_hh_in_hh', trans_hh)
    buildings.update_col('transient_pop_in_hh', trans_pers)


@orca.step()
def county_transient(year, county_transient_controls, buildings, households, transient_pph):
    """
    Allocate transient households and population using county-level controls.

    """
    run_transient_in_hotels(
        year, county_transient_controls, buildings, transient_pph)
    run_transient_in_hh(
        year, county_transient_controls, buildings, households, transient_pph)


@orca.step()
def msa_transient(year, msa_transient_controls, buildings, households, transient_pph):
    """
    Allocate transient households and population at the MSA-level.

    """
    run_transient_in_hotels(
        year, msa_transient_controls, buildings, transient_pph, False)
    run_transient_in_hh(
        year, msa_transient_controls, buildings, households, transient_pph, False)


#################
# cache clearing
#################


def clear_hh_vars():
    """
    Invalidates caches related to household changes.

    Note: right now this isn't updating zonal/access variables, I think we
    just want to do this at the start of the iteration?

    """
    orca.clear_columns('households')
    orca.clear_columns('persons')
    orca.clear_columns(
        'buildings',
        [
            'res_hh',
            'total_hh',
            'vac_res_units'
        ]
    )


def clear_seas_vars():
    """
    Invalidates caches related to seasonal household
    changes.

    """
    orca.clear_columns('seasonal_households')
    orca.clear_columns(
        'buildings',
        [
            'seas_hh',
            'total_hh',
            'vac_res_units'
        ]
    )


#################
# hh re-location
#################


orca.add_injectable('simple_hh_relocation_pct', 0.05)


@orca.step()
def simple_hh_relocation(households, simple_hh_relocation_pct):
    """
    Randomly choose n% of households to un-locate.

    """

    # randomly choose n% of the households and remove their building ID
    num_to_relocate = np.round(len(households) * simple_hh_relocation_pct).astype(int)
    ran_idx = np.random.choice(households.index, num_to_relocate, replace=False)
    households.local.loc[ran_idx, 'building_id'] = -1

    # make sure we invalidate caches so we free up the space
    clear_hh_vars()



######################
# hh locaction choice
######################


@orca.injectable(cache=True)
def hlcm(config_root):
    """
    Configured location choice model.

    """
    yaml_src = r'{}//Estimation//MCPC//HLCM//hlcm_full_segs.yaml'.format(config_root)
    return MnlChoiceModel.from_yaml(str_or_buffer=yaml_src)



@orca.step()
def run_msa_hlcm(year, hlcm, households, buildings):
    """
    Runs the hlcm at the MSA-level.

    """

    # get data frames
    # for cols that exist in both, take from buildings
    cols_used = hlcm.columns_used()
    bldgs = buildings.to_frame(cols_used)
    hh_cols_needed = list(
        set(['building_id']).union(set(cols_used) - set(bldgs.columns)))
    hh = households.to_frame(hh_cols_needed)

    # run the prediction
    # note: right now we're dropping nulls
    choices, new_cap, samples = hlcm.predict(hh, bldgs.dropna(), debug=True)

    # update the environment
    orca.add_injectable('hlcm_samples', samples)
    orca.add_injectable('hlcm_choices', choices)
    households.local.loc[choices.index, 'building_id'] = choices.astype(int)
    clear_hh_vars()


@orca.step()
def run_county_hlcm(year, hlcm, households, buildings):
    """
    Runs the hlcm at the County-level.

    """

    # get data frames
    # for cols that exist in both, take from buildings
    hlcm.predict_sampling_segmentation_col = 'county'
    hlcm.predict_sampling_within_percent = 1
    hlcm.predict_sampling_within_segments = {'MC': 1, 'PC': 1}
    cols_used = hlcm.columns_used()
    bldgs = buildings.to_frame(cols_used + ['county'])
    hh_cols_needed = list(
        set(['building_id', 'src_building_id']).union(set(cols_used) - set(bldgs.columns)))
    hh = households.to_frame(hh_cols_needed)
    hh['county'] = broadcast(buildings['county'], hh['src_building_id'])

    # run the prediction
    # note: right now we're dropping nulls
    choices, new_cap, samples = hlcm.predict(hh, bldgs.dropna(), debug=True)

    # update the environment
    orca.add_injectable('hlcm_samples', samples)
    orca.add_injectable('hlcm_choices', choices)
    households.local.loc[choices.index, 'building_id'] = choices.astype(int)
    clear_hh_vars()


#######################################
# seasonal households location choice
#######################################


@orca.step()
def county_seasonal_lcm(year, seasonal_households, buildings):
    """
    Locates seasonal households to vacant residential units.

    """
    # get households
    s = seasonal_households.local
    b = buildings.to_frame(['building_id', 'residential_units', 'vac_res_units', 'county'])
    all_res_bldgs = b[b['residential_units'] > 0]

    # retain the county on the new emp and get nsb
    s['county'] = broadcast(buildings['county'], s['src_building_id'])

    for county in ['MC', 'PC']:

        # get current households to allocate
        to_locate = s.query(
            "building_id == -1 and county == '{}'".format(county))

        # current buildings to allocate to in the county
        res_bldgs = all_res_bldgs.query("county == '{}'".format(county))
        # "explode" residential units
        vac_units = res_bldgs.index.repeat(res_bldgs.vac_res_units.astype(int))

        # make the choice
        s['building_id'].loc[to_locate.index] = np.random.choice(
            vac_units, len(to_locate), replace=False).astype(int)

    # tidy up
    # TODO -- see about keeping the county on as a permanent variable?
    s.drop(['county'], axis=1, inplace=True)
    clear_seas_vars()


@orca.step()
def msa_seasonal_lcm(year, seasonal_households, buildings):
    """
    MSA level allocation of seasonal households.

    """
    # get data items
    s = seasonal_households.local
    b = buildings.to_frame(['building_id', 'residential_units', 'vac_res_units'])
    res_bldgs = b[b['residential_units'] > 0]

    # make the choice
    to_locate = s.query('building_id == -1')
    vac_units = res_bldgs.index.repeat(res_bldgs.vac_res_units.astype(int))
    choices = np.random.choice(vac_units, len(to_locate), replace=False).astype(int)

    # update the environment
    s['building_id'].loc[to_locate.index] = choices
    clear_seas_vars()


########################################################
# ad-hoc updates
# TODO: move this into a proper events table framework.
########################################################


@orca.step()
def srp_update(year, buildings, households, seasonal_households):
    """
    Based on Salt River Pima comments, remove a couple mobile homes.

    This should run before the hlcm.
    """
    # key is the year, value is the parcel id
    srp_updates = {
        2019: 1357996,  # taz 1313
        2024: 183230    # taz 1307
    }

    if year in srp_updates.keys():
        par_id = srp_updates[year]

        # remove residential units
        b = buildings.local
        b_in = b['parcel_id'] == par_id
        b.loc[b_in, 'residential_units'] = 0

        # un-locate the households
        h = households.local
        bldg_ids = b[b_in].index
        h_in = h['building_id'].isin(bldg_ids)
        h.loc[h_in, 'building_id'] = -1

        # un-locate seasonal households
        s = seasonal_households.local
        s_in = s['building_id'].isin(bldg_ids)
        s.loc[s_in, 'building_id'] = -1

        orca.clear_columns(
            'buildings',
            ['res_hh', 'seas_hh', 'total_hh', 'vac_res_units']
        )